import os
import sys

# Configure project root for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import glob
import time
import copy
import tqdm
import torch
import numpy as np
import pickle as pkl
from absl import app, flags
from natsort import natsorted
from typing import Dict, Any
from gymnasium.wrappers import RecordEpisodeStatistics
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore
from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment from CONFIG_MAPPING.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Run as the learner node.")
flags.DEFINE_boolean("actor", False, "Run as an actor node.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("checkpoint_path", None, "Path for saving/loading checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step of checkpoint to evaluate.")
flags.DEFINE_integer("eval_n_trajs", 10, "Number of evaluation trajectories.")
flags.DEFINE_boolean("save_video", False, "Save evaluation videos.")
flags.DEFINE_boolean("debug", False, "Enable debug mode.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_green(x):
    print("\033[92m {}\033[00m".format(x))


def state_dict_to_numpy(state_dict: Dict) -> Dict:
    """Convert torch state_dict to numpy for ZMQ transmission."""
    result = {}
    for k, v in state_dict.items():
        if "optimizer" in k or k == "config":
            continue
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().cpu().numpy()
        elif isinstance(v, dict):
            result[k] = state_dict_to_numpy(v)
    return result


def numpy_to_state_dict(params: Dict, device: torch.device) -> Dict:
    """Convert received numpy params back to torch tensors."""
    result = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.as_tensor(v, device=device)
        elif isinstance(v, dict):
            result[k] = numpy_to_state_dict(v, device)
        else:
            result[k] = v
    return result


def actor(agent: SACAgent, data_store, intvn_data_store, env):
    """Actor loop for environmental interaction and data collection."""
    agent.eval()

    # Evaluation mode
    if FLAGS.eval_checkpoint_step:
        ckpt = os.path.join(FLAGS.checkpoint_path, f"checkpoint_{FLAGS.eval_checkpoint_step}.pt")
        agent.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        success_counter = 0
        for i in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            while not done:
                # Use deterministic actions for evaluation
                actions = agent.sample_actions(obs, argmax=True)
                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().numpy()
                obs, reward, done, truncated, _ = env.step(actions)
                if done or truncated:
                    success_counter += reward
                    print(f"Eval Traj {i+1} | Success: {reward}")
        print_green(f"Final Success Rate: {success_counter / FLAGS.eval_n_trajs}")
        return

    # Training Actor mode
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores={"actor_env": data_store, "actor_env_intvn": intvn_data_store},
        wait_for_server=True,
    )
    client.recv_network_callback(lambda p: agent.load_state_dict(numpy_to_state_dict(p, DEVICE), strict=False))

    obs, _ = env.reset()
    timer, transitions, demo_transitions = Timer(), [], []
    running_return, intervention_count, intervention_steps = 0.0, 0, 0
    already_intervened = False

    pbar = tqdm.tqdm(range(config.max_steps), dynamic_ncols=True, desc="Actor")
    for step in pbar:
        timer.tick("total")
        # Action sampling
        if step < config.random_steps:
            actions = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = {k: torch.as_tensor(v, device=DEVICE).unsqueeze(0) for k, v in obs.items()}
                actions = agent.sample_actions(obs_t, argmax=False).squeeze(0).cpu().numpy()

        # Environment step
        next_obs, reward, done, truncated, info = env.step(actions)

        # Human-In-The-Loop intervention tracking
        if "intervene_action" in info:
            actions = info["intervene_action"]
            intervention_steps += 1
            if not already_intervened:
                intervention_count += 1
            already_intervened = True
        else:
            already_intervened = False

        # Store transition
        transition = dict(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            rewards=reward,
            masks=1.0 - float(done),
            dones=done or truncated,
        )
        data_store.insert(transition)
        transitions.append(copy.deepcopy(transition))
        if already_intervened:
            intvn_data_store.insert(transition)
            demo_transitions.append(copy.deepcopy(transition))

        running_return += reward
        obs = next_obs

        # Episode termination
        if done or truncated:
            info["episode"].update({"intervention_count": intervention_count, "intervention_steps": intervention_steps})
            client.request("send-stats", {"environment": info})
            pbar.set_description(f"Return: {running_return:.2f}")
            running_return, intervention_count, intervention_steps, already_intervened = 0.0, 0, 0, False
            client.update()  # Sync weights and send data
            obs, _ = env.reset()

        # Periodic buffer persistence
        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            for name, data in [("buffer", transitions), ("demo_buffer", demo_transitions)]:
                path = os.path.join(FLAGS.checkpoint_path, name)
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(data, f)
            transitions, demo_transitions = [], []

        timer.tock("total")
        if step % config.log_period == 0:
            client.request("send-stats", {"timer": timer.get_average_times()})


def learner(agent: SACAgent, replay_buffer, demo_buffer, wandb_logger=None):
    """Learner loop for RLPD network updates."""
    agent.train()
    # Fixed: Use state dict to avoid NameError: step in callback
    state = {"step": 0}

    # Resume training check
    if FLAGS.checkpoint_path:
        ckpts = glob.glob(os.path.join(FLAGS.checkpoint_path, "*.pt"))
        if ckpts:
            latest = natsorted(ckpts)[-1]
            state["step"] = int(os.path.basename(latest).split("_")[1].split(".")[0]) + 1

    def stats_callback(type: str, payload: dict) -> dict:
        if wandb_logger:
            wandb_logger.log(payload, step=state["step"])
        return {}

    # Initialize distributed server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Wait for buffer warm-up
    pbar = tqdm.tqdm(total=config.training_starts, initial=len(replay_buffer), desc="Warm-up")
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.close()

    # Publish initial weights
    server.publish_network(state_dict_to_numpy(agent.state_dict()))
    print_green("Learner: Initial network published.")

    # Data iterators for RLPD (50/50 split)
    replay_iter = replay_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs_and_next_obs": True}
    )
    demo_iter = demo_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs_and_next_obs": True}
    )

    timer = Timer()
    for current_step in tqdm.tqdm(range(state["step"], config.max_steps), dynamic_ncols=True, desc="Learner"):
        state["step"] = current_step

        # RLPD Update with CTA Ratio
        for _ in range(config.cta_ratio - 1):
            batch = concat_batches(next(replay_iter), next(demo_iter), axis=0)
            agent.update(batch, networks_to_update=frozenset({"critic"}))

        batch = concat_batches(next(replay_iter), next(demo_iter), axis=0)
        update_info = agent.update(batch, networks_to_update=frozenset({"critic", "actor", "temperature"}))

        # Weight synchronization (Optimal steps_per_update=5)
        if current_step % config.steps_per_update == 0:
            server.publish_network(state_dict_to_numpy(agent.state_dict()))

        # Logging
        if current_step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=current_step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=current_step)

        # Checkpointing (Optimal checkpoint_period=2500)
        if current_step > 0 and current_step % config.checkpoint_period == 0:
            path = os.path.join(FLAGS.checkpoint_path, f"checkpoint_{current_step}.pt")
            torch.save(agent.state_dict(), path)
            print_green(f"Saved checkpoint: {path}")


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    # Seeding
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    # Env setup
    env = RecordEpisodeStatistics(
        config.get_environment(fake_env=FLAGS.learner, save_video=FLAGS.save_video, classifier=True)
    )

    # Agent setup with optimized [512, 512] ReLU architecture
    agent = make_sac_pixel_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=torch.from_numpy(env.action_space.sample()).to(DEVICE),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        device=str(DEVICE),
        image_size=config.image_size,
    )

    if FLAGS.checkpoint_path and not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path, exist_ok=True)

    if FLAGS.checkpoint_path:
        ckpts = glob.glob(os.path.join(FLAGS.checkpoint_path, "*.pt"))
        if ckpts:
            latest = natsorted(ckpts)[-1]
            agent.load_state_dict(torch.load(latest, map_location=DEVICE))
            print_green(f"Loaded: {latest}")

    if FLAGS.learner:
        # Buffer setup (Optimal capacity=50000)
        common_args = dict(capacity=config.replay_buffer_capacity, image_keys=config.image_keys, device="cpu")
        replay_buffer = MemoryEfficientReplayBufferDataStore(env.observation_space, env.action_space, **common_args)
        demo_buffer = MemoryEfficientReplayBufferDataStore(env.observation_space, env.action_space, **common_args)

        logger = make_wandb_logger(project="hil-serl-pytorch", description=FLAGS.exp_name, debug=FLAGS.debug)

        # Load expert demonstrations
        if config.demo_path and os.path.exists(config.demo_path):
            for d in os.listdir(config.demo_path):
                with open(os.path.join(config.demo_path, d), "rb") as f:
                    for tx in pkl.load(f):
                        demo_buffer.insert(tx)
            print_green(f"Demo buffer size: {len(demo_buffer)}")

        learner(agent, replay_buffer, demo_buffer, wandb_logger=logger)

    elif FLAGS.actor:
        actor(agent, QueuedDataStore(50000), QueuedDataStore(50000), env)


if __name__ == "__main__":
    app.run(main)

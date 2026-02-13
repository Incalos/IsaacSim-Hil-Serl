import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import glob
import time
import torch
import numpy as np
import tqdm
from absl import app, flags
import copy
import pickle as pkl
from gymnasium.wrappers import RecordEpisodeStatistics
from natsort import natsorted

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

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("port_number", 5120, "Port number of the learner.")
flags.DEFINE_integer("broadcast_port", 5121, "Broadcast port of the learner.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("debug", False, "Debug mode.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def state_dict_to_numpy(state_dict, skip_optimizer=True):
    """Recursively convert a nested state dict to numpy arrays for network publishing.

    Args:
        state_dict: The state dict to convert
        skip_optimizer: If True, skip optimizer-related keys (they contain lists that can't be converted)
    """
    result = {}
    for k, v in state_dict.items():
        # Skip optimizer states - they have complex structures with lists
        if skip_optimizer and "optimizer" in k:
            continue
        # Skip config
        if k == "config":
            continue

        if isinstance(v, torch.Tensor):
            result[k] = v.detach().cpu().numpy()
        elif isinstance(v, dict):
            result[k] = state_dict_to_numpy(v, skip_optimizer=False)
        else:
            # Skip non-tensor items
            continue
    return result


def numpy_to_state_dict(params, device):
    """Recursively convert numpy arrays back to torch tensors."""
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
    """Actor loop: interact with env, collect transitions, and send them to the learner."""
    agent.eval()

    # Eval mode: load checkpoint at given step, run eval_n_trajs trajectories, report success rate and mean time.
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []
        ckpt_path = os.path.join(FLAGS.checkpoint_path, f"checkpoint_{FLAGS.eval_checkpoint_step}.pt")
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        agent.load_state_dict(state_dict)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                actions = agent.sample_actions(obs, argmax=False)
                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().numpy()
                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs
                if done:
                    if reward:
                        time_list.append(time.time() - start_time)
                    success_counter += reward
                    print(f"Reward: {reward} | {success_counter}/{episode + 1}")

        print(f"Success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"Average time: {np.mean(time_list) if time_list else 0}")
        return

    # Training mode: resume step from checkpoint (if buffer was saved), then connect to learner.
    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        buffer_files = glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
        if buffer_files:
            start_step = int(os.path.basename(natsorted(buffer_files)[-1])[12:-4]) + 1

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(port_number=FLAGS.port_number, broadcast_port=FLAGS.broadcast_port),
        data_stores=datastore_dict,
        wait_for_server=True,
    )

    def update_params(params):
        # state_dict = numpy_to_state_dict(params, DEVICE)
        # agent.load_state_dict(state_dict, strict=False)
        old_params = state_dict_to_numpy(agent.state_dict())

        def _diff_norm(old_p, new_p):
            total = 0.0
            for k in new_p:
                if k in old_p:
                    if isinstance(new_p[k], np.ndarray):
                        total += float(np.sum((new_p[k] - old_p[k]) ** 2))
                    elif isinstance(new_p[k], dict):
                        total += _diff_norm(old_p[k], new_p[k])
            return np.sqrt(total)

        diff = _diff_norm(old_params, params)
        print(f"[Weight Check] L2 diff between old and new params: {diff:.6e}")

        state_dict = numpy_to_state_dict(params, DEVICE)
        agent.load_state_dict(state_dict, strict=False)

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []
    obs, _ = env.reset()
    done = False
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = {k: torch.as_tensor(v, device=DEVICE) for k, v in obs.items()}
                    actions = agent.sample_actions(observations=obs_tensor, argmax=False)
                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().numpy()

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            for k in ["left", "right"]:
                info.pop(k, None)

            # HIL: if human intervened this step, use intervene_action and update intervention stats.
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False
            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done or truncated,
            )
            
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                client.request("send-stats", {"environment": info})
                pbar.set_description(f"last return: {running_return}")
                running_return, intervention_count, intervention_steps = 0.0, 0, 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()

        # Persist current transitions and demo_transitions to checkpoint dir every buffer_period.
        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            os.makedirs(buffer_path, exist_ok=True)
            os.makedirs(demo_buffer_path, exist_ok=True)
            for p, data in [(buffer_path, transitions), (demo_buffer_path, demo_transitions)]:
                with open(os.path.join(p, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(data, f)
            transitions, demo_transitions = [], []

        timer.tock("total")
        if step % config.log_period == 0:
            client.request("send-stats", {"timer": timer.get_average_times()})


def learner(agent: SACAgent, replay_buffer, demo_buffer, wandb_logger=None):
    """Learner loop: wait for buffer to fill, run SAC with RLPD (50/50) sampling, publish weights and save checkpoints periodically."""
    agent.train()

    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        ckpts = glob.glob(os.path.join(FLAGS.checkpoint_path, "*.pt"))
        if ckpts:
            latest_ckpt = natsorted(ckpts)[-1]
            start_step = int(os.path.basename(latest_ckpt).split("_")[1].split(".")[0]) + 1
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    server = TrainerServer(
        make_trainer_config(port_number=FLAGS.port_number, broadcast_port=FLAGS.broadcast_port),
        request_callback=stats_callback,
    )
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(total=config.training_starts, initial=len(replay_buffer), desc="Buffer filling")
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(state_dict_to_numpy(agent.state_dict()))
    print_green("Sent initial network to actor")

    # RLPD: sample batch_size//2 from replay and demo each, concat into one batch.
    replay_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs_and_next_obs": True}
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs_and_next_obs": True}
    )

    timer = Timer()
    train_networks = frozenset({"critic", "actor", "temperature"})
    critic_only = frozenset({"critic"})

    for step in tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"):
        # CTA: update only critic for first cta_ratio-1 steps, then full network update.
        for _ in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = concat_batches(next(replay_iterator), next(demo_iterator), axis=0)
            with timer.context("train_critics"):
                agent.update(batch, networks_to_update=critic_only)

        with timer.context("train"):
            batch = concat_batches(next(replay_iterator), next(demo_iterator), axis=0)
            update_info = agent.update(batch, networks_to_update=train_networks)

        if step > 0 and step % config.steps_per_update == 0:
            torch.cuda.synchronize()
            with torch.no_grad():
                state_dict = agent.state_dict()
                numpy_params = state_dict_to_numpy(state_dict)
            server.publish_network(numpy_params)
            del state_dict, numpy_params
            torch.cuda.empty_cache()

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0:
            path = os.path.join(FLAGS.checkpoint_path, f"checkpoint_{step}.pt")
            with torch.no_grad():
                torch.save({"step": step, "model_state_dict": agent.state_dict()}, path)
            print_green(f"Saved checkpoint to {path}")
            torch.cuda.empty_cache()
            torch.save(agent.state_dict(), path)


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    env = config.get_environment(fake_env=FLAGS.learner, save_video=FLAGS.save_video, classifier=True)
    env = RecordEpisodeStatistics(env)

    agent = make_sac_pixel_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=torch.from_numpy(env.action_space.sample()).to(DEVICE),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        device=str(DEVICE),
    )
    agent = agent.to(DEVICE)

    if FLAGS.checkpoint_path and not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path, exist_ok=True)

    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        ckpts = glob.glob(os.path.join(FLAGS.checkpoint_path, "*.pt"))
        if ckpts:
            latest = natsorted(ckpts)[-1]
            input(f"Found {latest}. Press Enter to resume.")
            agent.load_state_dict(torch.load(latest, map_location=DEVICE))
            print_green(f"Loaded checkpoint: {latest}")

    if FLAGS.learner:
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            device="cpu",
        )
        wandb_logger = make_wandb_logger(
            project="hil-serl-pytorch",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            device="cpu",
        )

        if config.demo_path:
            for d_path in os.listdir(config.demo_path):
                with open(os.path.join(config.demo_path, d_path), "rb") as f:
                    transitions = pkl.load(f)
                    for tx in transitions:
                        if "infos" in tx and "grasp_penalty" in tx["infos"]:
                            tx["grasp_penalty"] = tx["infos"]["grasp_penalty"]
                        demo_buffer.insert(tx)
            print_green(f"Demo buffer size: {len(demo_buffer)}")

        if FLAGS.checkpoint_path:
            for buf_type, store in [("buffer", replay_buffer), ("demo_buffer", demo_buffer)]:
                path = os.path.join(FLAGS.checkpoint_path, buf_type)
                if os.path.exists(path):
                    for f_path in glob.glob(os.path.join(path, "*.pkl")):
                        with open(f_path, "rb") as f:
                            for tx in pkl.load(f):
                                store.insert(tx)
            print_green(f"Replay buffer: {len(replay_buffer)} | Demo buffer: {len(demo_buffer)}")

        learner(agent, replay_buffer, demo_buffer, wandb_logger=None)

    elif FLAGS.actor:
        data_store = QueuedDataStore(20000)
        intvn_data_store = QueuedDataStore(20000)
        actor(agent, data_store, intvn_data_store, env)

    else:
        raise ValueError("Specify --actor or --learner")


if __name__ == "__main__":
    app.run(main)

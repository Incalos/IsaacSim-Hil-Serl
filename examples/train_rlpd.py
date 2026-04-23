import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.append(_project_root)

import glob
import time
import copy
import tqdm
import torch
import numpy as np
import pickle as pkl
from typing import Dict, Any
from absl import app, flags
from serl_launcher.agents.sac import SACAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.utils.logging_utils import RecordEpisodeStatistics
from serl_launcher.utils.launcher import make_sac_pixel_agent, make_wandb_logger, make_trainer_config
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from experiments.mappings import CONFIG_MAPPING
from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment from CONFIG_MAPPING.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Run as the learner node.")
flags.DEFINE_boolean("actor", False, "Run as an actor node.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("port_number", 6006, "Port number for stats communication.")
flags.DEFINE_integer("broadcast_port", 6007, "Broadcast port for stats communication.")
flags.DEFINE_string("checkpoint_path", "checkpoints", "Path for saving/loading checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step of checkpoint to evaluate.")
flags.DEFINE_integer("eval_n_trajs", 10, "Number of evaluation trajectories.")
flags.DEFINE_boolean("debug", False, "Enable debug mode.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = None
PARAMS = None
CHECKPOINT_PATH = None


def print_green(x: str) -> None:
    """Print text in green color.
    
    Args:
        x: Text to print
    """
    print("\033[92m {}\033[00m".format(x))


def state_dict_to_numpy(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert torch tensors in state dict to numpy arrays for ZMQ transmission.
    
    Args:
        state_dict: PyTorch model state dictionary
        
    Returns:
        State dictionary with numpy arrays instead of tensors
    """
    result = {}
    for k, v in state_dict.items():
        if "optimizer" in k or k == "config":
            continue
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().cpu().numpy()
        elif isinstance(v, dict):
            result[k] = state_dict_to_numpy(v)
    return result


def numpy_to_state_dict(params: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Convert numpy arrays back to torch tensors for agent loading.
    
    Args:
        params: State dictionary with numpy arrays
        device: Target compute device (CPU/GPU)
        
    Returns:
        State dictionary with PyTorch tensors
    """
    result = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.as_tensor(v, device=device)
        elif isinstance(v, dict):
            result[k] = numpy_to_state_dict(v, device)
        else:
            result[k] = v
    return result


def actor(agent: SACAgent, data_store: QueuedDataStore, intvn_data_store: QueuedDataStore, env: Any) -> None:
    """Actor node logic for data collection and policy execution.
    
    Args:
        agent: SAC agent instance
        data_store: Data store for regular transitions
        intvn_data_store: Data store for intervention transitions
        env: Environment instance
    """
    agent.eval()

    if FLAGS.eval_checkpoint_step:
        ckpt = os.path.join(CHECKPOINT_PATH, f"checkpoint_{FLAGS.eval_checkpoint_step}.pt")
        agent.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state_dict"])
        success_counter = 0
        for i in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            while not done:
                actions = agent.sample_actions(obs, argmax=True)
                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().numpy()
                obs, reward, done, truncated, _ = env.step(actions)
                if done or truncated:
                    success_counter += reward
                    print(f"Evaluation Trajectory {i+1} | Success: {reward}")
        print_green(f"Final Success Rate: {success_counter / FLAGS.eval_n_trajs}")
        return

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(port_number=FLAGS.port_number, broadcast_port=FLAGS.broadcast_port),
        data_stores={
            "actor_env": data_store,
            "actor_env_intvn": intvn_data_store
        },
        wait_for_server=True,
        timeout_ms=3000,
    )

    def update_params(params: Dict[str, Any]) -> None:
        """Update agent parameters from server payload.
        
        Args:
            params: Numpy-based state dictionary from server
        """
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

    pbar = tqdm.tqdm(range(PARAMS["max_steps"]), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < PARAMS["random_steps"]:
                actions = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = {k: torch.as_tensor(v, device=DEVICE) for k, v in obs.items()}
                    actions = agent.sample_actions(
                        observations=obs_tensor,
                        argmax=False,
                    )
                actions = actions.cpu().numpy()

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            reward = np.asarray(reward, dtype=np.float32)

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
                dones=done,
            )
            if "grasp_penalty" in info:
                transition["grasp_penalty"] = info["grasp_penalty"]

            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs

            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                stats = {"environment": info}
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()
                print_green("Environment has been reset")

        if step > 0 and PARAMS["buffer_period"] > 0 and step % PARAMS["buffer_period"] == 0:
            buffer_path = os.path.join(CHECKPOINT_PATH, "buffer")
            demo_buffer_path = os.path.join(CHECKPOINT_PATH, "demo_buffer")
            os.makedirs(buffer_path, exist_ok=True)
            os.makedirs(demo_buffer_path, exist_ok=True)

            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % PARAMS["log_period"] == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


def learner(agent: SACAgent, replay_buffer: MemoryEfficientReplayBufferDataStore, demo_buffer: MemoryEfficientReplayBufferDataStore) -> None:
    """Learner node logic for policy training and parameter broadcasting.
    
    Args:
        agent: SAC agent instance
        replay_buffer: Replay buffer for regular transitions
        demo_buffer: Replay buffer for demonstration/intervention transitions
    """
    agent.train()
    update_steps = 0

    wandb_logger = make_wandb_logger(project="isaacsim-hil-serl", description=FLAGS.exp_name, debug=FLAGS.debug)

    def stats_callback(type: str, payload: dict) -> dict:
        """Process statistics received from actor nodes.
        
        Args:
            type: Request type (should be "send-stats")
            payload: Statistics payload from actors
            
        Returns:
            Empty dictionary (required by callback interface)
        """
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}

    server = TrainerServer(
        make_trainer_config(port_number=FLAGS.port_number, broadcast_port=FLAGS.broadcast_port),
        request_callback=stats_callback,
    )
    server.register_data_store("actor_env", replay_buffer)
    if demo_buffer is not None:
        server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)
    pbar = tqdm.tqdm(
        total=PARAMS["training_starts"],
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < PARAMS["training_starts"]:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(state_dict_to_numpy(agent.state_dict()))
    print_green("Initial network parameters sent to actor nodes")

    if demo_buffer is None or len(demo_buffer) == 0:
        single_buffer_batch_size = PARAMS["batch_size"]
        demo_iterator = None
    else:
        single_buffer_batch_size = PARAMS["batch_size"] // 2
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": single_buffer_batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=DEVICE,
        )

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=DEVICE,
    )

    timer = Timer()
    pbar = tqdm.tqdm(
        total=PARAMS["replay_buffer_capacity"],
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    for step in tqdm.tqdm(range(PARAMS["max_steps"]), dynamic_ncols=True, desc="learner"):
        for critic_step in range(PARAMS["cta_ratio"] - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                if demo_iterator is not None:
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent.update(batch, networks_to_update=frozenset({"critic"}))

        with timer.context("train"):
            batch = next(replay_iterator)
            if demo_iterator is not None:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
            update_info = agent.update(batch, networks_to_update=frozenset({"actor", "critic", "temperature"}))

        if step > 0 and step % (PARAMS["steps_per_update"]) == 0:
            torch.cuda.synchronize()
            with torch.no_grad():
                state_dict = agent.state_dict()
                numpy_params = state_dict_to_numpy(state_dict)
            server.publish_network(numpy_params)
            del state_dict, numpy_params
            torch.cuda.empty_cache()

        if update_steps % PARAMS["log_period"] == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if PARAMS["checkpoint_period"] and update_steps > 0 and update_steps % PARAMS["checkpoint_period"] == 0:
            assert CHECKPOINT_PATH is not None
            os.makedirs(CHECKPOINT_PATH, exist_ok=True)
            checkpoint_file = os.path.join(CHECKPOINT_PATH, f"checkpoint_{update_steps}.pt")

            with torch.no_grad():
                torch.save(
                    {
                        "step": update_steps,
                        "model_state_dict": agent.state_dict(),
                    },
                    checkpoint_file,
                )
            print_green(f"Checkpoint saved to {checkpoint_file}")
            torch.cuda.empty_cache()

        pbar.update(len(replay_buffer) - pbar.n)
        update_steps += 1


def main(_: Any) -> None:
    """Main function to initialize and run RLPD training (learner/actor mode).
    
    Args:
        _: Unused argument from absl app
    """
    global CONFIG, PARAMS, CHECKPOINT_PATH
    CONFIG = CONFIG_MAPPING[FLAGS.exp_name]()
    PARAMS = CONFIG.robot_params
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "experiments", FLAGS.exp_name, FLAGS.checkpoint_path)

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    env = RecordEpisodeStatistics(CONFIG.get_environment(fake_env=FLAGS.learner, classifier=True))

    agent = make_sac_pixel_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=PARAMS["image_keys"],
        encoder_type=PARAMS["encoder_type"],
        discount=PARAMS["discount"],
        device=str(DEVICE),
        image_size=PARAMS["image_size"],
    )

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    agent = agent.to(DEVICE)

    if FLAGS.learner:
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=PARAMS["replay_buffer_capacity"],
            image_keys=PARAMS["image_keys"],
            include_grasp_penalty=False,
            device="cpu",
        )

        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=PARAMS["replay_buffer_capacity"],
            image_keys=PARAMS["image_keys"],
            include_grasp_penalty=False,
            device="cpu",
        )

        if CONFIG.demo_path and os.path.exists(CONFIG.demo_path):
            for d in os.listdir(CONFIG.demo_path):
                with open(os.path.join(CONFIG.demo_path, d), "rb") as f:
                    for tx in pkl.load(f):
                        demo_buffer.insert(tx)

        if CHECKPOINT_PATH:
            for buf_type, store in [("buffer", replay_buffer), ("demo_buffer", demo_buffer)]:
                path = os.path.join(CHECKPOINT_PATH, buf_type)
                if os.path.exists(path):
                    for f_path in glob.glob(os.path.join(path, "*.pkl")):
                        with open(f_path, "rb") as f:
                            for tx in pkl.load(f):
                                store.insert(tx)

        print_green(f"Replay buffer size: {len(replay_buffer)} | Demo buffer size: {len(demo_buffer)}")

        learner(agent, replay_buffer, demo_buffer)

    elif FLAGS.actor:
        actor(agent, QueuedDataStore(100000), QueuedDataStore(100000), env)


if __name__ == "__main__":
    app.run(main)



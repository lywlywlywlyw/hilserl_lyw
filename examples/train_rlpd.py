#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
# from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING
import redis
import json
FLAGS = flags.FLAGS


flags.DEFINE_string("exp_name", "blockassembly", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("cal_metric", False, "Whether to calculate metrics.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_boolean("metric", False, "Whether this is a metric calculator.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", "/home/admin01/lyw_2/hil-serl_original/data/demos/demo_2026-01-06_22-31-58_3dof.pkl", "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", "/home/admin01/lyw_2/hil-serl_original/checkpoints/serl_without_intervene_3dof", "Path to save checkpoints.")#"/home/admin01/lyw_2/hil-serl_original/checkpoints/"+time.strftime("%Y%m%d_%H%M%S", time.localtime())
flags.DEFINE_integer("eval_checkpoint_step", 18000 ,"Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 20, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", True, "Save video.")
flags.DEFINE_boolean("control_double_arm", False, "Control double arm.")
flags.DEFINE_boolean("pretrain", False, "Whether to pretrain the agent.")


flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

import signal, sys, torch

def handler(sig, frame):
    torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGINT, handler)
devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################

def cal_single_sample_metrics(agent, obs, actions, next_obs, reward, done, info, rng):
    start_time = time.time()
    transition = dict(
        observations={k: jnp.array([v]) for k, v in obs.items()},
        actions=jnp.array([actions]),
        next_observations={k: jnp.array([v]) for k, v in next_obs.items()},
        rewards=jnp.array([reward]),
        masks=jnp.array([1.0 - done]),
        dones=jnp.array([done]),
    )
    rng, next_action_sample_key = jax.random.split(rng)

    ###########from critic loss##########
    next_actions, next_actions_log_probs = agent._compute_next_actions(
        transition, next_action_sample_key
    )

    # Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)
    target_next_qs = agent.forward_target_critic(
        transition["next_observations"],
        next_actions,
        rng=rng
    )  # (critic_ensemble_size, batch_size)
    time1 = time.time()
    print("time1:", time1 - start_time)
    # Minimum Q across (subsampled) ensemble members
    target_next_min_q = target_next_qs.min(axis=0)

    target_q = (
        transition["rewards"]
        + agent.config["discount"] * transition["masks"] * target_next_min_q
    )

    predicted_qs = agent.forward_critic(
        transition["observations"], transition["actions"], rng=rng
    )
    time2 = time.time()
    print("time2:", time2 - time1)
    target_qs = target_q[None].repeat(agent.config["critic_ensemble_size"], axis=0)
    critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

    if "intervene_action" in info:
        actions = jnp.array([info["intervene_action"]])
        demo_predicted_qs = agent.forward_critic(
            transition["observations"], actions, rng=rng, train=False
        )
    else:
        demo_predicted_qs = copy.deepcopy(predicted_qs)
    time3 = time.time()
    print("time3:", time3 - time2)

    ###########from actor loss##########
    batch_size = transition["rewards"].shape[0]
    temperature = agent.forward_temperature()
    time4 = time.time()
    print("time4:", time4 - time3)
    rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
    action_distributions = agent.forward_policy(
        transition["observations"], rng=policy_rng
    )
    time5 = time.time()
    print("time5:", time5 - time4)
    actions, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)

    predicted_qs = agent.forward_critic(
        transition["observations"],
        actions,
        rng=critic_rng,
    )
    time6 = time.time()
    print("time6:", time6 - time5)
    predicted_q = predicted_qs.mean(axis=0)

    actor_objective = predicted_q - temperature * log_probs
    actor_loss = -jnp.mean(actor_objective)

    info = {
        "critic_loss": critic_loss,
        "predicted_qs": jnp.mean(predicted_qs),
        "target_qs": jnp.mean(target_qs),
        "std_td_error" : jnp.std(predicted_qs - target_qs),
        "mean_td_error" : jnp.mean(jnp.abs(predicted_qs - target_qs)),
        "max_td_error" : jnp.max(jnp.abs(predicted_qs - target_qs)),
        "ensemble_std": jnp.mean(jnp.std(predicted_qs, axis=0)),
        "ensemble_gap": jnp.mean(jnp.abs(predicted_qs.max(axis=0) - predicted_qs.min(axis=0))),
        "bootstrap_ratio": jnp.mean(jnp.where(target_q != 0, transition["rewards"] / target_q, transition["rewards"] / (target_q + 0.001))),
        "demo_Q_delta": jnp.mean(demo_predicted_qs - predicted_qs),
        "mean_target_next_min_q": jnp.mean(target_next_min_q),
        "actor_loss": actor_loss,
        "temperature": temperature,
        "entropy": -log_probs.mean(),
        "mean_std": action_distributions.distribution.stddev().mean(),
        "pred_qs": predicted_q.mean(),
        "temperature_log_probs_mean": (temperature * log_probs).mean(),
    }
    return info

def metric(agent, sampling_rng):
    wandb_logger = make_wandb_logger(
        project="hil-serl-metric",
        description=FLAGS.exp_name,
        debug=FLAGS.debug,
    )
    redis_server = redis.Redis(host='localhost', port=6379, db=0)
    
    data_store = QueuedDataStore(50000)
    datastore_dict = {
        "cal_metric": data_store}
    client = TrainerClient(
        "cal_metric",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)
    last_recv_dict = {"step": -1}
    first_recv = True
    # while True:
    #     recv = redis_server.get("latest_transition")
    #     if (recv is None):
    #         continue
    #     recv = json.loads(recv)
        
    #     if first_recv and (recv[0]["step"] != 0):
    #         continue
    #     if last_recv_dict["step"] == recv[-1]["step"]:
    #         continue
    #     for recv_dict in recv:
    #         info = cal_single_sample_metrics(agent, recv_dict["observations"], recv_dict["actions"], recv_dict["next_observations"], recv_dict["rewards"], recv_dict["dones"], recv_dict, sampling_rng)
    #         info["intervene_flag"] = jnp.array(1) if "intervene_action" in recv_dict else jnp.array(0)
    #         info["dones"] = jnp.array(1 if recv_dict["dones"] else 0)
    #         wandb_logger.log(info, step=recv_dict["step"])
    #     last_recv_dict = copy.deepcopy(recv[-1])
    #     first_recv = False
    while True:
        recv = redis_server.get("latest_transition")
        if (recv is None):
            continue
        recv_dict = json.loads(recv)
        if type(recv_dict) != dict:
            continue
        if first_recv and (recv_dict["step"] != 0):
            continue
        if last_recv_dict["step"] == recv_dict["step"]:
            continue
        info = cal_single_sample_metrics(agent, recv_dict["observations"], recv_dict["actions"], recv_dict["next_observations"], recv_dict["rewards"], recv_dict["dones"], recv_dict, sampling_rng)
        info["intervene_flag"] = jnp.array(1) if "intervene_action" in recv_dict else jnp.array(0)
        info["dones"] = jnp.array(1 if recv_dict["dones"] else 0)
        wandb_logger.log(info, step=recv_dict["step"])
        last_recv_dict = copy.deepcopy(recv_dict)
        first_recv = False

def to_serializable(obj):
    """
    递归把 dict/list/tuple 中的 ndarray 或 jnp.array 转成 list
    """
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    else:
        return obj

def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))
                # actions = np.zeros_like(actions)
                next_obs, reward, done, truncated, info = env.step(actions)
                
                print("***************reward:", reward)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit
    
    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    if FLAGS.cal_metric:
        redis_server = redis.Redis(host='localhost', port=6379, db=0)
    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    send_transition_list = []
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))
                

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")
            if FLAGS.cal_metric:
                
                send_transition = dict(
                    step=step,
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )
                if "intervene_action" in info:
                    send_transition["intervene_action"] = copy.deepcopy(info["intervene_action"])
                # send_transition_list.append(send_transition)
                send_transition = json.dumps(to_serializable(send_transition))
                redis_server.set("latest_transition", send_transition)
                
            # override the action with the intervention action
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
            
            if 'grasp_penalty' in info:
                transition['grasp_penalty']= info['grasp_penalty']
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()
                # if FLAGS.cal_metric:
                #     send_transition_list_json = json.dumps(to_serializable(send_transition_list))
                #     redis_server.set("latest_transition", send_transition_list_json)
                #     send_transition_list = []

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})
    last_agent = copy.deepcopy(agent) 
    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        # calculate KL divergence between the current and last agent
        new_batch = next(replay_iterator)
        new_demo_batch = next(demo_iterator)
        new_batch = concat_batches(new_batch, new_demo_batch, axis=0)
        
        # KL_div = agent.compute_kl_divergence(new_batch, last_agent, rng)
        # last_agent = copy.deepcopy(agent) 
        # update_info.update({"KL_div": KL_div})
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )

def pretrain(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})
    last_agent = copy.deepcopy(agent) 
    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                demo_batch = next(demo_iterator)
                batch = demo_batch

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            demo_batch = next(demo_iterator)
            batch = demo_batch
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name](FLAGS.control_double_arm)

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner|FLAGS.metric|FLAGS.pretrain, evaluate=(FLAGS.eval_checkpoint_step>0), save_video=FLAGS.save_video
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )
    elif FLAGS.pretrain:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        pretrain(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )
    elif FLAGS.metric:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        
        print_green("starting metric loop")
        metric(
            agent,
            sampling_rng
        )
    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)

import gymnasium as gym
import jax 

from experiments.block_assembly.blockassembly_env import BlockAssemblyEnv

from experiments.config import DefaultTrainingConfig

from serl_launcher.networks.reward_classifier import load_classifier_func

import os
import jax.numpy as jnp
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
class TrainConfig(DefaultTrainingConfig):
    image_keys = ['shelf', 'ground']
    proprio_keys = ['tcp_pose', 'tcp_vel', 'tcp_force']
    classifier_keys = ['shelf', 'ground']
    discount = 0.97
    buffer_period = 1000
    checkpoint_period = 2000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"
    
    def get_environment(self, fake_env=False, evaluate=False):
        env = BlockAssemblyEnv(fake_env, None, evaluate)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        
        # classifier = load_classifier_func(
        #     key=jax.random.PRNGKey(0),
        #     sample=env.observation_space.sample(),
        #     image_keys=self.classifier_keys,
        #     checkpoint_path=os.path.abspath("classifier_ckpt/"),
        # )

        # def reward_func(obs):
        #     sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
        #     return int(sigmoid(classifier(obs)) > 0.7)# and obs["state"][0, 0] > 0.4
        # env.reward_func = reward_func

        return env
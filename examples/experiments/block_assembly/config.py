import gymnasium as gym
import jax 

from experiments.block_assembly.blockassembly_env import BlockAssemblyEnv

from experiments.config import DefaultTrainingConfig

from serl_launcher.networks.reward_classifier import load_classifier_func
import os
import jax.numpy as jnp
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.reward_classifer import MultiCameraBinaryRewardClassifierWrapper

import numpy as np
class TrainConfig(DefaultTrainingConfig):
    image_keys = ['shelf', 'ground']
    proprio_keys = ['tcp_pose', 'r_force']#, 'l_force', 'r_hand_force', 'l_hand_force']#, 'tcp_vel' , 'r_force'

    classifier_keys = ['shelf', 'ground']
    discount = 0.97
    buffer_period = 1000
    checkpoint_period = 2000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"
    
    def __init__(
        self,
        control_double_arm
    ):
        super().__init__()

        if control_double_arm is not None:
            self.control_double_arm = control_double_arm
        else:
            self.control_double_arm = False
        if self.control_double_arm:
            self.proprio_keys = ['rtcp_pose', 'ltcp_pose', 'r_force', 'l_force']

    def get_environment(self, fake_env=True, evaluate=False, save_video=False, classifer=False):
        env = BlockAssemblyEnv(fake_env, self.control_double_arm, evaluate, save_video, classifer)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        classifier = load_classifier_func(
            key=jax.random.PRNGKey(0),
            sample=env.observation_space.sample(),
            image_keys=self.classifier_keys,
            checkpoint_path=os.path.abspath("classifier_ckpt/"),
        )

        def reward_func(obs):
            sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
            return int(jax.device_get(sigmoid(classifier(obs)) > 0.7)[0])# and obs["state"][0, 0] > 0.4
        
        env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
 # load policy
# from isaacgym import gymapi
# from isaacgym.torch_utils import *
import numpy as np
import torch
import time
import rtde_control
import rtde_receive
from examples.experiments.block_assembly.inspire_hand_module_modbus_lyw import Hand
import redis
from examples.experiments.block_assembly.rl_distill import *
from examples.experiments.block_assembly.rl_distill_max_likelihood import *

from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import pickle
import trimesh as tm
from pointnet2_ops import pointnet2_utils
from pynput import keyboard
import wandb
import datetime
from ur_analytic_ik import ur5e

import tkinter as tk
from tkinter import ttk
from multiprocessing import Process, Manager

# params
dof_lower_limits = [-6.2800, -6.2800, -3.1400, -6.2800, -6.2800, -6.2800,  0.0000,  0.0000, 0.0000,  0.0000,  0.0000,  0.2000, -6.2800, -6.2800, -3.1400, -6.2800, -6.2800, -6.2800,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.2000]
dof_upper_limits = [6.2800, 6.2800, 3.1400, 6.2800, 6.2800, 6.2800, 1.7000, 1.7000, 1.7000, 1.7000, 0.5000, 1.3000, 6.2800, 6.2800, 3.1400, 6.2800, 6.2800, 6.2800, 1.7000, 1.7000, 1.7000, 1.7000, 0.5000, 1.3000]

class BlockAssemblyEnv(gym.Env):
    def __init__(self, fake_env, evaluate=0, offline_train=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.hand_action_scale = 1/60*8
        self.max_episode_steps = 100#14
        self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(1, 36), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1, 6), dtype=np.float32)
        self.agent = None
        self.rng = None
        self.queue = np.zeros(50,)
        self.queue_index = 0
        self.evaluate = evaluate
        self.offline_train = offline_train
        if not fake_env:
            from examples.experiments.block_assembly.spacemouse.spacemouse_expert import SpaceMouseExpert
            self.expert = SpaceMouseExpert()
            self.episode_reward = 0
            file = open('/home/admin01/lyw_2/control_ur5/replay_record.pkl', 'rb')
            self.rh_data = np.array(pickle.load(file))
            self.steps = 0
            self.total_steps = 0
            self.robot_ip = "172.16.17.93"
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self.hand = Hand(lower_limit=dof_lower_limits[6:12], upper_limit=dof_upper_limits[6:12], port='/dev/ttyUSB0')
            self.another_robot_ip = "172.16.17.92"#"192.168.1.101"#?
            self.another_rtde_r = rtde_receive.RTDEReceiveInterface(self.another_robot_ip)
            self.another_rtde_c = rtde_control.RTDEControlInterface(self.another_robot_ip)
            self.another_hand = Hand(lower_limit=dof_lower_limits[18:24], upper_limit=dof_upper_limits[18:24], port='/dev/ttyUSB1')
            self.redis_server = redis.Redis(host='localhost', port=6379, db=0)

            self.success_flag = False

            # 拼接参数
            self.target_relative_pos = torch.tensor([0.0, 0, -0.0375], dtype=torch.float32).to(self.device)
            self.target_relative_quat = torch.tensor([0, 0, 0, 1], dtype=torch.float32).to(self.device)# unit quaternion

            self.ik_target_relative_pos0 = torch.tensor([0.0, 0, -0.0375-0.03], dtype=torch.float32).to(self.device)
            self.ik_target_relative_pos1 = torch.tensor([0.0, 0, -0.0375-0.01], dtype=torch.float32).to(self.device)
            self.ik_target_relative_pos2 = torch.tensor([0.0, 0, -0.0375+0.005], dtype=torch.float32).to(self.device)
            
            self.ik_target_relative_quat = torch.tensor([0, 0, 0, 1], dtype=torch.float32).to(self.device)# unit quaternion

            # baseur和world的关系
            arm_initial_pos = torch.tensor([-0.8625, -0.2400,  1.3946]).to(self.device)
            another_arm_initial_pos = torch.tensor([-0.8625,  0.2400,  1.3946]).to(self.device)
            arm_initial_rot = torch.tensor([ 3.3444e-06, -9.2388e-01,  3.8268e-01, -1.3853e-06]).to(self.device)
            another_arm_initial_rot = torch.tensor([-3.3444e-06, -9.2388e-01, -3.8268e-01, -1.3853e-06]).to(self.device)

            world2arm_trans_right = arm_initial_pos
            world2arm_rot_right = arm_initial_rot
            world2arm_trans_left = another_arm_initial_pos
            world2arm_rot_left = another_arm_initial_rot

            r_basemy2baseur_quat = torch.tensor([ 1.06572167e-03 ,-5.62792849e-04,  7.09452728e-01 ,-7.04751995e-01]).to(self.device)#torch.tensor([-3.60496599e-05, -5.34978704e-06, 7.07114340e-01, -7.07099221e-01]).to(self.device)
            r_basemy2baseur_pos = torch.tensor( [-0.02776959, -0.02058528,  0.01197951]).to(self.device)#torch.tensor([-0.02073179, 0.01909007, 0.00689948]).to(self.device)
            r_world2basemy_rot = world2arm_rot_right#world2arm_rot_left
            r_world2basemy_pos = world2arm_trans_right#world2arm_trans_left
            self.r_world2baseur_pos = self.quat_apply(r_world2basemy_rot, r_basemy2baseur_pos) + r_world2basemy_pos
            self.r_world2baseur_rot = self.quat_mul(r_world2basemy_rot, r_basemy2baseur_quat)
            l_basemy2baseur_quat = torch.tensor([-3.60496599e-05, -5.34978704e-06, 7.07114340e-01, -7.07099221e-01]).to(self.device)
            l_basemy2baseur_pos = torch.tensor([-0.02073179, 0.01909007, 0.00689948]).to(self.device)
            l_world2basemy_rot = world2arm_rot_left
            l_world2basemy_pos = world2arm_trans_left
            self.l_world2baseur_pos = self.quat_apply(l_world2basemy_rot, l_basemy2baseur_pos) + l_world2basemy_pos
            self.l_world2baseur_rot = self.quat_mul(l_world2basemy_rot, l_basemy2baseur_quat)

            self.rbaseur2lbaseur_pos = self.quat_apply(self.quat_conjugate(self.r_world2baseur_rot), self.l_world2baseur_pos - self.r_world2baseur_pos)
            self.rbaseur2lbaseur_rot = self.quat_mul(self.quat_conjugate(self.r_world2baseur_rot), self.l_world2baseur_rot)
            
            self.eemy2ee_pos = torch.tensor([0, 0.0993, 0]).to(self.device)
            self.eemy2ee_rot = torch.tensor([-0.707,0,0,0.707]).to(self.device)
            self.ee2eemy_pos = self.quat_apply(self.quat_conjugate(self.eemy2ee_rot), torch.zeros_like(self.eemy2ee_pos) - self.eemy2ee_pos)
            self.ee2eemy_rot = self.quat_conjugate(self.eemy2ee_rot)

            # 动作范围定义
            range_constantx = 0.05
            range_constanty = 0.05
            range_constantz = 0.05
            range_constant1 = 0.02
            # 高处未插入状态前后左右5cm的动作范围
            # self.target_eeur_action_pos_range_lower = torch.tensor([-0.0684 - range_constantx, -0.1720 - range_constanty,  1.2523 - range_constantz, -0.2242 - range_constantx,  0.2118 - range_constanty,  1.3284 - range_constantz]).to(self.device)
            # self.target_eeur_action_pos_range_upper = torch.tensor([-0.0684 + range_constantx, -0.1720 + range_constanty,  1.2523 + range_constantz, -0.2242 + range_constantx,  0.2118 + range_constanty,  1.3284 + range_constantz]).to(self.device)
            # 高处插入状态前后左右5cm的动作范围
            # self.target_eeur_action_pos_range_lower = torch.tensor([-0.0917 - range_constantx, -0.1152 - range_constanty,  1.2296 - range_constantz, -0.2242 - range_constantx,  0.2118 - range_constanty,  1.3284 - range_constantz]).to(self.device)
            # self.target_eeur_action_pos_range_upper = torch.tensor([-0.0917 + range_constantx, -0.1152 + 0.02,  1.2296 + range_constantz, -0.2242 + range_constantx,  0.2118 + range_constanty,  1.3284 + range_constantz]).to(self.device)
            # 低处插入状态前后左右5cm的动作范围
            # self.target_eeur_action_pos_range_lower = torch.tensor([-0.1728 - range_constantx, -0.1632 - range_constanty,  1.0446, -0.1567 - range_constantx,  0.1867 - 0.02,  1.0493]).to(self.device)
            # self.target_eeur_action_pos_range_upper = torch.tensor([-0.1728 + range_constantx, -0.1632 + 0.02,  1.0446 + range_constantz, -0.1567 + range_constantx,  0.1867 + range_constanty,  1.0493 + range_constantz]).to(self.device)
            
            # self.action_pos_range_lower = torch.tensor([-range_constant1, -range_constant1, -range_constant1, -range_constant1, -range_constant1, -range_constant1]).to(self.device)
            # self.action_pos_range_upper = torch.tensor([range_constant1, range_constant1, range_constant1, range_constant1, range_constant1, range_constant1]).to(self.device)
            # # self.action_rot_range_lower = torch.tensor([-0.523, -0.523, -0.523]).to(self.device)
            # # self.action_rot_range_upper = torch.tensor([0.523, 0.523, 0.523]).to(self.device)
            # self.action_rot_range_lower = torch.tensor([-0.08, -0.08, -0.08]).to(self.device)
            # self.action_rot_range_upper = torch.tensor([0.08, 0.08, 0.08]).to(self.device)
            self.target_eeur_action_pos_range_lower = torch.tensor([-0.1728 - range_constantx, -0.1632 - range_constanty,  1.0446, -0.1567 - range_constantx,  0.1867 - 0.02,  1.0493]).to(self.device)
            self.target_eeur_action_pos_range_upper = torch.tensor([-0.1728 + range_constantx, -0.1632 + 0.02,  1.0446 + range_constantz, -0.1567 + range_constantx,  0.1867 + range_constanty,  1.0493 + range_constantz]).to(self.device)
            
            #origin set before 1126
            # self.action_pos_range_lower = torch.tensor([-range_constant1, -range_constant1, -range_constant1, -range_constant1, -range_constant1, -range_constant1]).to(self.device)
            # self.action_pos_range_upper = torch.tensor([range_constant1, range_constant1, range_constant1, range_constant1, range_constant1, range_constant1]).to(self.device)
            # self.action_rot_range_lower = torch.tensor([-0.2, -0.2, -0.2]).to(self.device)
            # self.action_rot_range_upper = torch.tensor([0.2, 0.2, 0.2]).to(self.device)

            # set same as hilserl usb insertion
            self.action_pos_range_value = 0.015
            self.action_rot_range_value = 0.1
            self.action_pos_range_lower = torch.tensor([-0.015, -0.015, -0.015, -0.015, -0.015, -0.015]).to(self.device)
            self.action_pos_range_upper = torch.tensor([0.015, 0.015, 0.015, 0.015, 0.015, 0.015]).to(self.device)
            self.action_rot_range_lower = torch.tensor([-0.1, -0.1, -0.1]).to(self.device)
            self.action_rot_range_upper = torch.tensor([0.1, 0.1, 0.1]).to(self.device)
            self.init_world2eemy_quat0 = torch.tensor([ 0.2703,  0.6882, -0.2025,  0.6419]).to(self.device)
            self.tttarget_world2eemy_quat0 = torch.tensor([ 0.2744,  0.7125, -0.2559,  0.5927]).to(self.device)

            self.eemy02eemy1_limit_euler = torch.from_numpy(R.from_quat(self.quat_mul(self.quat_conjugate(self.init_world2eemy_quat0), self.tttarget_world2eemy_quat0).cpu().numpy()).as_euler("xyz")).float().to(self.device)#torch.tensor([ 1.70888554, -0.20639995, -0.23665574]).to(self.device)
            self.eemy02eemy1_limit_euler_low = torch.tensor([ 0-0.1, -0.20639995-0.1, -0.23665574-0.1]).to(self.device)
            self.eemy02eemy1_limit_euler_high = torch.tensor([ 1.70888554+0.1, 0+0.1, 0+0.1]).to(self.device)
            
            # angle difference相关
            self.center0_local = torch.tensor([0, 0, 0.02875], dtype=torch.float, device=self.device)#torch.tensor([0, 2.51952321e-2, 0], dtype=torch.float, device=self.device)#lego1x3:
            self.axes0_local = [torch.tensor([1,0,0], dtype=torch.float, device=self.device), torch.tensor([0,1,0], dtype=torch.float, device=self.device), torch.tensor([0,0,1], dtype=torch.float, device=self.device)]
            self.half_lengths0 = [0.045, 0.015, 0.01]#[0.02387213, 0.0129716, 0.06278301]#lego1x3:
            self.center1_local = torch.tensor([0, 0, -0.02875], dtype=torch.float, device=self.device)#torch.tensor([0, 1.64973424e-3, 2.67172183e-2], dtype=torch.float, device=self.device)#lego1x3:
            self.axes1_local = [torch.tensor([1,0,0], dtype=torch.float, device=self.device), torch.tensor([0,1,0], dtype=torch.float, device=self.device), torch.tensor([0,0,1], dtype=torch.float, device=self.device)]
            self.half_lengths1 = [0.045, 0.015, 0.01]#[0.01040546, 0.01289264, 0.00916166]#lego1x3:
            # intervene控制相关
            listener = keyboard.Listener(on_press=self.on_press)
            listener.start() 
            if self.evaluate != 1:
                wandb.init(project="hil-serl-realworldRL-actor", name=datetime.datetime.strftime(datetime.datetime.now(), '%m%d_%H%M%S'),
                    save_code=True)

    def on_press(self, key):
        if key == keyboard.Key.esc:
            self.success_flag = True

    def quat_apply(self, a, b):
        shape = b.shape
        a = a.reshape(-1, 4)
        b = b.reshape(-1, 3)
        xyz = a[:, :3]
        t = xyz.cross(b, dim=-1) * 2
        return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

    def quat_conjugate(self, a):
        shape = a.shape
        a = a.reshape(-1, 4)
        return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

    def quat_mul(self, a, b):
        assert a.shape == b.shape
        shape = a.shape
        a = a.reshape(-1, 4)
        b = b.reshape(-1, 4)

        x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)

        quat = torch.stack([x, y, z, w], dim=-1).view(shape)

        return quat
    
    def scale(self, x, lower, upper):
        return (0.5 * (x + 1.0) * (upper - lower) + lower)

    def unscale(self, x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)
    

    def uniform_mean_var(self, a: float, b: float):
        mean = (a + b) / 2.0
        var = ((b - a) ** 2) / 12.0
        return mean, var

    def normalize_obs(self, obs):
        # 其中积木相对pos的均值和方差是大概估计的，相对rotxy使用demo数据集算的
        mean_array = np.array([self.uniform_mean_var(-0.2926, -0.1926)[0], self.uniform_mean_var(-0.2858, -0.2158)[0], self.uniform_mean_var(1.0738, 1.1438)[0], -0.22652714732867568, 0.2561069464505608, 1.0362594955003084, 0.35423625049306384, 0.765887027356162, -0.2066563985018588, 0.4941543829974844, 0.6193085962267064, 0.20342207144000637, -0.6824457222846017, 0.33021267492379713, self.uniform_mean_var(-0.05, 0.02)[0], self.uniform_mean_var(-0.05, 0.05)[0], self.uniform_mean_var(-0.05, 0.02)[0], 0.9966682711643959, -0.028899417283124666, -0.011227106159672822, 0.027460898248367567, 0.9932702824250975, -0.08206924790066472])
        std_array = np.array([self.uniform_mean_var(-0.2926, -0.1926)[1], self.uniform_mean_var(-0.2858, -0.2158)[1], self.uniform_mean_var(1.0738, 1.1438)[1], 5.941238936679739e-15, 9.545197483090476e-15, 1.701646175270429e-14, 0.0003587173791237157, 9.183422020042598e-05, 0.00017044439733786802, 0.00011359052424739716, 1.3704262595397993e-13, 4.359876361820508e-13, 4.0608292946685986e-14, 1.0110473879856631e-14, self.uniform_mean_var(-0.05, 0.02)[1], self.uniform_mean_var(-0.05, 0.05)[1], self.uniform_mean_var(-0.05, 0.02)[1], 1.1219276644658074e-05, 0.004665335037759593, 0.0010145852726955434, 0.0049483726544047365, 1.6996579429569566e-05, 0.0009593111255655045])
        obs = (obs - mean_array) / np.sqrt(std_array + 1e-8)
        return obs
    
    # def intervene_ik(self):
    #     flag = input("0 rotate/1 insert half/2 insert full/3 run up and down?")
    #     while flag not in ["0", "1", "2", "3"]:
    #         flag = input("0 rotate/1 insert half/2 insert full/3 run up and down?")
    #     if flag == "3":
    #         self.run_updown_leftright()
    #     else:
    #         if flag == "0":
    #             ik_target_relative_pos = self.ik_target_relative_pos0   
    #         elif flag == "1":
    #             ik_target_relative_pos = self.ik_target_relative_pos1
    #         elif flag == "2":
    #             ik_target_relative_pos = self.ik_target_relative_pos2
    #         lego1_pose = np.frombuffer(self.redis_server.get("latest_pose1"), dtype=np.float32)#np.array([-0.06876533, -0.00709992, 1.29667835, -0.56338671, 0.49874087, 0.62272573, 0.21462903])
    #         lego_rot1 = torch.from_numpy(lego1_pose[3:]).float().to(self.device)
    #         lego_pos1 = torch.from_numpy(lego1_pose[:3]).float().to(self.device)
    #         lego_target0 = self.quat_apply(lego_rot1, ik_target_relative_pos) + lego_pos1
    #         lego_quat_target0 = self.quat_mul(lego_rot1, self.ik_target_relative_quat)
            
    #         lego0_pose = np.frombuffer(self.redis_server.get("latest_pose0"), dtype=np.float32)
    #         lego_pos0 = torch.from_numpy(lego0_pose[:3]).to(self.device)
    #         lego_rot0 = torch.from_numpy(lego0_pose[3:]).to(self.device)
    #         # 用真机的FK
    #         baseur2ee_pose = self.rtde_c.getForwardKinematics()
    #         baseur2ee_pos = torch.tensor(baseur2ee_pose[:3]).to(self.device)
    #         baseur2ee_quat = torch.tensor(R.from_rotvec(baseur2ee_pose[3:]).as_quat()).to(self.device)
            
    #         # 如果foundationpose测出来的是baseur2obj
    #         baseur2lego0_pos = lego_pos0.clone()
    #         baseur2lego0_rot = lego_rot0.clone()
    #         lego02ee_pos = self.quat_apply(self.quat_conjugate(baseur2lego0_rot), baseur2ee_pos - baseur2lego0_pos)
    #         lego02ee_quat = self.quat_mul(self.quat_conjugate(baseur2lego0_rot), baseur2ee_quat)
    #         baseur2ee_target_pos = self.quat_apply(lego_quat_target0, lego02ee_pos) + lego_target0
    #         baseur2ee_target_rot = self.quat_mul(lego_quat_target0, lego02ee_quat)

    #         baseur2ee_target_rotvec = R.from_quat(baseur2ee_target_rot.cpu().numpy()).as_rotvec()
    #         target_joint_pos = self.cal_ik_real_robot(baseur2ee_target_pos.cpu().numpy(), baseur2ee_target_rotvec, self.rtde_r.getActualQ())#self.rtde_c.getInverseKinematics(np.concatenate([baseur2ee_target_pos.cpu().numpy(), baseur2ee_target_rotvec]))#another_rtde_c.getInverseKinematics(np.concatenate([baseur2ee_target_pos.cpu().numpy(), baseur2ee_target_rotvec]))
    #         self.rtde_c.moveJ(target_joint_pos, 0.1, 1)
    
    def constrain_steplen_intervene(self):
        action_flag = input("rotate0/insert1/manual intervene2:")
        while action_flag not in ["0", "1", "2"]:
            action_flag = input("rotate0/insert1/manual intervene2:")
        obs, world2eemy_pos0, world2eemy_pos1, world2eemy_rot0, world2eemy_rot1, lego_rot0, lego_rot1, lego_pos0, lego_pos1 = self.compute_obs()
        
        if action_flag == "2":
            self.intervene_panel.reset()
            # print("target_difference_rot:", target_difference_rot.cpu().numpy())
            input("if you finish intervention, press enter to continue...")
            action_values = self.intervene_panel.get_values().astype(np.float32)
            values = np.concatenate([action_values[:3]*self.action_pos_range_value, action_values[3:]*self.action_rot_range_value])
        else:
            if action_flag == "0":
                replay_relative_pos = self.ik_target_relative_pos0
            else:
                replay_relative_pos = self.ik_target_relative_pos2
            
            current_world2eemy_pos0 = world2eemy_pos0.clone()
            current_world2eemy_pos1 = world2eemy_pos1.clone()
            current_world2eemy_rot0 = world2eemy_rot0.clone()
            current_world2eemy_rot1 = world2eemy_rot1.clone()
            lego1_pose = np.frombuffer(self.redis_server.get("latest_pose1"), dtype=np.float32)#np.array([-0.06876533, -0.00709992, 1.29667835, -0.56338671, 0.49874087, 0.62272573, 0.21462903])
            lego_rot1 = torch.from_numpy(lego1_pose[3:]).float().to(self.device)
            lego_pos1 = torch.from_numpy(lego1_pose[:3]).float().to(self.device)
            lego0_pose = np.frombuffer(self.redis_server.get("latest_pose0"), dtype=np.float32)
            lego_pos0 = torch.from_numpy(lego0_pose[:3]).float().to(self.device)
            lego_rot0 = torch.from_numpy(lego0_pose[3:]).float().to(self.device)
            
            lego_target0 = self.quat_apply(lego_rot1, replay_relative_pos) + lego_pos1
            lego_quat_target0 = self.quat_mul(lego_rot1, self.target_relative_quat)
            
            # 用真机的FK
            baseur2ee_pose = self.rtde_c.getForwardKinematics()
            baseur2ee_pos = torch.tensor(baseur2ee_pose[:3]).to(self.device)
            baseur2ee_quat = torch.tensor(R.from_rotvec(baseur2ee_pose[3:]).as_quat()).float().to(self.device)
            # 如果foundationpose测出来的是baseur2obj
            baseur2lego0_pos = lego_pos0.clone()
            baseur2lego0_rot = lego_rot0.clone()
            lego02ee_pos = self.quat_apply(self.quat_conjugate(baseur2lego0_rot), baseur2ee_pos - baseur2lego0_pos)
            lego02ee_quat = self.quat_mul(self.quat_conjugate(baseur2lego0_rot), baseur2ee_quat)
            baseur2ee_target_pos = self.quat_apply(lego_quat_target0, lego02ee_pos) + lego_target0
            baseur2ee_target_rot = self.quat_mul(lego_quat_target0, lego02ee_quat)
            world2ee_target_pos = self.quat_apply(self.r_world2baseur_rot, baseur2ee_target_pos) + self.r_world2baseur_pos
            world2ee_target_quat = self.quat_mul(self.r_world2baseur_rot, baseur2ee_target_rot)
            world2eemy_target_pos = self.quat_apply(world2ee_target_quat, self.ee2eemy_pos) + world2ee_target_pos
            world2eemy_target_rot = self.quat_mul(world2ee_target_quat, self.ee2eemy_rot)
            # eecurrent2eetarget
            target_difference_rot = torch.from_numpy(R.from_quat((self.quat_mul(world2eemy_target_rot, self.quat_conjugate(current_world2eemy_rot0))).cpu().numpy()).as_euler("xyz")).float()
            target_difference_pos = world2eemy_target_pos - current_world2eemy_pos0#self.quat_apply(self.quat_conjugate(current_world2eemy_rot0), world2eemy_target_pos - current_world2eemy_pos0)
            real_deploy_differece_rot = torch.clamp(target_difference_rot/self.action_rot_range_value, -1*torch.ones_like(target_difference_rot), 1*torch.ones_like(target_difference_rot))
            real_deploy_differece_pos = torch.clamp(target_difference_pos/self.action_pos_range_value, -1*torch.ones_like(target_difference_pos), 1*torch.ones_like(target_difference_pos))
            current_rgb_img = np.frombuffer(self.redis_server.get("color_image"), dtype=np.float32)
            current_tcp_force_torque = np.concatenate([self.rtde_r.getActualTCPForce(), self.another_rtde_r.getActualTCPForce()])
            qpos0_r = np.array(self.hand.get_hand_angle())[::-1]
            qpos0_l = np.array(self.another_hand.get_hand_angle())[::-1]
            pos0_r, pos0_l, rot0_r, rot0_l = world2eemy_pos0.clone(), world2eemy_pos1.clone(), world2eemy_rot0.clone(), world2eemy_rot1.clone()
            # self.intervene_panel.reset()
            # print("target_difference_rot:", target_difference_rot.cpu().numpy())
            # input("if you finish intervention, press enter to continue...")
            # action_values = self.intervene_panel.get_values().astype(np.float32)
            # values = action_values.copy()
            # values[:3] = action_values[:3] * self.action_pos_range_value
            # values[3:] = action_values[3:] * self.action_rot_range_value
            action_values = np.concatenate([real_deploy_differece_pos.cpu().numpy(), real_deploy_differece_rot.cpu().numpy()])
            values = np.concatenate([real_deploy_differece_pos.cpu().numpy()*self.action_pos_range_value, real_deploy_differece_rot.cpu().numpy()*self.action_rot_range_value])
        
        initial_world2eemy_pos0 = world2eemy_pos0.clone()
        initial_world2eemy_pos1 = world2eemy_pos1.clone()
        # 右臂6dof
        target_world2eemy_pos0 = initial_world2eemy_pos0 + torch.from_numpy(values[:3]).to(self.device)
        world2eemy_euler_err0 = values[3:]
        world2eemy_quat_err0 = torch.from_numpy(R.from_euler('xyz', world2eemy_euler_err0).as_quat()).float().to(self.device)
        world2eemy_quat_dt0 = self.quat_mul(self.quat_mul(self.quat_conjugate(world2eemy_rot0), world2eemy_quat_err0), world2eemy_rot0)    
        target_world2eemy_quat0 = self.quat_mul(world2eemy_rot0, world2eemy_quat_dt0)
        clamp_target_world2eemy_quat0 = target_world2eemy_quat0
        target_world2ee_pos0 = self.quat_apply(clamp_target_world2eemy_quat0, self.eemy2ee_pos) + target_world2eemy_pos0
        clamp_target_world2ee_pos0 = target_world2ee_pos0#torch.clamp(target_world2ee_pos0, self.target_eeur_action_pos_range_lower[:3], self.target_eeur_action_pos_range_upper[:3])
        clamp_target_baseur2ee_pos0 = self.quat_apply(self.quat_conjugate(self.r_world2baseur_rot), clamp_target_world2ee_pos0 - self.r_world2baseur_pos)
        clamp_target_world2ee_rot0 = self.quat_mul(clamp_target_world2eemy_quat0, self.eemy2ee_rot)#self.quat_mul(world2eemy_rot0, self.eemy2ee_rot)
        clamp_target_baseur2ee_rot0 = R.from_quat(self.quat_mul(self.quat_conjugate(self.r_world2baseur_rot), clamp_target_world2ee_rot0).cpu().numpy()).as_rotvec()
        arm_pos = self.cal_ik_real_robot(clamp_target_baseur2ee_pos0.cpu().numpy(), clamp_target_baseur2ee_rot0, self.rtde_r.getActualQ())#self.rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos0.cpu().numpy(), baseur2ee_rot0]))
        print("initial_dof_pos0:", self.rtde_r.getActualQ())
        print("target dof pos0:", arm_pos)
        input("press enter to continue...")
        self.rtde_c.moveJ(arm_pos, 0.1, 1)
        time.sleep(0.02)
        input("remember to esc again...")
        return action_values

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!mybutton:", buttons)
        print("!!!!!!!!!!expert_a:", expert_a)
        # self.left, self.right = tuple(buttons)
        self.left, self.right = buttons[0], buttons[-1]
        intervened = False
        
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        # if intervened:
        return expert_a, True

        # return action, False
        
    def step(self, action):
        print("--------------------------step:", self.steps)
        print("action:", action)
        obs, world2eemy_pos0, world2eemy_pos1, world2eemy_rot0, world2eemy_rot1, lego_rot0, lego_rot1, lego_pos0, lego_pos1 = self.compute_obs()
        
        info = {}
        bad_data = False
        action, is_intervened = self.action(action)
        if is_intervened:
            info["intervened"] = action
        action = torch.from_numpy(action).float().to(self.device)
        initial_world2eemy_pos0 = world2eemy_pos0.clone()
        initial_world2eemy_pos1 = world2eemy_pos1.clone()
        # target_world2eemy_pos0 = initial_world2eemy_pos0 + self.scale(action[:3], self.action_pos_range_lower[:3], self.action_pos_range_upper[:3])
        # target_world2eemy_pos1 = initial_world2eemy_pos1 + self.scale(action[6:9], self.action_pos_range_lower[3:], self.action_pos_range_upper[3:])
        
        # world2eemy_euler_err0 = self.scale(action[3:6], self.action_rot_range_lower, self.action_rot_range_upper).cpu().numpy()
        # world2eemy_quat_err0 = torch.from_numpy(R.from_euler('xyz', world2eemy_euler_err0).as_quat()).float().to(self.device)
        # world2eemy_euler_err1 = self.scale(action[9:12], self.action_rot_range_lower, self.action_rot_range_upper).cpu().numpy()
        # world2eemy_quat_err1 = torch.from_numpy(R.from_euler('xyz', world2eemy_euler_err1).as_quat()).float().to(self.device)
        # world2eemy_quat_dt0 = self.quat_mul(self.quat_mul(self.quat_conjugate(world2eemy_rot0), world2eemy_quat_err0), world2eemy_rot0)
        # world2eemy_quat_dt1 = self.quat_mul(self.quat_mul(self.quat_conjugate(world2eemy_rot1), world2eemy_quat_err1), world2eemy_rot1)

        # target_world2eemy_quat0 = self.quat_mul(world2eemy_rot0, world2eemy_quat_dt0)
        # target_world2eemy_quat1 = self.quat_mul(world2eemy_rot1, world2eemy_quat_dt1)
        # target_world2ee_pos0 = self.quat_apply(target_world2eemy_quat0, self.eemy2ee_pos) + target_world2eemy_pos0
        # target_world2ee_pos1 = self.quat_apply(target_world2eemy_quat1, self.eemy2ee_pos) + target_world2eemy_pos1
        # clamp_target_world2ee_pos0 = torch.clamp(target_world2ee_pos0, self.target_eeur_action_pos_range_lower[:3], self.target_eeur_action_pos_range_upper[:3])
        # clamp_target_world2ee_pos1 = torch.clamp(target_world2ee_pos1, self.target_eeur_action_pos_range_lower[3:], self.target_eeur_action_pos_range_upper[3:])

        # clamp_target_baseur2ee_pos0 = self.quat_apply(self.quat_conjugate(self.r_world2baseur_rot), clamp_target_world2ee_pos0 - self.r_world2baseur_pos)
        # clamp_target_baseur2ee_pos1 = self.quat_apply(self.quat_conjugate(self.l_world2baseur_rot), clamp_target_world2ee_pos1 - self.l_world2baseur_pos)
        # world2ee_rot0 = self.quat_mul(world2eemy_rot0, self.eemy2ee_rot)
        # world2ee_rot1 = self.quat_mul(world2eemy_rot1, self.eemy2ee_rot)
        # baseur2ee_rot0 = R.from_quat(self.quat_mul(self.quat_conjugate(self.r_world2baseur_rot), world2ee_rot0).cpu().numpy()).as_rotvec()
        # baseur2ee_rot1 = R.from_quat(self.quat_mul(self.quat_conjugate(self.l_world2baseur_rot), world2ee_rot1).cpu().numpy()).as_rotvec()
        
        # 右臂6dof
        target_world2eemy_pos0 = initial_world2eemy_pos0 + action[:3]*self.action_pos_range_value#self.scale(action[:3], self.action_pos_range_lower[:3], self.action_pos_range_upper[:3])
        # print("initial_world2eemy_pos0:", initial_world2eemy_pos0)
        # print("target_world2eemy_pos0:", target_world2eemy_pos0)
        # print("self.scale(action[:3], self.action_pos_range_lower[:3], self.action_pos_range_upper[:3]):", self.scale(action[:3], self.action_pos_range_lower[:3], self.action_pos_range_upper[:3]))
        world2eemy_euler_err0 = (action[3:]*self.action_rot_range_value).cpu().numpy()#self.scale(action[3:6], self.action_rot_range_lower, self.action_rot_range_upper).cpu().numpy()
        world2eemy_quat_err0 = torch.from_numpy(R.from_euler('xyz', world2eemy_euler_err0).as_quat()).float().to(self.device)
        world2eemy_quat_dt0 = self.quat_mul(self.quat_mul(self.quat_conjugate(world2eemy_rot0), world2eemy_quat_err0), world2eemy_rot0)
        
        target_world2eemy_quat0 = self.quat_mul(world2eemy_rot0, world2eemy_quat_dt0)
        # world2eemy_euler_dt = torch.from_numpy(R.from_quat(self.quat_mul(self.quat_conjugate(self.init_world2eemy_quat0), target_world2eemy_quat0).cpu().numpy()).as_euler("xyz")).to(self.device)
        # clamp_world2eemy_euler_dt = torch.clamp(world2eemy_euler_dt, -2*torch.abs(self.eemy02eemy1_limit_euler), 2*torch.abs(self.eemy02eemy1_limit_euler))#self.eemy02eemy1_limit_euler_low, self.eemy02eemy1_limit_euler_high)
        # clamp_world2eemy_quat_dt = torch.from_numpy(R.from_euler('xyz', clamp_world2eemy_euler_dt.cpu().numpy()).as_quat()).float().to(self.device)
        # clamp_target_world2eemy_quat0 = self.quat_mul(self.init_world2eemy_quat0, clamp_world2eemy_quat_dt)
        #TODO
        clamp_target_world2eemy_quat0 = target_world2eemy_quat0
        
        
        target_world2ee_pos0 = self.quat_apply(clamp_target_world2eemy_quat0, self.eemy2ee_pos) + target_world2eemy_pos0
        # print("current_world2ee_pos0:", self.quat_apply(world2eemy_rot0, self.eemy2ee_pos) + world2eemy_pos0)
        # print("target_world2ee_pos0:", target_world2ee_pos0)
        clamp_target_world2ee_pos0 = target_world2ee_pos0#torch.clamp(target_world2ee_pos0, self.target_eeur_action_pos_range_lower[:3], self.target_eeur_action_pos_range_upper[:3])
        
        # clamp quat
        # print("clamp_target_world2ee_pos0:", clamp_target_world2ee_pos0)
        # print("action[3:]:", action[3:])
        # print("world2eemy_euler_err0:", world2eemy_euler_err0)
        # print("world2eemy_quat_dt0:", world2eemy_quat_dt0)
        # print("world2eemy_rot0:", world2eemy_rot0)
        # print("target_world2eemy_quat0:", target_world2eemy_quat0)
        clamp_target_baseur2ee_pos0 = self.quat_apply(self.quat_conjugate(self.r_world2baseur_rot), clamp_target_world2ee_pos0 - self.r_world2baseur_pos)
        world2ee_rot0 = self.quat_mul(clamp_target_world2eemy_quat0, self.eemy2ee_rot)#self.quat_mul(world2eemy_rot0, self.eemy2ee_rot)
        baseur2ee_rot0 = R.from_quat(self.quat_mul(self.quat_conjugate(self.r_world2baseur_rot), world2ee_rot0).cpu().numpy()).as_rotvec()
        # print("action_baseur2ee_rot0:", R.from_quat(self.quat_mul(self.quat_conjugate(self.r_world2baseur_rot), world2ee_rot0).cpu().numpy()).as_quat())
        # 左右臂12dof+手12dof
        # exec_action = np.zeros(24)
        # arm_pos = self.rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos0.cpu().numpy(), baseur2ee_rot0]))
        # another_arm_pos = self.another_rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos1.cpu().numpy(), baseur2ee_rot1]))
        # exec_action[6:12] = np.array(self.hand.get_hand_angle())[::-1] + self.hand_action_scale*action[12:18].cpu().numpy()
        # exec_action[18:24] = np.array(self.another_hand.get_hand_angle())[::-1] + self.hand_action_scale*action[18:24].cpu().numpy()
        # exec_action[:6] = arm_pos#np.array([1.2606e+00, -5.4895e-01,  1.3290e+00, -3.0221e+00,  4.3747e+00, -5.0953e-01])#arm_pos
        # exec_action[12:18] = another_arm_pos
        # print("initial_dof_pos0:", np.concatenate([self.rtde_r.getActualQ(), np.array(self.hand.get_hand_angle())[::-1]]))
        # print("initial_dof_pos1:", np.concatenate([self.another_rtde_r.getActualQ(), np.array(self.another_hand.get_hand_angle())[::-1]]))
        # print("action0:", exec_action[:12])
        # print("action1:", exec_action[12:])
        # input("Press Enter to continue...")
        # # self.rtde_c.servoJ(exec_action[:6], 0.2, 0.1, 0.05, 0.2, 600)
        # self.rtde_c.moveJ(exec_action[:6], 0.1, 1)
        # self.hand.set_hand_angle(exec_action[6:12][::-1])
        # time.sleep(0.02)
        # input("Press Enter to continue...")
        # # self.another_rtde_c.servoJ(exec_action[12:18], 0.2, 0.1, 0.05, 0.2, 600)
        # self.another_rtde_c.moveJ(exec_action[12:18], 0.1, 1)
        # self.another_hand.set_hand_angle(exec_action[18:24][::-1])
        # time.sleep(0.02)

        # 左右臂12dof
        # exec_action = np.zeros(12)
        # arm_pos = self.rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos0.cpu().numpy(), baseur2ee_rot0]))
        # another_arm_pos = self.another_rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos1.cpu().numpy(), baseur2ee_rot1]))
        # exec_action[:6] = arm_pos#np.array([1.2606e+00, -5.4895e-01,  1.3290e+00, -3.0221e+00,  4.3747e+00, -5.0953e-01])#arm_pos
        # exec_action[6:12] = another_arm_pos
        # print("initial_dof_pos0:", self.rtde_r.getActualQ())
        # print("initial_dof_pos1:", self.another_rtde_r.getActualQ())
        # print("action0:", exec_action[:6])
        # print("action1:", exec_action[6:])
        # input("Press Enter to continue...")
        # self.rtde_c.moveJ(exec_action[:6], 0.1, 1)
        # time.sleep(0.02)
        # input("Press Enter to continue...")
        # self.another_rtde_c.moveJ(exec_action[6:], 0.1, 1)
        # time.sleep(0.02)

        # 防止碰撞
        ################之前对于解ik的措施/撞盒子的措施，现在不需要嘞
        # collision_flag = 0
        # curr_rbaseur2ee_pose0 = self.cal_fk_real_robot("right")
        # curr_rbaseur2ee_pos0 = torch.tensor(curr_rbaseur2ee_pose0[:3]).float().to(self.device)
        # curr_rbaseur2ee_rot0 = torch.tensor(R.from_rotvec(curr_rbaseur2ee_pose0[3:6]).as_quat()).float().to(self.device)
        # curr_lbaseur2ee_pose1 = self.cal_fk_real_robot("left")
        # curr_lbaseur2ee_pos1 = torch.tensor(curr_lbaseur2ee_pose1[:3]).float().to(self.device)
        # curr_lbaseur2ee_rot1 = torch.tensor(R.from_rotvec(curr_lbaseur2ee_pose1[3:6]).as_quat()).float().to(self.device)
        # curr_rbaseur2ee_pos1 = self.quat_apply(self.rbaseur2lbaseur_rot, curr_lbaseur2ee_pos1) + self.rbaseur2lbaseur_pos
        # curr_rbaseur2ee_rot1 = self.quat_mul(self.rbaseur2lbaseur_rot, curr_lbaseur2ee_rot1)

        # curr_lego_pos0 = lego_pos0.clone()
        # curr_lego_pos1 = lego_pos1.clone()
        # ee02lego0_pos = self.quat_apply(self.quat_conjugate(curr_rbaseur2ee_rot0), curr_lego_pos0 - curr_rbaseur2ee_pos0)
        # ee12lego1_pos = self.quat_apply(self.quat_conjugate(curr_rbaseur2ee_rot1), curr_lego_pos1 - curr_rbaseur2ee_pos1)
        
        # # 右手
        # pred_rbaseur2ee_pos0 = clamp_target_baseur2ee_pos0.clone()
        # pred_rbaseur2ee_rot0 = torch.from_numpy(R.from_rotvec(baseur2ee_rot0).as_quat()).float().to(self.device)
        # # 左手不动
        # pred_rbaseur2ee_pos1 = curr_rbaseur2ee_pos1.clone()
        # pred_rbaseur2ee_rot1 = curr_rbaseur2ee_rot1.clone()

        # pred_rbaseur2lego0_pos = self.quat_apply(pred_rbaseur2ee_rot0, ee02lego0_pos) + pred_rbaseur2ee_pos0
        # pred_rbaseur2lego1_pos = self.quat_apply(pred_rbaseur2ee_rot1, ee12lego1_pos) + pred_rbaseur2ee_pos1
        # world2lego0_pos = self.quat_apply(self.r_world2baseur_rot, pred_rbaseur2lego0_pos) + self.r_world2baseur_pos
        # world2lego1_pos = self.quat_apply(self.r_world2baseur_rot, pred_rbaseur2lego1_pos) + self.r_world2baseur_pos
        # lowest_z_pos = 1.0467+0.01-0.1 # 积木放在台子上的世界坐标系下的z坐标
        # print("!!!!world2lego0_pos:", world2lego0_pos)
        # print("!!!!world2lego1_pos:", world2lego1_pos)
        # if (world2lego0_pos[2] < lowest_z_pos) or (world2lego1_pos[2] < lowest_z_pos):
        #     collision_flag = 1
        # # 右臂6dof
        # if collision_flag == 0:
        exec_action = np.zeros(6)
        arm_pos = self.cal_ik_real_robot(clamp_target_baseur2ee_pos0.cpu().numpy(), baseur2ee_rot0, self.rtde_r.getActualQ())#self.rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos0.cpu().numpy(), baseur2ee_rot0]))
        # another_arm_pos = self.another_rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos1.cpu().numpy(), baseur2ee_rot1]))
        # exec_action[6:12] = np.array(self.hand.get_hand_angle())[::-1] + self.hand_action_scale*action[12:18].cpu().numpy()
        # exec_action[18:24] = np.array(self.another_hand.get_hand_angle())[::-1] + self.hand_action_scale*action[18:24].cpu().numpy()
        if arm_pos is None:
            bad_data = True
        else:
            exec_action[:6] = arm_pos#np.array([1.2606e+00, -5.4895e-01,  1.3290e+00, -3.0221e+00,  4.3747e+00, -5.0953e-01])#arm_pos
            # exec_action[12:18] = another_arm_pos
            print("initial_dof_pos0:", self.rtde_r.getActualQ())
            # print("initial_dof_pos1:", np.concatenate([self.another_rtde_r.getActualQ(), np.array(self.another_hand.get_hand_angle())[::-1]]))
            print("action0:", exec_action[:6])
            # print("action1:", exec_action[12:])
            # if self.total_steps >= 39:
            
            # self.rtde_c.servoJ(exec_action[:6], 0.2, 0.1, 0.05, 0.2, 600)
            # if exec_action[0]*self.rtde_r.getActualQ()[0] > 0:
            self.rtde_c.moveJ(exec_action[:6], 0.1, 1, True)
            time.sleep(0.2)
            self.rtde_c.stopJ(1, False)
            #     else:
            #         print("danger!!!!!!!!!!!!!!!!!!!!!!")
            ################
        # if self.evaluate == 0:
        #     if self.total_steps >= 39:
        #         flag = input("0:normal;1:success;2:dropped")
        #         while (flag != "0" and flag != "1" and flag != "2"):
        #             flag = input("0:normal;1:success;2:dropped")
        # else:
        # flag = input("0:normal;1:success;2:dropped")
        # while (flag != "0" and flag != "1" and flag != "2"):
        #     flag = input("0:normal;1:success;2:dropped")
        
        new_obs, new_world2eemy_pos0, new_world2eemy_pos1, new_world2eemy_rot0, new_world2eemy_rot1, lego_rot0, lego_rot1, lego_pos0, lego_pos1 = self.compute_obs()
        # print("new_world2eemy_rot0:", new_world2eemy_rot0)
        self.steps += 1
        self.total_steps += 1
        # if self.evaluate == 0:
        # if self.total_steps >= 40:
        # success_flag = 1 if flag == "1" else 0
        # drop_flag = 1 if flag == "2" else 0
        # else:
        #     success_flag = 0
        #     drop_flag = 0
        # else:
        # success_flag = 0#1 if flag == "1" else 0
        # drop_flag = 0#1 if flag == "2" else 0
            
        # if self.steps == 1:
        # align_flag = input("1:align;0:not align")
        # while (align_flag != "0" and align_flag != "1"):
        #     align_flag = input("1:align;0:not align")
        reward, done = self.compute_reward(self.steps, 0, self.success_flag, lego_rot0, lego_rot1, lego_pos0, lego_pos1)
        # else:
        #     reward, done = self.compute_reward(self.steps, 0, drop_flag, success_flag, lego_rot0, lego_rot1, lego_pos0, lego_pos1)

        # self.queue[self.queue_index%self.queue.shape[0]] = difference
        # self.queue_index += 1
        self.episode_reward += reward
        info["bad_data"] = bad_data
        # new_obs = torch.cat((new_obs, torch.tensor([0]).float().to(self.device)), dim=-1)
        return new_obs.cpu().numpy(), reward.cpu().numpy(), done, done, info

    def cal_fk_real_robot(self, hand_str, max_retries=100, retry_delay=1):
        for attempt in range(max_retries):
            try:
                if hand_str == "right":
                    baseur2ee_pose = self.rtde_c.getForwardKinematics()
                else:
                    baseur2ee_pose = self.another_rtde_c.getForwardKinematics()
                break  # 成功则跳出循环
                
            except RuntimeError as e:
                print("error:", e)
                input("把手臂移动一下，脱离安全性停止。。。")
                # hand_num = input("left:1;right:0——")
                # while (hand_num != "1" and hand_num != "0"):
                #     hand_num = input("left:1;right:0——")
                # if hand_num == "0":
                #     self.rtde_c.disconnect()
                #     self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
                # else:
                #     self.another_rtde_c.disconnect()
                #     self.another_rtde_c = rtde_control.RTDEControlInterface(self.another_robot_ip)
                self.reconnect_to_robot()
                if attempt < max_retries - 1:
                    # 不是最后一次尝试，等待后重试
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                    continue
                else:
                    break
        return baseur2ee_pose
    
    
    def cal_ik_real_robot(self, target_pos, target_rotvec, current_qpos):
        rot_matrix = R.from_rotvec(target_rotvec).as_matrix()
        trans_matrix = np.eye(4)
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[:3, 3] = target_pos
        solutions = ur5e.inverse_kinematics(trans_matrix)
        fs = []
        for solution in solutions:
            choose_solution = []
            for s in range(len(solution)):
                if (np.sign(solution[s]) == np.sign(current_qpos[s])) or (abs(solution[s] - current_qpos[s]) < 2):
                    choose_solution.append(solution[s])
            if len(choose_solution) == 6:
                fs.append(choose_solution)
        if len(fs) == 0:
            print("current_qpos:", current_qpos)
            print("choose one solution:")
            for solution in solutions:
                print(solution)
            chosen_solution = np.fromstring("", sep=' ')
            while (len(np.fromstring(chosen_solution, sep=' ')) != 6):
                chosen_solution = input("input solution:")
                if chosen_solution == "-1":
                    break
            # while (len(chosen_solution.split(' ')) != 6):
            #     chosen_solution = input("input index:")
            #-1.1082551  -2.50834712 -0.48989332  -3.91007166 -1.61642848  0.71793799
            if chosen_solution == "-1":
                return None
            else:
                return np.fromstring(chosen_solution, sep=' ')
        return fs[0]
        
    def compute_obs(self):
        lego_pose0 = np.frombuffer(self.redis_server.get("latest_pose0"), dtype=np.float32)
        lego_pos0 = torch.from_numpy(lego_pose0[:3]).float().to(self.device)
        lego_rot0 = torch.from_numpy(lego_pose0[3:7]).float().to(self.device)
        lego_pose1 = np.frombuffer(self.redis_server.get("latest_pose1"), dtype=np.float32)
        lego_pos1 = torch.from_numpy(lego_pose1[:3]).float().to(self.device)
        lego_rot1 = torch.from_numpy(lego_pose1[3:7]).float().to(self.device)
        lego02lego1_pos = self.quat_apply(self.quat_conjugate(lego_rot0), lego_pos1 - lego_pos0)
        lego02lego1_rot = self.quat_mul(self.quat_conjugate(lego_rot0), lego_rot1)
        lego02lego1_matrix = torch.from_numpy(R.from_quat(lego02lego1_rot.cpu().numpy()).as_matrix()).float().to(self.device)
        lego02lego1_rotxy = torch.cat((lego02lego1_matrix[:3, 0], lego02lego1_matrix[:3, 1]), dim=-1)
        # print("lego02lego1_pos:", lego02lego1_pos)
        # print("lego02lego1_rot:", lego02lego1_rot)
        # baseur2ee_pose0 = self.rtde_c.getForwardKinematics()
        baseur2ee_pose0 = self.cal_fk_real_robot("right")
        baseur2ee_pos0 = torch.tensor(baseur2ee_pose0[:3]).float().to(self.device)
        baseur2ee_rot0 = torch.tensor(R.from_rotvec(baseur2ee_pose0[3:6]).as_quat()).float().to(self.device)
        baseur2eemy_pos0 = self.quat_apply(baseur2ee_rot0, self.ee2eemy_pos) + baseur2ee_pos0
        baseur2eemy_rot0 = self.quat_mul(baseur2ee_rot0, self.ee2eemy_rot)
        world2eemy_rot0 = self.quat_mul(self.r_world2baseur_rot, baseur2eemy_rot0)
        world2eemy_pos0 = self.quat_apply(self.r_world2baseur_rot, baseur2eemy_pos0) + self.r_world2baseur_pos

        # baseur2ee_pose1 = self.another_rtde_c.getForwardKinematics()
        baseur2ee_pose1 = self.cal_fk_real_robot("left")
        baseur2ee_pos1 = torch.tensor(baseur2ee_pose1[:3]).float().to(self.device)
        baseur2ee_rot1 = torch.tensor(R.from_rotvec(baseur2ee_pose1[3:6]).as_quat()).float().to(self.device)
        baseur2eemy_pos1 = self.quat_apply(baseur2ee_rot1, self.ee2eemy_pos) + baseur2ee_pos1
        baseur2eemy_rot1 = self.quat_mul(baseur2ee_rot1, self.ee2eemy_rot)
        world2eemy_rot1 = self.quat_mul(self.l_world2baseur_rot, baseur2eemy_rot1)
        world2eemy_pos1 = self.quat_apply(self.l_world2baseur_rot, baseur2eemy_pos1) + self.l_world2baseur_pos
        world2ee_pos0 = self.quat_apply(self.r_world2baseur_rot, baseur2ee_pos0) + self.r_world2baseur_pos
        world2ee_pos1 = self.quat_apply(self.l_world2baseur_rot, baseur2ee_pos1) + self.l_world2baseur_pos
        # print("world2ee_pos0:", world2ee_pos0)
        # print("world2ee_pos1:", world2ee_pos1)
        # print("baseur2ee_rot0:", baseur2ee_rot0)
        # obs = torch.cat((world2eemy_pos0, world2eemy_pos1, world2eemy_rot0, world2eemy_rot1, lego02lego1_pos, lego02lego1_rotxy), dim=-1)
        obs = torch.cat((world2eemy_pos0, world2eemy_pos1, world2eemy_rot0, world2eemy_rot1, lego02lego1_pos, lego02lego1_rotxy, torch.tensor([0]).to(self.device), torch.from_numpy(np.concatenate([self.rtde_r.getActualTCPForce(), self.another_rtde_r.getActualTCPForce()])).to(self.device)), dim=-1)
        
        # obs = torch.cat((lego_pos0, lego_rotxy0, lego_pos1, lego_rotxy1), dim=-1)
        # obs = torch.from_numpy(self.normalize_obs(obs)).to(self.device)
        return obs, world2eemy_pos0, world2eemy_pos1, world2eemy_rot0, world2eemy_rot1, lego_rot0, lego_rot1, lego_pos0, lego_pos1
    
    def is_line_intersecting_rectangle(self, P0, d, V0, V1, V2, V3):
        """
        判断直线是否与任意方向的矩形相交
        参数:
        P0: torch.Tensor, (3,) 直线上的一点
        d: torch.Tensor, (3,) 直线的方向向量
        V0, V1, V2, V3: torch.Tensor, (3,) 矩形的四个顶点

        返回:
        bool: 直线是否与矩形相交
        """
        # Step 1: 计算矩形平面的法向量
        u = V1 - V0
        v = V3 - V0
        n = torch.cross(u, v)

        # Step 2: 检查直线是否平行于平面
        nd_dot = torch.dot(n, d)
        if torch.isclose(nd_dot, torch.tensor(0.0)):
            return False  # 平行不相交

        # Step 3: 计算直线与平面的交点
        t = torch.dot(n, V0 - P0) / nd_dot
        P_intersect = P0 + t * d

        # Step 4: 检查交点是否在矩形内
        P_local = P_intersect - V0
        alpha = torch.dot(P_local, u) / torch.dot(u, u)
        beta = torch.dot(P_local, v) / torch.dot(v, v)

        # 判断是否在矩形范围内
        return (0 <= alpha <= 1) and (0 <= beta <= 1)

    def calculate_angle_difference(self, lego_target0, lego_target1, world2lego_pos0, world2lego_pos1, world2lego_rot0, world2lego_rot1, axes_local1, center_local1, half_lengths1, P1, axes_local0, center_local0, half_lengths0, P0, target_relative_pos, negative_target_relative_pos, device):
        """
        计算中心点目标点连线向量和拼接面法向量的角度差距（帮助进一步对齐）
        """
        for i in range(len(axes_local1)):
            V0 = center_local1 + half_lengths1[(i + 1)%3]*axes_local1[(i + 1)%3] + half_lengths1[(i + 2)%3]*axes_local1[(i + 2)%3]
            V1 = center_local1 + half_lengths1[(i + 1)%3]*axes_local1[(i + 1)%3] - half_lengths1[(i + 2)%3]*axes_local1[(i + 2)%3]
            V2 = center_local1 - half_lengths1[(i + 1)%3]*axes_local1[(i + 1)%3] - half_lengths1[(i + 2)%3]*axes_local1[(i + 2)%3]
            V3 = center_local1 - half_lengths1[(i + 1)%3]*axes_local1[(i + 1)%3] + half_lengths1[(i + 2)%3]*axes_local1[(i + 2)%3]
            if self.is_line_intersecting_rectangle(P1, target_relative_pos, V0, V1, V2, V3):
                break

        u1 = V1 - V0
        v1 = V3 - V0
        A_local1 = torch.cross(u1, v1)
        if torch.sum(A_local1*target_relative_pos, dim=-1) < 0:
            A_local1 = -A_local1
        A1 = self.quat_apply(world2lego_rot1, A_local1.unsqueeze(0))# left lego prime tensor
        A1_normal = A1/torch.norm(A1, p=2, dim=-1).unsqueeze(-1)
        current_tensor1 = world2lego_pos0 - lego_target0
        current_tensor1_length = torch.norm(current_tensor1, p=2, dim=-1)
        sign1 = torch.zeros_like(current_tensor1_length)
        sign1 = torch.where(current_tensor1_length < 1e-4, torch.ones_like(sign1), sign1)
        current_tensor1_length = torch.where(current_tensor1_length < 1e-4, torch.ones_like(current_tensor1_length), current_tensor1_length)
        angle1 = torch.zeros(1, dtype=torch.float, device=device)
        angle0 = torch.zeros(1, dtype=torch.float, device=device)
        angle1 = torch.where(sign1 < 0.5, torch.acos(torch.clamp(torch.sum(A1_normal*(current_tensor1/current_tensor1_length.unsqueeze(-1)), dim=-1), -1, 1)), angle1)
        # if torch.norm(current_tensor1, p=2, dim=-1).unsqueeze(-1) > 1e-4:
        #     current_tensor1_normal = current_tensor1/torch.norm(current_tensor1, p=2, dim=-1).unsqueeze(-1)
        #     angle1 = torch.acos(torch.sum(A1_normal*current_tensor1_normal, dim=-1))
        # else:
        #     angle1 = 0.0

        for i in range(len(axes_local0)):
            V0 = center_local0 + half_lengths0[(i + 1)%3]*axes_local0[(i + 1)%3] + half_lengths0[(i + 2)%3]*axes_local0[(i + 2)%3]
            V1 = center_local0 + half_lengths0[(i + 1)%3]*axes_local0[(i + 1)%3] - half_lengths0[(i + 2)%3]*axes_local0[(i + 2)%3]
            V2 = center_local0 - half_lengths0[(i + 1)%3]*axes_local0[(i + 1)%3] - half_lengths0[(i + 2)%3]*axes_local0[(i + 2)%3]
            V3 = center_local0 - half_lengths0[(i + 1)%3]*axes_local0[(i + 1)%3] + half_lengths0[(i + 2)%3]*axes_local0[(i + 2)%3]
            if self.is_line_intersecting_rectangle(P0, negative_target_relative_pos, V0, V1, V2, V3):
                break

        u0 = V1 - V0
        v0 = V3 - V0
        A_local0 = torch.cross(u0, v0)
        if torch.sum(A_local0*negative_target_relative_pos, dim=-1) < 0:
            A_local0 = -A_local0
        A0 = self.quat_apply(world2lego_rot0, A_local0.unsqueeze(0))# left lego prime tensor
        A0_normal = A0/torch.norm(A0, p=2, dim=-1).unsqueeze(-1)
        current_tensor0 = world2lego_pos1 - lego_target1
        current_tensor0_length = torch.norm(current_tensor0, p=2, dim=-1)
        sign0 = torch.zeros_like(current_tensor0_length)
        sign0 = torch.where(current_tensor0_length < 1e-4, torch.ones_like(sign0), sign0)
        current_tensor0_length = torch.where(current_tensor0_length < 1e-4, torch.ones_like(current_tensor0_length), current_tensor0_length)
        angle0 = torch.where(sign0 < 0.5, torch.acos(torch.clamp(torch.sum(A0_normal*(current_tensor0/current_tensor0_length.unsqueeze(-1)), dim=-1), -1, 1)), angle0)
        # if torch.norm(current_tensor0, p=2, dim=-1).unsqueeze(-1) > 1e-4:
        #     current_tensor0_normal = current_tensor0/torch.norm(current_tensor0, p=2, dim=-1).unsqueeze(-1)
        #     angle0 = torch.acos(torch.sum(A0_normal*current_tensor0_normal, dim=-1))
        # else:
        #     angle0 = 0.0
        return angle0, angle1

    def compute_target_dist(self, lego_rot0, lego_rot1, lego_pos0, lego_pos1):
        lego_target0 = self.quat_apply(lego_rot1, self.target_relative_pos) + lego_pos1
        negative_target_relative_quat = self.quat_conjugate(self.target_relative_quat)
        negative_target_relative_pos = self.quat_apply(negative_target_relative_quat, (torch.tensor([0,0,0]).to(self.device) - self.target_relative_pos).float())
        
        negative_target_relative_rotmatrix = R.from_quat(negative_target_relative_quat.cpu().numpy()).as_matrix()
       
        lego_target1 = self.quat_apply(lego_rot0, negative_target_relative_pos) + lego_pos0

        # prepare for relative_quat_reward rot dist0
        world2legol_quat = R.from_quat((lego_rot1.cpu()/np.linalg.norm(lego_rot1.cpu().numpy())).numpy())
        legol2legortarget_quat = R.from_quat((self.target_relative_quat.cpu()/np.linalg.norm(self.target_relative_quat.cpu().numpy())).numpy())
        world2legortarget_quat = torch.from_numpy((legol2legortarget_quat*world2legol_quat).as_quat()).float().to(self.device)

        world2legor_quat = R.from_quat((lego_rot0.cpu()/np.linalg.norm(lego_rot0.cpu().numpy())).numpy())
        legor2legoltarget_quat = R.from_quat((negative_target_relative_quat.cpu()/np.linalg.norm(negative_target_relative_quat.cpu().numpy())).numpy())
        world2legoltarget_quat = torch.from_numpy((legor2legoltarget_quat*world2legor_quat).as_quat()).float().to(self.device)

        quat_diff0 = self.quat_mul(world2legortarget_quat, self.quat_conjugate(lego_rot0))
        rot_dist0 = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff0[0:3], p=2, dim=-1), max=1.0))


        # prepare for align_reward angle_difference1
        angle_difference0, angle_difference1 = self.calculate_angle_difference(lego_target0, lego_target1, lego_pos0, lego_pos1, lego_rot0, lego_rot1, self.axes1_local, self.center1_local, self.half_lengths1, torch.tensor([0, 0, 0], dtype=torch.float, device=self.device), self.axes0_local, self.center0_local, self.half_lengths0, torch.tensor([0, 0, 0], dtype=torch.float, device=self.device), self.target_relative_pos, negative_target_relative_pos, self.device)
        return rot_dist0, angle_difference1.squeeze(0)

    def compute_point_cloud(self, lego_rot0, lego_rot1, lego_pos0, lego_pos1):
        pointCloudDownsampleNum_right = 50
        pointCloudDownsampleNum_left = 50
        lego0_verts = self.generate_uniform_points("/home/admin01/lyw_2/SeqDex_blockassembly/assets/urdf/blender/origin_obj/1x3/1x3.stl", self.device)
        lego1_verts = self.generate_uniform_points("/home/admin01/lyw_2/SeqDex_blockassembly/assets/urdf/blender/origin_obj/1x3/1x3.stl", self.device)
        lego0_verts = self.sample_points_furthest(lego0_verts, pointCloudDownsampleNum_right)
        lego1_verts = self.sample_points_furthest(lego1_verts, pointCloudDownsampleNum_left)
        lego0_points = self.quat_apply(lego_rot0.unsqueeze(0).repeat(lego0_verts.shape[0], 1), lego0_verts) + lego_pos0.unsqueeze(0).repeat(lego0_verts.shape[0], 1)
        lego1_points = self.quat_apply(lego_rot1.unsqueeze(0).repeat(lego1_verts.shape[0], 1), lego1_verts) + lego_pos1.unsqueeze(0).repeat(lego1_verts.shape[0], 1)
        negative_target_relative_quat = self.quat_conjugate(self.target_relative_quat)
        negative_target_relative_pos = self.quat_apply(negative_target_relative_quat, (torch.tensor([0,0,0]).to(self.device) - self.target_relative_pos).float())
        lego_target0 = self.quat_apply(lego_rot1, self.target_relative_pos) + lego_pos1
        lego_target1 = self.quat_apply(lego_rot0, negative_target_relative_pos) + lego_pos0
        world2legor_quat = R.from_quat(lego_rot0.cpu().numpy()/np.linalg.norm(lego_rot0.cpu().numpy()))
        legor2legoltarget_quat = R.from_quat(negative_target_relative_quat.cpu().numpy()/np.linalg.norm(negative_target_relative_quat.cpu().numpy()))
        world2legoltarget_quat = torch.from_numpy((legor2legoltarget_quat*world2legor_quat).as_quat()).float().to(self.device)
        world2legol_quat = R.from_quat(lego_rot1.cpu().numpy()/np.linalg.norm(lego_rot1.cpu().numpy()))
        legol2legortarget_quat = R.from_quat(self.target_relative_quat.cpu().numpy()/np.linalg.norm(self.target_relative_quat.cpu().numpy())) 
        world2legortarget_quat = torch.from_numpy((legol2legortarget_quat*world2legol_quat).as_quat()).float().to(self.device)
        lego0_target_points = self.quat_apply(world2legortarget_quat.unsqueeze(0).repeat(lego0_verts.shape[0], 1), lego0_verts) + lego_target0.unsqueeze(0).repeat(lego0_verts.shape[0], 1)
        lego1_target_points = self.quat_apply(world2legoltarget_quat.unsqueeze(0).repeat(lego1_verts.shape[0], 1), lego1_verts) + lego_target1.unsqueeze(0).repeat(lego1_verts.shape[0], 1)
        right_pcd = lego0_points.reshape(3*50,)
        left_pcd = lego1_points.reshape(3*50,)
        multipoint_lego0_dist = torch.sum(torch.norm(lego0_points - lego0_target_points, dim=-1, p=2))
        multipoint_lego1_dist = torch.sum(torch.norm(lego1_points - lego1_target_points, dim=-1, p=2))
        return multipoint_lego0_dist, multipoint_lego1_dist

    # def compute_reward(self, drop_flag, success_flag, lego_rot0, lego_rot1, lego_pos0, lego_pos1):
    #     multipoint_lego0_dist, multipoint_lego1_dist = self.compute_point_cloud(lego_rot0, lego_rot1, lego_pos0, lego_pos1)
    #     rot_dist1 = self.compute_target_dist(lego_rot0, lego_rot1, lego_pos0, lego_pos1)
    #     # lego_up_height0, lego_up_height1 = lego_pos0[2] - (0.639-0.0003 + 0.155-0.03), lego_pos1[2] - (0.618+0.0010 + 0.155-0.03)
        
    #     fingerpos_constraint_reward = torch.ones_like(rot_dist1)#torch.clamp(1/(total_fingertip_lego_dist_lh/0.03 + 0.001), 0, 1)*torch.clamp(1/(total_fingertip_lego_dist_rh/0.03 + 0.001), 0, 1)#15/(total_fingertip_lego_dist_lh/0.025 + total_fingertip_lego_dist_rh/0.025 + 1)#10*torch.exp(-5*(torch.abs(total_fingertip_lego_dist_rh) + torch.abs(total_fingertip_lego_dist_lh)))#
        
    #     # object_up_reward0 = 4*torch.exp(20*torch.clamp(lego_up_height0 - 0.35, None, 0))
    #     # object_up_reward1 = 4*torch.exp(20*torch.clamp(lego_up_height1 - 0.35, None, 0))
    #     # object_up_reward = object_up_reward0*object_up_reward1

    #     relative_quat_reward0 = 30*torch.exp(-rot_dist1)
    #     # relative_quat_reward0 = torch.where((object_up_reward0 >= 4)&(object_up_reward1 >= 4), relative_quat_reward0, torch.zeros_like(relative_quat_reward0))

    #     relative_quat_reward = (relative_quat_reward0)#relative_quat_reward0 * relative_quat_reward1/10 + relative_quat_reward1

    #     more_close_reward = torch.where((relative_quat_reward0 > 22), 100000*torch.exp(-multipoint_lego0_dist)*torch.exp(-multipoint_lego1_dist), torch.zeros_like(relative_quat_reward))

    #     insert_success_reward = torch.zeros_like(relative_quat_reward)
    #     if success_flag == 1:
    #         insert_success_reward = 100000#(fingerpos_constraint_reward*(relative_quat_reward + more_close_reward))*0.5
        
    #     reward = (fingerpos_constraint_reward*(relative_quat_reward + more_close_reward) + insert_success_reward)
    #     if drop_flag == 1:
    #         reward = reward - 0.25*(relative_quat_reward + more_close_reward)*fingerpos_constraint_reward
        
    #     if (self.steps == self.max_episode_steps)|(drop_flag == 1)|(success_flag == 1):
    #         done = True
    #     else:
    #         done = False

    #     print("reward:", reward)
    #     print("rot_dist1:", rot_dist1)
    #     print("relative_quat_reward:", relative_quat_reward)
    #     print("insert_success_reward:", insert_success_reward)
    #     print("multipoint_lego0_dist:", multipoint_lego0_dist)
    #     print("multipoint_lego1_dist:", multipoint_lego1_dist)
    #     print("more_close_reward:", more_close_reward)
    #     return reward, done

    # SERL 6dof 对齐为初始状态
    # def compute_reward(self, steps, align_flag, drop_flag, success_flag, lego_rot0, lego_rot1, lego_pos0, lego_pos1):
    #     multipoint_lego0_dist, multipoint_lego1_dist = self.compute_point_cloud(lego_rot0, lego_rot1, lego_pos0, lego_pos1)
    #     rot_dist1, angle_difference1 = self.compute_target_dist(lego_rot0, lego_rot1, lego_pos0, lego_pos1)
    #     # lego_up_height0, lego_up_height1 = lego_pos0[2] - (0.639-0.0003 + 0.155-0.03), lego_pos1[2] - (0.618+0.0010 + 0.155-0.03)
        
    #     fingerpos_constraint_reward = torch.ones_like(rot_dist1)#torch.clamp(1/(total_fingertip_lego_dist_lh/0.03 + 0.001), 0, 1)*torch.clamp(1/(total_fingertip_lego_dist_rh/0.03 + 0.001), 0, 1)#15/(total_fingertip_lego_dist_lh/0.025 + total_fingertip_lego_dist_rh/0.025 + 1)#10*torch.exp(-5*(torch.abs(total_fingertip_lego_dist_rh) + torch.abs(total_fingertip_lego_dist_lh)))#
        
    #     # object_up_reward0 = 4*torch.exp(20*torch.clamp(lego_up_height0 - 0.35, None, 0))
    #     # object_up_reward1 = 4*torch.exp(20*torch.clamp(lego_up_height1 - 0.35, None, 0))
    #     # object_up_reward = object_up_reward0*object_up_reward1

    #     relative_quat_reward0 = 30*torch.exp(-rot_dist1)
    #     # relative_quat_reward0 = torch.where((object_up_reward0 >= 4)&(object_up_reward1 >= 4), relative_quat_reward0, torch.zeros_like(relative_quat_reward0))

    #     relative_quat_reward = (relative_quat_reward0)#relative_quat_reward0 * relative_quat_reward1/10 + relative_quat_reward1
        
    #     align_reward = torch.where((relative_quat_reward0 > 0), 0.5*torch.exp(-10*torch.clamp(angle_difference1 - 0.1, 0, None)) + 0.2*align_flag*torch.ones_like(relative_quat_reward), torch.zeros_like(relative_quat_reward))

    #     more_close_reward = torch.where((relative_quat_reward0 > 0), 1*torch.exp(-multipoint_lego0_dist)*torch.exp(-multipoint_lego1_dist), torch.zeros_like(relative_quat_reward))

    #     insert_success_reward = torch.zeros_like(relative_quat_reward)
    #     if success_flag == 1:
    #         insert_success_reward = 10#(self.max_episode_steps - steps + 1)#(fingerpos_constraint_reward*(relative_quat_reward + more_close_reward))*0.5
        
    #     reward = (fingerpos_constraint_reward*(more_close_reward+align_reward) + insert_success_reward)
    #     if drop_flag == 1:
    #         reward = reward - 0.25*(more_close_reward+align_reward)*fingerpos_constraint_reward
        
    #     if (self.steps == self.max_episode_steps)|(drop_flag == 1)|(success_flag == 1):
    #         done = True
    #     else:
    #         done = False

    #     print("reward:", reward)
    #     print("rot_dist1:", rot_dist1)
    #     print("relative_quat_reward:", relative_quat_reward)
    #     print("insert_success_reward:", insert_success_reward)
    #     print("multipoint_lego0_dist:", multipoint_lego0_dist)
    #     print("multipoint_lego1_dist:", multipoint_lego1_dist)
    #     print("more_close_reward:", more_close_reward)
    #     print("align_reward:", align_reward)
    #     return reward, done

    # SERL不对齐为初始状态
    def compute_reward(self, steps, drop_flag, success_flag, lego_rot0, lego_rot1, lego_pos0, lego_pos1):
        # multipoint_lego0_dist, multipoint_lego1_dist = self.compute_point_cloud(lego_rot0, lego_rot1, lego_pos0, lego_pos1)
        rot_dist1, angle_difference1 = self.compute_target_dist(lego_rot0, lego_rot1, lego_pos0, lego_pos1)
        # # lego_up_height0, lego_up_height1 = lego_pos0[2] - (0.639-0.0003 + 0.155-0.03), lego_pos1[2] - (0.618+0.0010 + 0.155-0.03)
        
        # fingerpos_constraint_reward = torch.ones_like(rot_dist1)#torch.clamp(1/(total_fingertip_lego_dist_lh/0.03 + 0.001), 0, 1)*torch.clamp(1/(total_fingertip_lego_dist_rh/0.03 + 0.001), 0, 1)#15/(total_fingertip_lego_dist_lh/0.025 + total_fingertip_lego_dist_rh/0.025 + 1)#10*torch.exp(-5*(torch.abs(total_fingertip_lego_dist_rh) + torch.abs(total_fingertip_lego_dist_lh)))#
        
        # # object_up_reward0 = 4*torch.exp(20*torch.clamp(lego_up_height0 - 0.35, None, 0))
        # # object_up_reward1 = 4*torch.exp(20*torch.clamp(lego_up_height1 - 0.35, None, 0))
        # # object_up_reward = object_up_reward0*object_up_reward1

        # relative_quat_reward0 = 0.3*torch.exp(-rot_dist1)
        # # relative_quat_reward0 = torch.where((object_up_reward0 >= 4)&(object_up_reward1 >= 4), relative_quat_reward0, torch.zeros_like(relative_quat_reward0))

        # relative_quat_reward = (relative_quat_reward0)#relative_quat_reward0 * relative_quat_reward1/10 + relative_quat_reward1
        
        # align_reward = torch.where((relative_quat_reward > 0.22), 0.2*torch.exp(-10*torch.clamp(angle_difference1 - 0.1, 0, None)), torch.zeros_like(relative_quat_reward))# + 0.2*align_flag*torch.ones_like(relative_quat_reward)

        # more_close_reward = torch.where((relative_quat_reward > 0.22), 1*torch.exp(-multipoint_lego0_dist)*torch.exp(-multipoint_lego1_dist), torch.zeros_like(relative_quat_reward))

        # insert_success_reward = torch.zeros_like(relative_quat_reward)
        # if success_flag == 1:
        #     insert_success_reward = 10#(self.max_episode_steps - steps + 1)#(fingerpos_constraint_reward*(relative_quat_reward + more_close_reward))*0.5
        
        # reward = (fingerpos_constraint_reward*(more_close_reward+align_reward+relative_quat_reward) + insert_success_reward)
        # if drop_flag == 1:
        #     reward = reward - 0.25*(more_close_reward+align_reward+relative_quat_reward)*fingerpos_constraint_reward
        if success_flag == 1:
            reward = 0*torch.ones_like(rot_dist1)
        elif drop_flag == 1:
            reward = -50*torch.ones_like(rot_dist1)
        else:
            reward = -1*torch.ones_like(rot_dist1)
        if (self.steps == self.max_episode_steps) and (success_flag == 0):
            reward = -50*torch.ones_like(rot_dist1)
        if (self.steps == self.max_episode_steps)|(drop_flag == 1)|(success_flag == 1):
            done = True
        else:
            done = False

        print("reward:", reward)
        # print("rot_dist1:", rot_dist1)
        # print("relative_quat_reward:", relative_quat_reward)
        # print("insert_success_reward:", insert_success_reward)
        # print("multipoint_lego0_dist:", multipoint_lego0_dist)
        # print("multipoint_lego1_dist:", multipoint_lego1_dist)
        # print("more_close_reward:", more_close_reward)
        # print("align_reward:", align_reward)
        # print("angle_difference1:", angle_difference1)
        return reward, done
    
    def reset_within_certain_range(self):
        while True:
            random_angle = np.random.uniform(low=-0.523, high=0.523, size=3)
            random_angle[0] = 0
            random_angle[1] = 0
            # random3cm
            # random_angle[2] = 0
            lego12lego0_quat = torch.from_numpy(R.from_euler("xyz", random_angle).as_quat()).float().to(self.device)
            lego12lego0_pos = ((torch.rand(3) - 0.5) * 2 * 0.1).to(self.device) 
            lego12lego0_pos[2] = -0.0375-0.1
            # random3cm
            # lego12lego0_pos = ((torch.rand(3) - 0.5) * 2 * 0.03).to(self.device) 
            # lego12lego0_pos[2] = -0.0375-0.1

            lego_pose0 = np.frombuffer(self.redis_server.get("latest_pose0"), dtype=np.float32)
            lego_pos0 = torch.from_numpy(lego_pose0[:3]).float().to(self.device)
            lego_rot0 = torch.from_numpy(lego_pose0[3:7]).float().to(self.device)
            lego_pose1 = np.frombuffer(self.redis_server.get("latest_pose1"), dtype=np.float32)
            lego_pos1 = torch.from_numpy(lego_pose1[:3]).float().to(self.device)
            lego_rot1 = torch.from_numpy(lego_pose1[3:7]).float().to(self.device)

            baseur2lego0_target_pos = self.quat_apply(lego_rot1, lego12lego0_pos) + lego_pos1
            # 积木沿y轴负方向移动10cm
            # world2lego0_target_pos = self.quat_apply(self.r_world2baseur_rot, baseur2lego0_target_pos) + self.r_world2baseur_pos
            # world2lego0_target_pos[1] -= 0.10
            # baseur2lego0_target_pos = self.quat_apply(self.quat_conjugate(self.r_world2baseur_rot), world2lego0_target_pos - self.r_world2baseur_pos)

            baseur2lego0_target_quat = self.quat_mul(lego_rot1, lego12lego0_quat)

            baseur2ee_pose0 = self.cal_fk_real_robot("right")
            baseur2ee_pos0 = torch.tensor(baseur2ee_pose0[:3]).float().to(self.device)
            baseur2ee_rot0 = torch.tensor(R.from_rotvec(baseur2ee_pose0[3:6]).as_quat()).float().to(self.device)    

            lego02ee_pos0 = self.quat_apply(self.quat_conjugate(lego_rot0), baseur2ee_pos0 - lego_pos0)
            lego02ee_quat0 = self.quat_mul(self.quat_conjugate(lego_rot0), baseur2ee_rot0)

            baseur2ee_target_pos0 = self.quat_apply(baseur2lego0_target_quat, lego02ee_pos0) + baseur2lego0_target_pos
            baseur2ee_target_quat0 = self.quat_mul(baseur2lego0_target_quat, lego02ee_quat0)
            baseur2ee_target_rotvec0 = R.from_quat(baseur2ee_target_quat0.cpu().numpy()).as_rotvec()
            
            target_dof_pos0 = self.cal_ik_real_robot(baseur2ee_target_pos0.cpu().numpy(), baseur2ee_target_rotvec0, self.rtde_r.getActualQ())
            if target_dof_pos0 is None:
                continue
            else:
                print("current_dof_pos0:", self.rtde_r.getActualQ())
                print("随机化初始化target_dof_pos0:", target_dof_pos0)
                input("press enter to move robot to random pose...")
                self.rtde_c.moveJ(target_dof_pos0, 0.1, 1)
                break

    def reset(self, **kwargs):
        input("please move the robot to the start position, press enter to continue...")
            
        while True:
            reset_flag = input("reset successfully?")
            while reset_flag not in ["1", "0"]:
                reset_flag = input("reset successfully?")
            if reset_flag == "1":
                break
            else:
                self.reconnect_to_robot()
                self.reset_real_robot()
        
        input("open the foundationpose and press enter to continue...")
        # alignment
        # right_target_align_pos = np.array([0.9971511960029602, -0.8422225157367151, 1.0153789520263672, 0.3041379451751709, 1.4818161725997925, 2.655287027359009])
        # self.rtde_c.moveJ(right_target_align_pos, 0.1, 1)
        self.rtde_c.moveJ(self.rh_data[-1][0:6], 0.1, 1)
        self.another_rtde_c.moveJ(self.rh_data[-1][18:24], 0.1, 1)
        self.reset_within_certain_range()
        # for i in range(20):
        # lego1_pose = np.frombuffer(self.redis_server.get("latest_pose1"), dtype=np.float32)#np.array([-0.06876533, -0.00709992, 1.29667835, -0.56338671, 0.49874087, 0.62272573, 0.21462903])
        # lego_rot1 = torch.from_numpy(lego1_pose[3:]).float().to(self.device)
        # lego_pos1 = torch.from_numpy(lego1_pose[:3]).float().to(self.device)
        # lego_target0 = self.quat_apply(lego_rot1, self.ik_target_relative_pos) + lego_pos1
        # lego_quat_target0 = self.quat_mul(lego_rot1, self.ik_target_relative_quat)
        
        # lego0_pose = np.frombuffer(self.redis_server.get("latest_pose0"), dtype=np.float32)
        # lego_pos0 = torch.from_numpy(lego0_pose[:3]).to(self.device)
        # lego_rot0 = torch.from_numpy(lego0_pose[3:]).to(self.device)
        # # 用真机的FK
        # # baseur2ee_pose = self.rtde_c.getForwardKinematics()
        # baseur2ee_pose = self.cal_fk_real_robot("right")
        # baseur2ee_pos = torch.tensor(baseur2ee_pose[:3]).to(self.device)
        # baseur2ee_quat = torch.tensor(R.from_rotvec(baseur2ee_pose[3:]).as_quat()).to(self.device)
        
        # # 如果foundationpose测出来的是baseur2obj
        # baseur2lego0_pos = lego_pos0.clone()
        # baseur2lego0_rot = lego_rot0.clone()
        # lego02ee_pos = self.quat_apply(self.quat_conjugate(baseur2lego0_rot), baseur2ee_pos - baseur2lego0_pos)
        # lego02ee_quat = self.quat_mul(self.quat_conjugate(baseur2lego0_rot), baseur2ee_quat)
        # baseur2ee_target_pos = self.quat_apply(lego_quat_target0, lego02ee_pos) + lego_target0
        # baseur2ee_target_rot = self.quat_mul(lego_quat_target0, lego02ee_quat)

        # baseur2ee_target_rotvec = R.from_quat(baseur2ee_target_rot.cpu().numpy()).as_rotvec()
        # target_joint_pos = self.rtde_c.getInverseKinematics(np.concatenate([baseur2ee_target_pos.cpu().numpy(), baseur2ee_target_rotvec]))#another_rtde_c.getInverseKinematics(np.concatenate([baseur2ee_target_pos.cpu().numpy(), baseur2ee_target_rotvec]))
        # self.rtde_c.moveJ(target_joint_pos, 0.1, 1)
            # if i >= 4:
            #     reset_align_flag = '*'
            #     while reset_align_flag not in ['1', '0']:
            #         reset_align_flag = input("reset alignment successfully?")
            #     if reset_align_flag == '1':
            #         break
        
#----------------------
        obs, world2eemy_pos0, world2eemy_pos1, world2eemy_rot0, world2eemy_rot1, lego_rot0, lego_rot1, lego_pos0, lego_pos1 = self.compute_obs()
        # obs = torch.cat((obs, torch.tensor([0]).float().to(self.device)), dim=-1)
        if self.evaluate != 1:
            wandb.log({'mujoco_reward/reward': self.episode_reward}, step=self.total_steps)
        self.episode_reward = 0
        self.steps = 0
        self.success_flag = False
        return obs.cpu().numpy(), {}

    
    def reset_real_robot(self):
        """
        0:right长程reset
        1:left长程reset
        2:right短程reset手+臂
        3:left短程reset手+臂
        4:right短程reset手
        5:left短程reset手
        6:right短程reset手到小角度
        7:left短程reset手到小角度
        """
        flag =input("do you need remove left hand:")
        while flag not in ["0","1"]:
            flag =input("do you need remove left hand:")
        if flag == "1":
            self.demo_ik()
        for i in range(2):
            flag =input("how to reset right hand:")
            while flag not in ["0","1","2","3","4","5","6","7","8"]:
                flag =input("how to reset right hand:")
            flag = int(flag)
            if flag == 0:
                self.control_right_hand0()
            elif flag == 1:
                self.control_left_hand1()
            elif flag == 2:
                self.control_right_hand2()
            elif flag == 3:
                self.control_left_hand3()
            elif flag == 4:
                self.control_right_hand4()
            elif flag == 5:
                self.control_left_hand5()
            elif flag == 6:
                self.control_right_hand6()
            elif flag == 7:
                self.control_left_hand7()

        if flag == 0:
            self.rtde_c.disconnect()
            for attempt in range(100):
                try:
                    print(f"[RTDE] 尝试第 {attempt+1} 次连接 UR 机器人...")
                    self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
                    print("[RTDE] 成功连接 UR 机器人！")
                    break
                except Exception as e:  # 捕获所有异常
                    print(f"[RTDE] 连接失败: {e}")
                    time.sleep(1)

        for i in range(2):
            flag = input("how to reset left hand:")
            while flag not in ["0","1","2","3","4","5","6","7","8"]:
                flag = input("how to reset left hand:")
            flag = int(flag)
            if flag == 0:
                self.control_right_hand0()
            elif flag == 1:
                self.control_left_hand1()
            elif flag == 2:
                self.control_right_hand2()
            elif flag == 3:
                self.control_left_hand3()
            elif flag == 4:
                self.control_right_hand4()
            elif flag == 5:
                self.control_left_hand5()
            elif flag == 6:
                self.control_right_hand6()
            elif flag == 7:
                self.control_left_hand7()
        if flag == 1:
            self.another_rtde_c.disconnect()
            for attempt in range(100):
                try:
                    print(f"[RTDE] 尝试第 {attempt+1} 次连接 UR 机器人...")
                    self.another_rtde_c = rtde_control.RTDEControlInterface(self.another_robot_ip)
                    print("[RTDE] 成功连接 UR 机器人！")
                    break
                except Exception as e:  # 捕获所有异常
                    print(f"[RTDE] 连接失败: {e}")
                    time.sleep(1)

    def control_left_hand1(self):
        self.another_hand.set_hand_angle(np.array([0, 0, 0, 0, 0, 0]))
        time.sleep(0.5)
        self.another_hand.set_hand_angle(np.array(self.rh_data[0])[[34, 32, 30, 28, 25, 24]])
        self.another_rtde_c.moveJ(self.rh_data[0][18:24], 0.1, 1)
        for i, arm_pose in enumerate(self.rh_data[0:]):
            # inp = input("press enter to continue")
            self.another_rtde_c.servoJ(arm_pose[18:24], 0.8, 0.6, 0.05, 0.2, 600)
            # print("left_hand_target_angle:", np.array(arm_pose)[[34, 32, 30, 28, 25, 24]])
            self.another_hand.set_hand_angle(np.array(arm_pose)[[34, 32, 30, 28, 25, 24]])
            # time.sleep(0.02)
        time.sleep(1)
    def control_right_hand0(self):
        self.hand.set_hand_angle(np.array([0, 0, 0, 0, 0, 0]))
        time.sleep(0.5)
        self.hand.set_hand_angle(np.array(self.rh_data[0])[[16, 14, 12, 10, 7, 6]])
        self.rtde_c.moveJ(self.rh_data[0][0:6], 0.1, 1)
        for i, arm_pose in enumerate(self.rh_data[0:]):
            self.rtde_c.servoJ(arm_pose[0:6], 0.8, 0.6, 0.05, 0.2, 600)
            self.hand.set_hand_angle(np.array(arm_pose)[[16, 14, 12, 10, 7, 6]])
            # time.sleep(0.02)
        time.sleep(1)
    def control_left_hand3(self):
        self.another_rtde_c.moveJ(self.rh_data[-1][18:24], 0.1, 1)#another_rtde_c.servoJ(self.rh_data[799][18:24], 0.2, 0.1, 0.05, 0.2, 600)
        # self.another_hand.set_hand_angle(np.array(self.rh_data[799])[[34, 32, 30, 28, 25, 24]])
    def control_right_hand2(self):
        self.rtde_c.moveJ(self.rh_data[-1][0:6], 1, 1)#rtde_c.servoJ(self.rh_data[799][0:6], 0.2, 0.1, 0.05, 0.2, 600)
        # self.hand.set_hand_angle(np.array(self.rh_data[799])[[16, 14, 12, 10, 7, 6]])
    def control_left_hand5(self):
        self.another_hand.set_hand_angle(np.array(self.rh_data[-1])[[34, 32, 30, 28, 25, 24]]+0.1)
    def control_right_hand4(self):
        self.hand.set_hand_angle(np.array(self.rh_data[-1])[[16, 14, 12, 10, 7, 6]])
    def control_left_hand7(self):
        self.another_rtde_c.moveJ(self.rh_data[-1][18:24], 1, 1)#another_rtde_c.servoJ(self.rh_data[799][18:24], 0.2, 0.1, 0.05, 0.2, 600)
        self.another_hand.set_hand_angle(np.array(self.rh_data[-1])[[34, 32, 30, 28, 25, 24]]-0.25)
    def control_right_hand6(self):
        self.rtde_c.moveJ(self.rh_data[-1][0:6], 1, 1)#rtde_c.servoJ(self.rh_data[799][0:6], 0.2, 0.1, 0.05, 0.2, 600)
        self.hand.set_hand_angle(np.array(self.rh_data[-1])[[16, 14, 12, 10, 7, 6]]-0.15)

    def demo_ik(self):  
        baseur2ee_pose = self.another_rtde_c.getForwardKinematics()
        baseur2ee_pos = np.array(baseur2ee_pose[:3])
        baseur2ee_rot = baseur2ee_pose[3:6]
        lego_pose = np.frombuffer(self.redis_server.get("latest_pose1"), dtype=np.float32)
        lego_rot = lego_pose[3:7]
        r_world2baseur_pos = torch.tensor([-0.8347, -0.2630,  1.4007])
        r_world2baseur_rot = torch.tensor([-0.6552,  0.6515, -0.2687, -0.2720])
        l_world2baseur_pos = torch.tensor([-0.8418,  0.2584,  1.4032])
        l_world2baseur_rot = torch.tensor([-0.6533,  0.6533,  0.2706,  0.2706])
        # baseurr2baseurl_pos = quat_apply(quat_conjugate(r_world2baseur_rot), l_world2baseur_pos - r_world2baseur_pos)
        baseurl2baseurr_rot = self.quat_mul(self.quat_conjugate(l_world2baseur_rot), r_world2baseur_rot)
        baseurl2_lego_rot = self.quat_mul(baseurl2baseurr_rot, lego_rot)
        lego_rot_z = R.from_quat(baseurl2_lego_rot).as_matrix()[:, 2]
        baseur2ee_target_pos = baseur2ee_pos + 0.1 * lego_rot_z
        target_dof_pos = self.another_rtde_c.getInverseKinematics(np.concatenate([baseur2ee_target_pos, baseur2ee_rot]))#self.cal_ik_real_robot(baseur2ee_target_pos, baseur2ee_rot, self.another_rtde_r.getActualQ())#
        self.another_rtde_c.moveJ(target_dof_pos, 0.1, 1)

    def random_point_on_triangle(self, v1, v2, v3):
        """在给定的三角形上生成一个随机点"""
        r1 = np.random.rand()
        r2 = np.random.rand()

        # 计算随机点的重心坐标
        sqrt_r1 = np.sqrt(r1)
        point = (1 - sqrt_r1) * v1 + (sqrt_r1 * (1 - r2)) * v2 + (sqrt_r1 * r2) * v3
        return point

    def generate_uniform_points(self, lego_path, device):
        # 加载 STL 文件
        lego_mesh = tm.load(lego_path)
        # 如果是 Scene 对象，提取第一个 mesh
        if isinstance(lego_mesh, tm.Scene):
            lego_mesh = lego_mesh.dump(concatenate=True)

        # 在模型的每个三角形面上生成随机点
        num_points = 100  # 生成的点数
        lego_points = list(lego_mesh.vertices.tolist())

        # 遍历每个面
        for face in lego_mesh.faces:
            v1, v2, v3 = lego_mesh.vertices[face]
            for _ in range(num_points):  # 为每个三角形生成点
                point = self.random_point_on_triangle(v1, v2, v3)
                lego_points.append(point)

        # 将生成的点转换为 NumPy 数组
        lego_points = torch.from_numpy(np.array(lego_points)/100).float().to(device)
        return lego_points
    
    def sample_points_furthest(self, points, sample_num=1000):
        sampled_points_id = pointnet2_utils.furthest_point_sample(points.reshape(1, *points.shape), sample_num)
        sampled_points = points.index_select(0, sampled_points_id[0].long())
        return sampled_points
    
    def reconnect_to_robot(self, max_retries=100, retry_delay=1.0):
        self.rtde_c.disconnect()
        for attempt in range(max_retries):
            try:
                print(f"[RTDE] 尝试第 {attempt+1} 次连接 UR 机器人...")
                self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
                print("[RTDE] 成功连接 UR 机器人！")
                break
            except Exception as e:  # 捕获所有异常
                print(f"[RTDE] 连接失败: {e}")
                time.sleep(retry_delay)
    
        self.another_rtde_c.disconnect()
        for attempt in range(max_retries):
            try:
                print(f"[RTDE] 尝试第 {attempt+1} 次连接 UR 机器人...")
                self.another_rtde_c = rtde_control.RTDEControlInterface(self.another_robot_ip)
                print("[RTDE] 成功连接 UR 机器人！")
                break
            except Exception as e:  # 捕获所有异常
                print(f"[RTDE] 连接失败: {e}")
                time.sleep(retry_delay)
    
    # def intervene_rotate(self):
    #     obs, world2eemy_pos0, world2eemy_pos1, world2eemy_rot0, world2eemy_rot1, lego_rot0, lego_rot1, lego_pos0, lego_pos1 = self.compute_obs()
    #     qpos0_r = np.array(self.hand.get_hand_angle())[::-1]
    #     qpos0_l = np.array(self.another_hand.get_hand_angle())[::-1]

    #     pos0_r, pos0_l, rot0_r, rot0_l = world2eemy_pos0.clone(), world2eemy_pos1.clone(), world2eemy_rot0.clone(), world2eemy_rot1.clone()
    #     lego1_pose = np.frombuffer(self.redis_server.get("latest_pose1"), dtype=np.float32)#np.array([-0.06876533, -0.00709992, 1.29667835, -0.56338671, 0.49874087, 0.62272573, 0.21462903])
    #     lego_rot1 = torch.from_numpy(lego1_pose[3:]).float().to(self.device)
    #     lego_pos1 = torch.from_numpy(lego1_pose[:3]).float().to(self.device)
    #     lego_target0 = self.quat_apply(lego_rot1, self.ik_target_relative_pos) + lego_pos1
    #     lego_quat_target0 = self.quat_mul(lego_rot1, self.ik_target_relative_quat)
        
    #     lego0_pose = np.frombuffer(self.redis_server.get("latest_pose0"), dtype=np.float32)
    #     lego_pos0 = torch.from_numpy(lego0_pose[:3]).to(self.device)
    #     lego_rot0 = torch.from_numpy(lego0_pose[3:]).to(self.device)
    #     # 用真机的FK
    #     baseur2ee_pose = self.rtde_c.getForwardKinematics()
    #     baseur2ee_pos = torch.tensor(baseur2ee_pose[:3]).to(self.device)
    #     baseur2ee_quat = torch.tensor(R.from_rotvec(baseur2ee_pose[3:]).as_quat()).to(self.device)
        
    #     # 如果foundationpose测出来的是baseur2obj
    #     baseur2lego0_pos = lego_pos0.clone()
    #     baseur2lego0_rot = lego_rot0.clone()
    #     lego02ee_pos = self.quat_apply(self.quat_conjugate(baseur2lego0_rot), baseur2ee_pos - baseur2lego0_pos)
    #     lego02ee_quat = self.quat_mul(self.quat_conjugate(baseur2lego0_rot), baseur2ee_quat)
    #     baseur2ee_target_pos = self.quat_apply(lego_quat_target0, lego02ee_pos) + lego_target0
    #     baseur2ee_target_rot = self.quat_mul(lego_quat_target0, lego02ee_quat)

    #     baseur2ee_target_rotvec = R.from_quat(baseur2ee_target_rot.cpu().numpy()).as_rotvec()
    #     target_joint_pos = self.cal_ik_real_robot(baseur2ee_target_pos.cpu().numpy(), baseur2ee_target_rotvec, self.rtde_r.getActualQ())#self.rtde_c.getInverseKinematics(np.concatenate([baseur2ee_target_pos.cpu().numpy(), baseur2ee_target_rotvec]))#another_rtde_c.getInverseKinematics(np.concatenate([baseur2ee_target_pos.cpu().numpy(), baseur2ee_target_rotvec]))
        
    #     self.rtde_c.moveJ(target_joint_pos, 0.1, 1)
    #     qpos1_r = np.array(self.hand.get_hand_angle())[::-1]
    #     qpos1_l = np.array(self.another_hand.get_hand_angle())[::-1]
    #     next_obs, pos1_r, pos1_l, rot1_r, rot1_l, lego_rot0, lego_rot1, lego_pos0, lego_pos1 = self.compute_obs()

    # def run_updown_leftright(self):
    #     direction = input("please input the direction, x/y/z:")
    #     while direction not in ["x", "y", "z"]:
    #         direction = input("please input the direction, x/y/z:")
    #     direction = int(ord(direction) - ord('x'))
    #     distance = input("please input the moving distance:")#1cm为单位
    #     while not distance.lstrip('-').isdigit():
    #         distance = input("please input the moving distance:")
    #     distance = float(distance)/100
    #     direction_tensor = torch.tensor([0.0, 0.0, 0.0]).to(self.device)
    #     direction_tensor[direction] = distance
        
    #     lego0_pose = np.frombuffer(self.redis_server.get("latest_pose0"), dtype=np.float32)
    #     baseur2lego_pos0 = torch.from_numpy(lego0_pose[:3]).to(self.device)
    #     baseur2lego_rot0 = torch.from_numpy(lego0_pose[3:]).to(self.device)
    #     world2lego_pos0 = self.quat_apply(self.r_world2baseur_rot, baseur2lego_pos0) + self.r_world2baseur_pos
    #     target_world2lego_pos0 = world2lego_pos0.clone() + direction_tensor
    #     target_baseur2lego_pos0 = self.quat_apply(self.quat_conjugate(self.r_world2baseur_rot), target_world2lego_pos0 - self.r_world2baseur_pos)
    #     # 用真机的FK
    #     baseur2ee_pose = self.rtde_c.getForwardKinematics()
    #     baseur2ee_pos = torch.tensor(baseur2ee_pose[:3]).to(self.device)
    #     baseur2ee_quat = torch.tensor(R.from_rotvec(baseur2ee_pose[3:]).as_quat()).to(self.device)
        
    #     # 如果foundationpose测出来的是baseur2obj
    #     lego02ee_pos = self.quat_apply(self.quat_conjugate(baseur2lego_rot0), baseur2ee_pos - baseur2lego_pos0)
    #     lego02ee_quat = self.quat_mul(self.quat_conjugate(baseur2lego_rot0), baseur2ee_quat)
    #     baseur2ee_target_pos = self.quat_apply(baseur2lego_rot0, lego02ee_pos) + target_baseur2lego_pos0
    #     baseur2ee_target_rot = self.quat_mul(baseur2lego_rot0, lego02ee_quat)
    #     # world2ee_target_pos = self.quat_apply(self.r_world2baseur_rot, baseur2ee_target_pos) + self.r_world2baseur_pos
    #     # clamp_target_world2ee_pos = torch.clamp(world2ee_target_pos, self.target_eeur_action_pos_range_lower[:3], self.target_eeur_action_pos_range_upper[:3])
    #     clamp_target_baseur2ee_pos = baseur2ee_target_pos#self.quat_apply(self.quat_conjugate(self.r_world2baseur_rot), clamp_target_world2ee_pos - self.r_world2baseur_pos)
    #     baseur2ee_target_rotvec = R.from_quat(baseur2ee_target_rot.cpu().numpy()).as_rotvec()
    #     target_joint_pos = self.cal_ik_real_robot(clamp_target_baseur2ee_pos.cpu().numpy(), baseur2ee_target_rotvec, self.rtde_r.getActualQ())#self.rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos.cpu().numpy(), baseur2ee_target_rotvec]))#another_rtde_c.getInverseKinematics(np.concatenate([baseur2ee_target_pos.cpu().numpy(), baseur2ee_target_rotvec]))
    #     self.rtde_c.moveJ(target_joint_pos, 0.1, 1)
    
    def collect_demo_ik(self):
        rotate_flag = 1
        data = []
        cnt = 0
        while cnt < 60:
            input("please move the robot to the start position, press enter to continue...")
            
            while True:
                reset_flag = input("reset successfully?")
                while reset_flag not in ["1", "0"]:
                    reset_flag = input("reset successfully?")
                if reset_flag == "1":
                    break
                else:
                    self.reconnect_to_robot()
                    self.reset_real_robot()
            input("finish reset, press enter to continue...")

            # alignment
            # if rotate_flag == 0:
            #     right_target_align_pos = np.array([1.0480901002883911, -0.7790244261371058, 0.9312281608581543, 0.3658221960067749, 1.6303426027297974, 2.592921495437622])
            #     self.rtde_c.moveJ(right_target_align_pos, 0.1, 1)
            # else:
            #     self.rtde_c.moveJ(self.rh_data[-1][0:6], 0.1, 1)
            #     self.another_rtde_c.moveJ(self.rh_data[-1][18:24], 0.1, 1)
                #------------------
            self.reset_within_certain_range()
            episode_data = []
            success_flag = 0
            for i in range(100):
                if rotate_flag == 1:
                    # action_flag = input("rotate0/insert1/manual2/chosen3:")
                    # while action_flag not in ["0", "1", "2", "3"]:
                    #     action_flag = input("rotate0/insert1/manual2/chosen3:")
                    action_flag = input("rotate0/insert1/manual2:")
                    while action_flag not in ["0", "1", "2"]:
                        action_flag = input("rotate0/insert1/manual2:")
                    obs, world2eemy_pos0, world2eemy_pos1, world2eemy_rot0, world2eemy_rot1, lego_rot0, lego_rot1, lego_pos0, lego_pos1 = self.compute_obs()
                    qpos0_r = np.array(self.hand.get_hand_angle())[::-1]
                    qpos0_l = np.array(self.another_hand.get_hand_angle())[::-1]
                    pos0_r, pos0_l, rot0_r, rot0_l = world2eemy_pos0.clone(), world2eemy_pos1.clone(), world2eemy_rot0.clone(), world2eemy_rot1.clone()
                    current_rgb_img = np.frombuffer(self.redis_server.get("color_image"), dtype=np.float32)
                    current_tcp_force_torque = np.concatenate([self.rtde_r.getActualTCPForce(), self.another_rtde_r.getActualTCPForce()])
                    if action_flag == "2":
                        self.intervene_panel.reset()
                        # print("target_difference_rot:", target_difference_rot.cpu().numpy())
                        input("if you finish intervention, press enter to continue...")
                        action_values = self.intervene_panel.get_values().astype(np.float32)
                        values = np.concatenate([action_values[:3]*self.action_pos_range_value, action_values[3:]*self.action_rot_range_value])
                    # elif action_flag == "3":
                    #     direction = input("please input the direction, x/y/z:")
                    #     while direction not in ["x", "y", "z"]:
                    #         direction = input("please input the direction, x/y/z:")
                    #     direction = int(ord(direction) - ord('x'))
                    #     while True:
                    #         distance = input("please input the moving distance:")
                    #         try:
                    #             distance = float(distance)  # 尝试转换为浮点数
                    #             break  # 成功就退出循环
                    #         except ValueError:
                    #             print("Invalid input. Please enter a number.")  # 失败就提示重新输入
                    #     direction_tensor = torch.tensor([0.0, 0.0, 0.0]).to(self.device)
                    #     direction_tensor[direction] = float(distance)

                    #     action_values = np.concatenate([direction_tensor.cpu().numpy().astype(np.float32), np.array([0,0,0]).astype(np.float32)])#self.intervene_panel.get_values().astype(np.float32)
                    #     values = np.concatenate([action_values[:3]*self.action_pos_range_value, action_values[3:]*self.action_rot_range_value])
                    else: 
                        if action_flag == "0":
                            replay_relative_pos = self.ik_target_relative_pos0
                        else:
                            replay_relative_pos = self.ik_target_relative_pos2
                        
                        current_world2eemy_pos0 = world2eemy_pos0.clone()
                        current_world2eemy_pos1 = world2eemy_pos1.clone()
                        current_world2eemy_rot0 = world2eemy_rot0.clone()
                        current_world2eemy_rot1 = world2eemy_rot1.clone()
                        lego1_pose = np.frombuffer(self.redis_server.get("latest_pose1"), dtype=np.float32)#np.array([-0.06876533, -0.00709992, 1.29667835, -0.56338671, 0.49874087, 0.62272573, 0.21462903])
                        lego_rot1 = torch.from_numpy(lego1_pose[3:]).float().to(self.device)
                        lego_pos1 = torch.from_numpy(lego1_pose[:3]).float().to(self.device)
                        lego0_pose = np.frombuffer(self.redis_server.get("latest_pose0"), dtype=np.float32)
                        lego_pos0 = torch.from_numpy(lego0_pose[:3]).to(self.device)
                        lego_rot0 = torch.from_numpy(lego0_pose[3:]).to(self.device)
                        # if action_flag == "0":
                        #     lego_target0 = lego_pos0
                        # else:
                        lego_target0 = self.quat_apply(lego_rot1, replay_relative_pos) + lego_pos1
                        lego_quat_target0 = self.quat_mul(lego_rot1, self.target_relative_quat)
                        
                        # 用真机的FK
                        baseur2ee_pose = self.rtde_c.getForwardKinematics()
                        baseur2ee_pos = torch.tensor(baseur2ee_pose[:3]).to(self.device)
                        baseur2ee_quat = torch.tensor(R.from_rotvec(baseur2ee_pose[3:]).as_quat()).float().to(self.device)
                        # 如果foundationpose测出来的是baseur2obj
                        baseur2lego0_pos = lego_pos0.clone()
                        baseur2lego0_rot = lego_rot0.clone()
                        lego02ee_pos = self.quat_apply(self.quat_conjugate(baseur2lego0_rot), baseur2ee_pos - baseur2lego0_pos)
                        lego02ee_quat = self.quat_mul(self.quat_conjugate(baseur2lego0_rot), baseur2ee_quat)
                        baseur2ee_target_pos = self.quat_apply(lego_quat_target0, lego02ee_pos) + lego_target0
                        baseur2ee_target_rot = self.quat_mul(lego_quat_target0, lego02ee_quat)
                        world2ee_target_pos = self.quat_apply(self.r_world2baseur_rot, baseur2ee_target_pos) + self.r_world2baseur_pos
                        world2ee_target_quat = self.quat_mul(self.r_world2baseur_rot, baseur2ee_target_rot)
                        world2eemy_target_pos = self.quat_apply(world2ee_target_quat, self.ee2eemy_pos) + world2ee_target_pos
                        world2eemy_target_rot = self.quat_mul(world2ee_target_quat, self.ee2eemy_rot)
                        # eecurrent2eetarget
                        target_difference_rot = torch.from_numpy(R.from_quat((self.quat_mul(world2eemy_target_rot, self.quat_conjugate(current_world2eemy_rot0))).cpu().numpy()).as_euler("xyz")).float()
                        target_difference_pos = world2eemy_target_pos - current_world2eemy_pos0#self.quat_apply(self.quat_conjugate(current_world2eemy_rot0), world2eemy_target_pos - current_world2eemy_pos0)
                        real_deploy_differece_rot = torch.clamp(target_difference_rot/self.action_rot_range_value, -1*torch.ones_like(target_difference_rot), 1*torch.ones_like(target_difference_rot))
                        real_deploy_differece_pos = torch.clamp(target_difference_pos/self.action_pos_range_value, -1*torch.ones_like(target_difference_pos), 1*torch.ones_like(target_difference_pos))
                        
                        
                        # self.intervene_panel.reset()
                        # print("target_difference_rot:", target_difference_rot.cpu().numpy())
                        # input("if you finish intervention, press enter to continue...")
                        # action_values = self.intervene_panel.get_values().astype(np.float32)
                        # values = action_values.copy()
                        # values[:3] = action_values[:3] * self.action_pos_range_value
                        # values[3:] = action_values[3:] * self.action_rot_range_value
                        action_values = np.concatenate([real_deploy_differece_pos.cpu().numpy(), real_deploy_differece_rot.cpu().numpy()])
                        values = np.concatenate([real_deploy_differece_pos.cpu().numpy()*self.action_pos_range_value, real_deploy_differece_rot.cpu().numpy()*self.action_rot_range_value])
                        
                    initial_world2eemy_pos0 = world2eemy_pos0.clone()
                    initial_world2eemy_pos1 = world2eemy_pos1.clone()
                    # 右臂6dof
                    target_world2eemy_pos0 = initial_world2eemy_pos0 + torch.from_numpy(values[:3]).to(self.device)
                    world2eemy_euler_err0 = values[3:]
                    world2eemy_quat_err0 = torch.from_numpy(R.from_euler('xyz', world2eemy_euler_err0).as_quat()).float().to(self.device)
                    world2eemy_quat_dt0 = self.quat_mul(self.quat_mul(self.quat_conjugate(world2eemy_rot0), world2eemy_quat_err0), world2eemy_rot0)    
                    target_world2eemy_quat0 = self.quat_mul(world2eemy_rot0, world2eemy_quat_dt0)
                    clamp_target_world2eemy_quat0 = target_world2eemy_quat0
                    target_world2ee_pos0 = self.quat_apply(clamp_target_world2eemy_quat0, self.eemy2ee_pos) + target_world2eemy_pos0
                    clamp_target_world2ee_pos0 = target_world2ee_pos0#torch.clamp(target_world2ee_pos0, self.target_eeur_action_pos_range_lower[:3], self.target_eeur_action_pos_range_upper[:3])
                    clamp_target_baseur2ee_pos0 = self.quat_apply(self.quat_conjugate(self.r_world2baseur_rot), clamp_target_world2ee_pos0 - self.r_world2baseur_pos)
                    clamp_target_world2ee_rot0 = self.quat_mul(clamp_target_world2eemy_quat0, self.eemy2ee_rot)#self.quat_mul(world2eemy_rot0, self.eemy2ee_rot)
                    clamp_target_baseur2ee_rot0 = R.from_quat(self.quat_mul(self.quat_conjugate(self.r_world2baseur_rot), clamp_target_world2ee_rot0).cpu().numpy()).as_rotvec()
                    arm_pos = self.cal_ik_real_robot(clamp_target_baseur2ee_pos0.cpu().numpy(), clamp_target_baseur2ee_rot0, self.rtde_r.getActualQ())#self.rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos0.cpu().numpy(), baseur2ee_rot0]))
                    print("initial_dof_pos0:", self.rtde_r.getActualQ())
                    print("target dof pos0:", arm_pos)
                    input("press enter to continue...")
                    self.rtde_c.moveJ(arm_pos, 0.1, 1)
                    time.sleep(0.02)

                qpos1_r = np.array(self.hand.get_hand_angle())[::-1]
                qpos1_l = np.array(self.another_hand.get_hand_angle())[::-1]
                next_obs, pos1_r, pos1_l, rot1_r, rot1_l, lego_rot0, lego_rot1, lego_pos0, lego_pos1 = self.compute_obs()
                next_rgb_img = np.frombuffer(self.redis_server.get("color_image"), dtype=np.float32)
                next_tcp_force_torque = np.concatenate([self.rtde_r.getActualTCPForce(), self.another_rtde_r.getActualTCPForce()])
                
                action = np.zeros(24)
                action[:3] = action_values[:3]#(self.unscale((pos1_r - pos0_r), self.action_pos_range_lower[:3], self.action_pos_range_upper[:3])).cpu().numpy()
                action[3:6] = action_values[3:6]#(self.unscale(torch.from_numpy(R.from_quat((self.quat_mul(rot1_r, self.quat_conjugate(rot0_r))).cpu().numpy()).as_euler("xyz")).float().to(self.device), self.action_rot_range_lower, self.action_rot_range_upper)).cpu().numpy()
                # action[6:9] = (self.unscale((pos1_l - pos0_l), self.action_pos_range_lower[3:], self.action_pos_range_upper[3:])).cpu().numpy()
                # action[9:12] = (self.unscale(torch.from_numpy(R.from_quat((self.quat_mul(rot1_l, self.quat_conjugate(rot0_l))).cpu().numpy()).as_euler("xyz")).float().to(self.device), self.action_rot_range_lower, self.action_rot_range_upper)).cpu().numpy()
                action[12:18] = (qpos1_r - qpos0_r)/self.hand_action_scale
                action[18:24] = (qpos1_l - qpos0_l)/self.hand_action_scale
                print("action:", action)
                multipoint_lego0_dist, multipoint_lego1_dist = self.compute_point_cloud(lego_rot0, lego_rot1, lego_pos0, lego_pos1)
                rot_dist1, angle_difference1 = self.compute_target_dist(lego_rot0, lego_rot1, lego_pos0, lego_pos1)
                
                success_flag = input("bad:-1;normal:0;success:1")
                if success_flag == "-1":
                    break
                else:
                    demo_data = {"observations": obs, "actions": action, "next_observations": next_obs, "multipoint_lego0_dist": multipoint_lego0_dist, "multipoint_lego1_dist": multipoint_lego1_dist, "rot_dist1": rot_dist1, "dones": False, "masks": True, "success_flag": success_flag, "angle_difference": angle_difference1, "lego_rot0":lego_rot0, "lego_rot1":lego_rot1, "lego_pos0":lego_pos0, "lego_pos1":lego_pos1, "current_rgb_img": current_rgb_img, "next_rgb_img": next_rgb_img, "current_tcp_force_torque": current_tcp_force_torque, "next_tcp_force_torque": next_tcp_force_torque}
                    if success_flag == "1":
                        # episode_data.append(demo_data)
                        # next_obs = next_obs.tolist()
                        # next_obs[-9:-6] = [0.0000, 0.0000, 0.0375]
                        # next_obs[-6:-3] = [1., 0., 0.]
                        # next_obs[-3:] = [0., 0., 0.]
                        demo_data['dones'] = True
                        demo_data['masks'] = False
                        episode_data.append(demo_data)
                        # demo_data = {"observations": next_obs, "actions": np.zeros(24), "next_observations": next_obs, "multipoint_lego0_dist": multipoint_lego0_dist, "multipoint_lego1_dist": multipoint_lego1_dist, "rot_dist1": rot_dist1, "lego_up_height0": lego_up_height0, "lego_up_height1": lego_up_height1, "dones": True, "masks": False, "success_flag": success_flag}
                        # episode_data.extend([demo_data for m in range(i+1, 20)])
                        break
                    episode_data.append(demo_data)
                    print(f"collect demo data: num{cnt} step{i}")
            if success_flag != "-1":
                data.append(episode_data)
                if rotate_flag == 1:#correct_method_withrgbimg_rotate_lowloc_serl_demo_data_{cnt+4}
                    with open(f'/home/admin01/lyw_2/hil-serl/examples/dataset/低高度全程解ik只动右手_前7步转后面插/含有RGB图像受力信息/限制每步action步长/correct_method_withrgbimg_rotate_lowloc_serl_demo_data_{cnt+20}.pkl', 'wb') as f:
                        pickle.dump(episode_data, f) 
                else:
                    with open(f'success_classifier_{cnt}.pkl', 'wb') as f:
                        pickle.dump(episode_data, f) 
                print("collect a demo data!")
                cnt += 1    
            
        # with open(f'ik_realrobot_demo_data.pkl', 'wb') as f:
        #     pickle.dump(data, f)
    
if __name__ == '__main__':
    """
    !!!!world2lego0_pos: tensor([-0.0881, -0.0357,  1.0922], device='cuda:0')
    !!!!world2lego1_pos: tensor([-0.0969,  0.0316,  1.0872], device='cuda:0')
    initial_dof_pos0: [1.054001808166504, -0.4473608175860804, 0.2894291877746582, 0.9450591802597046, 1.5849055051803589, 2.3439486026763916]
    action0: [-1.81547987 -2.34962344 -1.2982676  -0.34044909  1.43396115 -0.59965008]
    """
    # env = BlockAssemblyEnv(False, 1)
    # env.collect_demo_ik()
    # action = np.array([-0.07097571,  1.1056882,  -1.0355694,   0.2993849,  -0.21103689, -0.0387444 ])
    
    # obs, world2eemy_pos0, world2eemy_pos1, world2eemy_rot0, world2eemy_rot1, lego_rot0, lego_rot1, lego_pos0, lego_pos1 = env.compute_obs()

    # action = torch.from_numpy(action).float().to(env.device)
    # initial_world2eemy_pos0 = world2eemy_pos0.clone()
    # initial_world2eemy_pos1 = world2eemy_pos1.clone()
    
    # # 右臂6dof
    # target_world2eemy_pos0 = initial_world2eemy_pos0 + env.scale(action[:3], env.action_pos_range_lower[:3], env.action_pos_range_upper[:3])
    # world2eemy_euler_err0 = env.scale(action[3:6], env.action_rot_range_lower, env.action_rot_range_upper).cpu().numpy()
    # world2eemy_quat_err0 = torch.from_numpy(R.from_euler('xyz', world2eemy_euler_err0).as_quat()).float().to(env.device)
    # world2eemy_quat_dt0 = env.quat_mul(env.quat_mul(env.quat_conjugate(world2eemy_rot0), world2eemy_quat_err0), world2eemy_rot0)
    
    # target_world2eemy_quat0 = env.quat_mul(world2eemy_rot0, world2eemy_quat_dt0)
    # clamp_target_world2eemy_quat0 = target_world2eemy_quat0
    
    # target_world2ee_pos0 = env.quat_apply(clamp_target_world2eemy_quat0, env.eemy2ee_pos) + target_world2eemy_pos0
    # clamp_target_world2ee_pos0 = target_world2ee_pos0#torch.clamp(target_world2ee_pos0, env.target_eeur_action_pos_range_lower[:3], env.target_eeur_action_pos_range_upper[:3])
    
    # clamp_target_baseur2ee_pos0 = env.quat_apply(env.quat_conjugate(env.r_world2baseur_rot), clamp_target_world2ee_pos0 - env.r_world2baseur_pos)
    # world2ee_rot0 = env.quat_mul(clamp_target_world2eemy_quat0, env.eemy2ee_rot)#env.quat_mul(world2eemy_rot0, env.eemy2ee_rot)
    # baseur2ee_rot0 = R.from_quat(env.quat_mul(env.quat_conjugate(env.r_world2baseur_rot), world2ee_rot0).cpu().numpy()).as_rotvec()
    # print("dddL:", np.concatenate([clamp_target_baseur2ee_pos0.cpu().numpy(), baseur2ee_rot0]))
    # current_pos = np.array([1.054001808166504, -0.4473608175860804, 0.2894291877746582, 0.9450591802597046, 1.5849055051803589, 2.3439486026763916])
    # while True:
    #     arm_pos = env.rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos0.cpu().numpy(), baseur2ee_rot0]), current_pos-0.1)
    #     print("---------------------------------------")
    #     print("current_pos: ", current_pos)
    #     print(f"arm_pos: {arm_pos}")
    #     input("press enter to continue")
    #     current_pos -= 0.1
    # [1.0155894756317139, 0.4386006295681, -1.29305100440979, -1.455275297164917, -1.4979195594787598, -0.8262981176376343]
    # qpos = np.array([-1.80527,  -2.22378,  -1.32779, -0.527953,   1.37779, -0.581507])
    # a = env.rtde_c.getForwardKinematics(qpos, [0,0,0,0,0,0])
    # print(a)
    #[-0.2972474694252014, -0.6917456388473511, 0.2626000642776489, 0.36971497535705566, -2.603018045425415, 0.680566668510437]


    # 正确的目标位姿：[-0.30574518 -0.71368343  0.20443916  0.39227033 -2.56836318  0.81568155]
    # print(R.from_rotvec(np.array([0.39227033, -2.56836318,  0.81568155])).as_matrix())
    # print(env.rtde_r.getActualTCPForce(), env.another_rtde_r.getActualTCPForce())
    from examples.experiments.block_assembly.spacemouse import pyspacemouse
    pyspacemouse.open()
    while True:
        state = pyspacemouse.read_all()
        state = state[0]
        print(state.roll, state.pitch, state.yaw)

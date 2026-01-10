 # load policy
# from isaacgym import gymapi
# from isaacgym.torch_utils import *
import numpy as np
import torch
import time
import rtde_control
import rtde_receive
from experiments.block_assembly.inspire_hand_module_modbus_lyw import Hand
import redis

from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import pickle
import trimesh as tm
# from pynput import keyboard
import wandb
import datetime
from ur_analytic_ik import ur5e

import tkinter as tk
from tkinter import ttk
from multiprocessing import Process, Manager
import copy
import pyrealsense2 as rs
import cv2
import jax.numpy as jnp
import jax


class AdmittanceController:
    def __init__(self, mass, damp, stiff, control_period):
        self.mass = mass
        self.damp = damp
        self.stiff = stiff
        self.control_period = control_period


        self.delta_pos = 0
        self.cmd_vel = 0
        self.cmd_acc = 0
    # err_pos, err_vel==0
    def cal_controller_output(self, err_wrench, err_pos, err_vel):
        # 导纳控制核心公式0.1+1/20*(-err_f+200*err_v)
        self.cmd_acc = (1.0 / self.mass) * (
            -err_wrench + self.damp * err_vel + self.stiff * err_pos
        )
        # 离散速度积分
        self.cmd_vel = self.cmd_acc * self.control_period
        # 离散位移增量
        self.delta_pos = self.cmd_vel * self.control_period
          
        return self.delta_pos#, self.cmd_acc, self.cmd_vel
    
# params
dof_lower_limits = [-6.2800, -6.2800, -3.1400, -6.2800, -6.2800, -6.2800,  0.0000,  0.0000, 0.0000,  0.0000,  0.0000,  0.2000, -6.2800, -6.2800, -3.1400, -6.2800, -6.2800, -6.2800,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.2000]
dof_upper_limits = [6.2800, 6.2800, 3.1400, 6.2800, 6.2800, 6.2800, 1.7000, 1.7000, 1.7000, 1.7000, 0.5000, 1.3000, 6.2800, 6.2800, 3.1400, 6.2800, 6.2800, 6.2800, 1.7000, 1.7000, 1.7000, 1.7000, 0.5000, 1.3000]

class BlockAssemblyEnv(gym.Env):
    def __init__(self, fake_env, control_double_arm=False, evaluate=0, save_video=False, classifer=False, offline_train=False):
        # contrl double arm
        self.control_double_arm = control_double_arm
        self.control_single_xyz = True
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(36,), dtype=np.float32)
        if self.control_double_arm:
            self.observation_space = gym.spaces.Dict(
                {
                    "images": gym.spaces.Dict({"shelf": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(128, 128, 3),
                        dtype=np.uint8,
                    ),
                    "ground": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(128, 128, 3),
                        dtype=np.uint8,
                    )}),
                        
                    "state": gym.spaces.Dict(
                        {
                            "rtcp_pose": gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32),
                            "ltcp_pose": gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32),
                            # "tcp_vel": gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                            "r_force": gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                            "l_force": gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                            # "l_force": gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                            # "r_hand_force": gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                            # "l_hand_force": gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),

                        }
                    ),
                }
            )
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "images": gym.spaces.Dict({"shelf": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(128, 128, 3),
                        dtype=np.uint8,
                    ),
                    "ground": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(128, 128, 3),
                        dtype=np.uint8,
                    )}),
                        
                    "state": gym.spaces.Dict(
                        {
                            "tcp_pose": gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32),
                            # "tcp_vel": gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                            "r_force": gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                            # "l_force": gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                            # "r_hand_force": gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                            # "l_hand_force": gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),

                        }
                    ),
                }
            )
            if self.control_single_xyz:
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
            else:
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.agent = None
        self.rng = None
        self.queue = np.zeros(50,)
        self.queue_index = 0
        self.evaluate = evaluate
        self.offline_train = offline_train
        self.classifer = classifer
        # important parameters-------------------
        # control hand
        self.control_hand = False
        
        # baseur和world的关系
        arm_initial_pos = torch.tensor([0.0,0.0,0.0]).to(self.device)#torch.tensor([-0.8625, -0.2400,  1.3946]).to(self.device)
        another_arm_initial_pos = torch.tensor([0.0,0.0,0.0]).to(self.device)#torch.tensor([-0.8625,  0.2400,  1.3946]).to(self.device)
        arm_initial_rot = torch.tensor([ -0.653, -0.653, 0.271, -0.271]).to(self.device)#torch.tensor([ 3.3444e-06, -9.2388e-01,  3.8268e-01, -1.3853e-06]).to(self.device)
        another_arm_initial_rot = torch.tensor([ -0.653, -0.653, -0.271, 0.271]).to(self.device)#torch.tensor([-3.3444e-06, -9.2388e-01, -3.8268e-01, -1.3853e-06]).to(self.device)

        world2arm_trans_right = arm_initial_pos
        world2arm_rot_right = arm_initial_rot
        world2arm_trans_left = another_arm_initial_pos
        world2arm_rot_left = another_arm_initial_rot
        
        r_basemy2baseur_quat = torch.tensor([0.0,0.0,1.0,0.0]).to(self.device)#torch.tensor([ 1.06572167e-03 ,-5.62792849e-04,  7.09452728e-01 ,-7.04751995e-01]).to(self.device)#torch.tensor([-3.60496599e-05, -5.34978704e-06, 7.07114340e-01, -7.07099221e-01]).to(self.device)
        r_basemy2baseur_pos = torch.tensor( [0.0,0.0,0.0]).to(self.device)#torch.tensor( [-0.02776959, -0.02058528,  0.01197951]).to(self.device)#torch.tensor([-0.02073179, 0.01909007, 0.00689948]).to(self.device)
        r_world2basemy_rot = world2arm_rot_right#world2arm_rot_left
        r_world2basemy_pos = world2arm_trans_right#world2arm_trans_left
        self.r_worldold2baseur_pos = self.quat_apply(r_world2basemy_rot, r_basemy2baseur_pos) + r_world2basemy_pos
        self.r_worldold2baseur_rot = self.quat_mul(r_world2basemy_rot, r_basemy2baseur_quat)
        self.r_world2baseur_rot = self.quat_mul(torch.tensor([-0.70710678, 0, 0, 0.70710678]).to(self.device), self.r_worldold2baseur_rot)
        self.r_world2baseur_pos = self.quat_apply(torch.tensor([-0.70710678, 0, 0, 0.70710678]).to(self.device), self.r_worldold2baseur_pos)
        l_basemy2baseur_quat = torch.tensor([0.0,0.0,1.0,0.0]).to(self.device)#torch.tensor([-3.60496599e-05, -5.34978704e-06, 7.07114340e-01, -7.07099221e-01]).to(self.device)
        l_basemy2baseur_pos = torch.tensor([0.0,0.0,0.0]).to(self.device)#torch.tensor([-0.02073179, 0.01909007, 0.00689948]).to(self.device)
        l_world2basemy_rot = world2arm_rot_left
        l_world2basemy_pos = world2arm_trans_left
        self.l_worldold2baseur_pos = self.quat_apply(l_world2basemy_rot, l_basemy2baseur_pos) + l_world2basemy_pos
        self.l_worldold2baseur_rot = self.quat_mul(l_world2basemy_rot, l_basemy2baseur_quat)
        self.l_world2baseur_rot = self.quat_mul(torch.tensor([0.70710678, 0, 0, 0.70710678]).to(self.device), self.l_worldold2baseur_rot)
        self.l_world2baseur_pos = self.quat_apply(torch.tensor([0.70710678, 0, 0, 0.70710678]).to(self.device), self.l_worldold2baseur_pos)
        # ps:左右臂其实分别对应两个世界坐标系，原点分别是左右臂的基坐标系原点，方向是z向左，y向上，x向前
        # 可调参数-------------------------------------
        # 距离完全插入位姿10cm
        if self.control_hand:
            self.init_dof_pos0 = np.array([0.987450361251831, -0.934936825429098, 1.25439453125, 0.02481377124786377, 1.4887490272521973, 2.568765163421631])
            self.init_dof_pos1 = np.array([-1.1089, -2.5020, -0.6554, -3.7734, -1.7296,  0.7556])
            self.init_hand_pos = np.array([5.317746399668977e-06, 1.0040700435638428, 0.9746702313423157, 1.01011061668396, 0.10482560843229294, 1.1795626878738403])
            self.init_baseur2tcp_pos0 = np.array([-0.2722, -0.6226,  0.1909])
            self.init_baseur2tcp_quat0 = np.array([ 0.0239, -0.8168,  0.4995,  0.2878])
        else:
            self.init_dof_pos0 = np.array([0.8021015524864197, -0.9664557615863245, 1.602889060974121, 0.36605024337768555, 2.1278767585754395, 2.575887441635132])
            self.init_dof_pos1 = np.array([-0.836181942616598, -2.1636832396136683, -1.6823161284076136, -3.378989044819967, -2.131158177052633, 0.19614507257938385])
            # self.init_hand_pos = np.array([5.317746399668977e-06, 1.0040700435638428, 0.9746702313423157, 1.01011061668396, 0.10482560843229294, 1.1795626878738403])
            self.init_baseur2tcp_pos0 = np.array([-0.3110, -0.4150,  0.0962])
            self.init_baseur2tcp_quat0 = np.array([-0.1658, -0.9101,  0.0492,  0.3767])
            if self.control_double_arm:
                self.init_dof_pos1 = np.array([-0.9524505774127405, -2.010629955922262, -1.7300599257098597, -3.549246613179342, -2.1965721289264124, 0.08016323298215866])
                self.init_baseur2tcp_pos1 = np.array([ 0.2174, -0.4106,  0.1455])
                self.init_baseur2tcp_quat1 = np.array([0.8582, 0.3255, 0.3684, 0.1477])
        self.init_world2tcp_pos0 = self.quat_apply(self.r_world2baseur_rot, torch.from_numpy(self.init_baseur2tcp_pos0).float().to(self.device)) + self.r_world2baseur_pos
        self.init_world2tcp_quat0 = self.quat_mul(self.r_world2baseur_rot, torch.from_numpy(self.init_baseur2tcp_quat0).float().to(self.device))
        if self.control_double_arm:
            self.init_world2tcp_pos1 = self.quat_apply(self.l_world2baseur_rot, torch.from_numpy(self.init_baseur2tcp_pos1).float().to(self.device)) + self.l_world2baseur_pos
            self.init_world2tcp_quat1 = self.quat_mul(self.l_world2baseur_rot, torch.from_numpy(self.init_baseur2tcp_quat1).float().to(self.device))
        # 随机化和范围参数
        if self.control_single_xyz:
            self.random_pos_range = 0.03
            self.random_rot_range = 0
        else:
            self.random_pos_range = 0.03
            self.random_rot_range = 0.175
        # 世界坐标系下的安全范围（y轴向右）
        self.safety_box_size_min0 = np.array([-self.random_pos_range-0.01, -self.random_pos_range-0.01, -0.1-0.03, -self.random_rot_range-0.05, -self.random_rot_range-0.05, -self.random_rot_range-0.05])
        self.safety_box_size_max0 = np.array([self.random_pos_range+0.01, self.random_pos_range+0.01, 0.01, self.random_rot_range+0.05, self.random_rot_range+0.05, self.random_rot_range+0.05])
        # self.random_pos_range = 0.03
        # self.random_rot_range = 0
        # 世界坐标系下的安全范围（y轴向右）
        # self.safety_box_size_min0 = np.array([-self.random_pos_range-0.01, -self.random_pos_range-0.01, -0.1-0.03, -self.random_rot_range, -self.random_rot_range, -self.random_rot_range])
        # self.safety_box_size_max0 = np.array([self.random_pos_range+0.01, self.random_pos_range+0.01, 0.01, self.random_rot_range, self.random_rot_range, self.random_rot_range])
        if self.control_double_arm:
            self.safety_box_size_min1 = np.array([-self.random_pos_range-0.01, -self.random_pos_range-0.01, -0.01, -self.random_rot_range-0.05, -self.random_rot_range-0.05, -self.random_rot_range-0.05])
            self.safety_box_size_max1 = np.array([self.random_pos_range+0.01, self.random_pos_range+0.01, 0.1+0.03, self.random_rot_range+0.05, self.random_rot_range+0.05, self.random_rot_range+0.05])
        self.random_pos_range = 0
        self.random_rot_range = 0
        # 步长# set same as hilserl peg insertion
        self.action_pos_range_value = 0.005
        self.action_rot_range_value = 0.05
        # 控制频率
        self.hz = 10
        # episode_len
        self.max_episode_steps = 200
        if not fake_env:
            
            from experiments.block_assembly.spacemouse.spacemouse_expert import SpaceMouseExpert
            self.expert = SpaceMouseExpert()
            self.episode_reward = 0
            file = open('/home/admin01/lyw_2/control_ur5/replay_record.pkl', 'rb')
            self.rh_data = np.array(pickle.load(file))
            self.steps = 0
            self.total_steps = 0
            self.robot_ip = "172.16.17.93"
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            if self.control_hand:
                self.hand = Hand(lower_limit=dof_lower_limits[6:12], upper_limit=dof_upper_limits[6:12], port='/dev/ttyUSB0')
            self.another_robot_ip = "172.16.17.92"#"192.168.1.101"#?
            self.another_rtde_r = rtde_receive.RTDEReceiveInterface(self.another_robot_ip)
            self.another_rtde_c = rtde_control.RTDEControlInterface(self.another_robot_ip)
            if self.control_hand:
                self.another_hand = Hand(lower_limit=dof_lower_limits[18:24], upper_limit=dof_upper_limits[18:24], port='/dev/ttyUSB1')
            self.redis_server = redis.Redis(host='localhost', port=6379, db=0)

            if self.evaluate != 1:
                wandb.init(project="hil-serl-realworldRL-actor", name=datetime.datetime.strftime(datetime.datetime.now(), '%m%d_%H%M%S'),
                    save_code=True)
            # init camera
            self.pipeline_shelf, self.aligner_shelf = self.init_camera("102422076572")
            self.pipeline_ground, self.aligner_ground = self.init_camera("406122070453")
            # reset
            self.img_size = gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
            self.first_reset_flag = True
            self.start_time = -1
            self.next_ep_long_reset = False
            self.sequence_random_reset = False # 按照范围大小逐步增加初始随机化程度
            if self.sequence_random_reset:
                self.sequence_random_reset_index = 0
                self.intervene_flag = False
                self.intervene_ep_record_list = []
                self.success_ep_record_list = []
                self.random_pos_min_list = [self.random_pos_range/5*(i) for i in range(5)]
                self.random_pos_max_list = [self.random_pos_range/5*(i+1) for i in range(5)]
                self.random_rot_min_list = [self.random_rot_range/5*(i) for i in range(5)]
                self.random_rot_max_list = [self.random_rot_range/5*(i+1) for i in range(5)]
                self.random_weight_list = [
                    [0.45, 0.30, 0.15, 0.07, 0.03],
                    [0.30, 0.30, 0.20, 0.12, 0.08],
                    [0.18, 0.22, 0.25, 0.20, 0.15],
                    [0.12, 0.18, 0.22, 0.25, 0.23],
                    [0.08, 0.12, 0.18, 0.27, 0.35],
                ]

            # listener = keyboard.Listener(on_press=self.on_press)
            # listener.start()
            # reward
            self.success_flag = []
            self.failure_flag = []
            self.stop_flag = False
            # record video
            self.save_video = save_video
            if self.save_video:
                currtime = datetime.datetime.strftime(datetime.datetime.now(), '%m%d_%H%M%S')
                output = f"/home/admin01/lyw_2/hil-serl_original/videos/record_{currtime}.avi"
                cap = cv2.VideoCapture(0)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.writer = cv2.VideoWriter(output, fourcc, 30, (640, 480))
                import sys
                import signal
                def stop_record(sig, frame):
                    self.writer.release()
                    cap.release()
                    sys.exit(0)

                signal.signal(signal.SIGINT, stop_record)
        # 导纳控制
        self.mass = 10
        self.stiffness = 0
        self.damping = 0
        self.admittance_controller = AdmittanceController(self.mass, self.damping, self.stiffness, 1)
        # 用手指受力进行力控
        self.x_force_positive = 0
        self.x_force_negative = 0
        self.last_x_force_positive = 0
        self.last_x_force_negative = 0
        self.last_left_hand_force = np.array([0, 0, 0, 0, 0, 0])
        # 自动化intervene
        # self.eva_list = [[-0.03, 0.03], [-0.03, -0.03], [0.03, 0.03], [0.03, -0.03]]
        # self.eva_num = 0
        self.auto_intervene = False
        if self.auto_intervene:
            self.slide_window_mode = True
            if self.slide_window_mode:
                self.force_cond = False
                self.collided_idx = -1
                self.slide_window = [i for i in range(20)]
                self.intervened_slide_idx = -1
            else:
                self.deviation_cnt = 0
                self.minidist_point_index_list = [-1 for i in range(25)]
                self.intervene_flag_list = [0, 0, 0]
                self.intervene_min_idx = -1
            with open('/home/admin01/lyw_2/hil-serl_original/data/demos/demo_2026-01-07_14-04-33_new.pkl', 'rb') as f:
                self.demo_data = pickle.load(f)
                cnt = 0
                for i in range(len(self.demo_data)):
                    cnt += 1
                    if self.demo_data[-2-i]["rewards"] == 1:
                        break
                demo_tcp_list = []
                demo_data_list = self.demo_data[-cnt:]
                for data in demo_data_list:
                    demo_tcp_list.append(data['observations']['state'][0][6:])
                self.demo_tcp_list = np.array(demo_tcp_list)
            self.fix_initial_state_demo_tcp_list = copy.deepcopy(self.demo_tcp_list)
           
    # def on_press(self, key):
    #     try:
    #         if isinstance(key, keyboard.Key):
    #             if key == keyboard.Key.esc:
    #                 self.next_ep_long_reset = True
    #             if key == keyboard.Key.space:
    #                 self.success_flag = 1
    #     except AttributeError:
    #         pass
    def select_minidist_point(self, tcp_pose, input_curr_demo_tcp_list):
        curr_demo_tcp_list = copy.deepcopy(input_curr_demo_tcp_list)
        demo_tcp_pos = torch.from_numpy(curr_demo_tcp_list[:, :3]).to(self.device)
        demo_tcp_quat = torch.from_numpy(curr_demo_tcp_list[:, 3:]).to(self.device)
        tcp_pose = torch.from_numpy(tcp_pose).unsqueeze(0).repeat(demo_tcp_pos.shape[0], 1).to(self.device)
        # rot_dist
        quat_diff = self.quat_mul(demo_tcp_quat, self.quat_conjugate(tcp_pose[:, 3:]))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        # trans_dist
        # demo_world2tcp_pos = self.quat_apply(self.r_world2baseur_rot, demo_tcp_pos) + self.r_world2baseur_pos
        # tcp_world2tcp_pos = self.quat_apply(self.r_world2baseur_rot, tcp_pose[:, :3]) + self.r_world2baseur_pos
        # trans_dist = torch.norm(demo_world2tcp_pos[:, :2] - tcp_world2tcp_pos[:, :2], p=2, dim=-1)
        # select_z_dist = torch.abs(demo_world2tcp_pos[:, 2] - tcp_world2tcp_pos[:, 2])
        trans_dist = torch.norm(demo_tcp_pos - tcp_pose[:, :3], p=2, dim=-1)
        min_idx = torch.argmin(trans_dist)
        
        return trans_dist[min_idx], rot_dist[min_idx], min_idx

    def init_camera(self, serial_number):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)

        align_to = rs.stream.color
        aligner = rs.align(align_to)

        # stable realsense
        for i in range(10):
            self.capture_frame(pipeline, aligner)
        return pipeline, aligner

    def capture_frame(self, pipeline, aligner):
        frames = pipeline.wait_for_frames()
        aligned_frames = aligner.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_image_np = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        # self.depth_image_np //= 10

        color_image_np = np.asanyarray(color_frame.get_data())[:, :, :3]
        return color_image_np

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

    def action(self, action: np.ndarray, initial_tcp_pose, initial_tcp_force) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        intervened = False
        
        
        if self.auto_intervene:
            if self.slide_window_mode:
                trans_dist, rot_dist, min_idx = self.select_minidist_point(initial_tcp_pose, self.demo_tcp_list[np.array(self.slide_window)])
                self.force_cond = (abs(initial_tcp_force[0]) > 50) or (abs(initial_tcp_force[1]) > 50) or (abs(initial_tcp_force[2]) > 50)
                print("@@@@@@@@@@L:", self.force_cond)
                print("$$$$$$$$$$$$$:", initial_tcp_force)
                if trans_dist > 0.03:
                    self.intervened_slide_idx = self.slide_window[min_idx]
                if self.force_cond:
                    self.intervened_slide_idx = max(0, self.slide_window[min_idx] - 5)
                    self.collided_idx = copy.deepcopy(self.intervened_slide_idx)
                if (self.collided_idx != -1):
                    target_point = self.demo_tcp_list[self.collided_idx]
                    if np.linalg.norm(target_point[:3] - initial_tcp_pose[:3]) < 0.005:
                        self.intervened_slide_idx = copy.deepcopy(self.collided_idx)
                        self.collided_idx = -1
                if (self.collided_idx == -1) and (self.intervened_slide_idx != -1):
                    target_point = self.demo_tcp_list[min(self.intervened_slide_idx+10, len(self.demo_tcp_list)-1)]
                    if np.linalg.norm(target_point[:3] - initial_tcp_pose[:3]) < 0.005:
                        self.intervened_slide_idx = -1
                if (self.intervened_slide_idx != -1) or (self.collided_idx != -1):
                    curr_baseur2ee_pose0 = np.array(self.cal_fk_real_robot("right"))
                    curr_baseur2ee_pos0 = curr_baseur2ee_pose0[:3].astype(np.float32)
                    curr_baseur2ee_rot0 = R.from_rotvec(curr_baseur2ee_pose0[3:6]).as_quat().astype(np.float32)
                    expert_a = self.ik_solver(target_point, curr_baseur2ee_pos0, curr_baseur2ee_rot0, self.action_pos_range_value, self.action_rot_range_value) 
                    intervened = True
                
                # if (self.intervened_slide_idx != -1) or (self.collided_idx != -1):
                #     if self.collided_idx != -1:
                #         target_point = self.demo_tcp_list[self.collided_idx]
                #     else:
                #         target_point = self.demo_tcp_list[min(self.intervened_slide_idx+10, len(self.demo_tcp_list)-1)]
                #     print("%%%:", target_point[:3])
                #     print("%%%@:", initial_tcp_pose[:3])
                #     print("^^^:", np.linalg.norm(target_point[:3] - initial_tcp_pose[:3]))
                #     if np.linalg.norm(target_point[:3] - initial_tcp_pose[:3]) < 0.005:
                #         print("ddddddddddddddddddddddbbbbbbbbbbbbbbbbbbbbbbbbbbbb!!!!!!!!!!!!")
                #         if self.collided_idx != -1:
                #             self.intervened_slide_idx = copy.deepcopy(self.collided_idx)
                            
                #         else:
                #             self.intervened_slide_idx = -1
                #         self.collided_idx = -1
                        
                #     else:
                #         curr_baseur2ee_pose0 = np.array(self.cal_fk_real_robot("right"))
                #         curr_baseur2ee_pos0 = curr_baseur2ee_pose0[:3].astype(np.float32)
                #         curr_baseur2ee_rot0 = R.from_rotvec(curr_baseur2ee_pose0[3:6]).as_quat().astype(np.float32)
                #         expert_a = self.ik_solver(target_point, curr_baseur2ee_pos0, curr_baseur2ee_rot0) 
                #         intervened = True
                print("!!!!!!!!!!!!!!!!!!!!!:", self.intervened_slide_idx)   
                print("!!!!!!!!!!!!!!!!!!!!!:", self.collided_idx)   

            else:
                # 均是baseur
                trans_dist, rot_dist, min_idx = self.select_minidist_point(initial_tcp_pose, self.demo_tcp_list)
                
                print("trans_dist:", trans_dist, "rot_dist:", rot_dist)
                print("mini_list:", self.minidist_point_index_list)
                # if (trans_dist > 0.02)|(rot_dist > 0.2):
                #     self.deviation_cnt += 1
                cond1 = False#(initial_world2tcp_pos[2] < (separate_world2tcp_pos[2] + 0.01))
                cond2 = (trans_dist > 0.02)#(self.deviation_cnt == 5)
                cond3 = ((self.intervene_flag_list[0] > 0) and (self.intervene_flag_list[1]&self.intervene_flag_list[2] == 0))
                cond4 = (self.minidist_point_index_list[-1] < self.minidist_point_index_list[-20])
                cond5 = ((self.minidist_point_index_list[-20] != -1) and (abs(self.minidist_point_index_list[-1] - self.minidist_point_index_list[-20]) <= 3))
                cond6 = (abs(initial_tcp_force[0]) > 80) or (abs(initial_tcp_force[1]) > 80) or (abs(initial_tcp_force[2]) > 80)
                print("!!!!!!!!!!!!!!!!!!!!!!!!")
                print("initial_tcp_force:", initial_tcp_force)
                print("cond1:", cond1, "cond2:", cond2, "cond3:", cond3, "cond4:", cond4, "cond5:", cond5, "cond6:", cond6)
                
                if cond1 or cond2 or cond3 or cond4 or cond5 or cond6:
                # if (initial_world2tcp_pos[2] < (separate_world2tcp_pos[2] + 0.01)) or (np.linalg.norm(expert_a) > 0.1) or ((self.intervene_flag_list[0] > 0) and (self.intervene_flag_list[1]&self.intervene_flag_list[2] == 0)):
                    # if (self.deviation_cnt == 5) or ((self.deviation_cnt > 0) and (trans_dist <= 0.02) and (rot_dist <= 0.2)):
                    #     self.deviation_cnt = 0
                    if self.intervene_flag_list[0] == 0:
                        self.intervene_min_idx = min_idx
                        if cond6:
                            self.intervene_min_idx = max(0, min_idx-5)
                        self.intervene_flag_list[0] = 1
                        if cond4:
                            self.intervene_flag_list[0] = 2   
                        if cond6:
                            self.intervene_flag_list[0] = 3                                                          
                    if self.intervene_flag_list[1] == 0:
                        target_point = self.demo_tcp_list[self.intervene_min_idx]
                        if np.linalg.norm(target_point[:3] - initial_tcp_pose[:3]) < 0.005:
                            self.intervene_flag_list[1] = 1
                    if self.intervene_flag_list[1] == 1:
                        if (self.intervene_flag_list[0] == 2) or (self.intervene_flag_list[0] == 3):
                            target_point = self.demo_tcp_list[min(self.intervene_min_idx+20, len(self.demo_tcp_list)-1)]
                        else:
                            target_point = self.demo_tcp_list[min(self.intervene_min_idx+10, len(self.demo_tcp_list)-1)]
                        if np.linalg.norm(target_point[:3] - initial_tcp_pose[:3]) < 0.005:
                            self.intervene_flag_list[2] = 1
                    curr_baseur2ee_pose0 = np.array(self.cal_fk_real_robot("right"))
                    curr_baseur2ee_pos0 = curr_baseur2ee_pose0[:3].astype(np.float32)
                    curr_baseur2ee_rot0 = R.from_rotvec(curr_baseur2ee_pose0[3:6]).as_quat().astype(np.float32)
                    expert_a = self.ik_solver(target_point, curr_baseur2ee_pos0, curr_baseur2ee_rot0, self.action_pos_range_value, self.action_rot_range_value)
                    intervened = True
                    if self.intervene_flag_list[2] == 1:
                        self.intervene_flag_list = [0, 0, 0]
                        if len(self.minidist_point_index_list) == 25:
                            self.minidist_point_index_list.pop(0)
                        self.minidist_point_index_list.append(min_idx)
                        print("min_idx:", min_idx)
                        intervened = False
                    # if (initial_world2tcp_pos[2] < (separate_world2tcp_pos[2] + 0.01)):
                    #     intervened = True
                    #     expert_a = self.ik_solver(self.demo_tcp_list[min(min_idx+10, len(self.demo_tcp_list)-1)])

                else:
                    if len(self.minidist_point_index_list) == 25:
                        self.minidist_point_index_list.pop(0)
                    self.minidist_point_index_list.append(min_idx)
                    print("min_idx:", min_idx)
        else:
            if np.linalg.norm(expert_a) > 0.1:#0.001:
                intervened = True

        if intervened:
            if self.control_single_xyz:
                expert_a = expert_a[:3]
            return expert_a, True, buttons[0], buttons[1]
        return action, False, buttons[0], buttons[1]
    
    def safety_box(self, target_world2ee_pos, target_world2ee_quat, init_world2tcp_quat, init_world2tcp_pos, safety_box_size_min, safety_box_size_max, world2baseur_rot, world2baseur_pos):
        # target_world2ee_pos = self.quat_apply(self.r_world2baseur_rot, target_baseur2ee_pos) + self.r_world2baseur_pos
        delta_world2ee_pos = (target_world2ee_pos - init_world2tcp_pos).cpu().numpy()
        clamp_delta_world2ee_pos = np.clip(delta_world2ee_pos, safety_box_size_min[:3], safety_box_size_max[:3])
        clamp_target_world2ee_pos = init_world2tcp_pos + torch.from_numpy(clamp_delta_world2ee_pos).float().to(self.device)
        clamp_target_baseur2ee_pos = self.quat_apply(self.quat_conjugate(world2baseur_rot), clamp_target_world2ee_pos - world2baseur_pos)
        
        # target_world2ee_quat = self.quat_mul(self.r_world2baseur_rot, target_baseur2ee_quat)
        delta_world2ee_euler = R.from_quat((self.quat_mul(target_world2ee_quat, self.quat_conjugate(init_world2tcp_quat))).cpu().numpy()).as_euler("xyz")
        clamp_delta_world2ee_euler = np.clip(delta_world2ee_euler, safety_box_size_min[3:], safety_box_size_max[3:])
        clamp_delta_world2ee_quat = torch.from_numpy(R.from_euler('xyz', clamp_delta_world2ee_euler).as_quat()).float().to(self.device)
        clamp_delta_world2ee_quat_dt = self.quat_mul(self.quat_mul(self.quat_conjugate(init_world2tcp_quat), clamp_delta_world2ee_quat), init_world2tcp_quat)
        clamp_target_world2ee_quat = self.quat_mul(init_world2tcp_quat, clamp_delta_world2ee_quat_dt)
        clamp_target_baseur2ee_quat = self.quat_mul(self.quat_conjugate(world2baseur_rot), clamp_target_world2ee_quat)
        clamp_target_baseur2ee_rotvec = R.from_quat(clamp_target_baseur2ee_quat.cpu().numpy()).as_rotvec()
        return clamp_target_baseur2ee_pos, clamp_target_baseur2ee_rotvec

    # 输入numpy
    def ik_solver(self, target_point, curr_baseur2ee_pos0, curr_baseur2ee_rot0, action_pos_range_value, action_rot_range_value):
        target_baseur2ee_pos0 = torch.from_numpy(target_point[:3]).float().to(self.device)
        target_baseur2ee_rot0 = torch.from_numpy(target_point[3:]).float().to(self.device)
        target_world2ee_pos0 = self.quat_apply(self.r_world2baseur_rot, target_baseur2ee_pos0) + self.r_world2baseur_pos
        target_world2ee_rot0 = self.quat_mul(self.r_world2baseur_rot, target_baseur2ee_rot0)
        
        curr_world2ee_pos0 = self.quat_apply(self.r_world2baseur_rot, torch.from_numpy(curr_baseur2ee_pos0).to(self.device)) + self.r_world2baseur_pos
        curr_world2ee_rot0 = self.quat_mul(self.r_world2baseur_rot, torch.from_numpy(curr_baseur2ee_rot0).to(self.device))
        
        #第一次clamp：将目标位姿和当前位姿的插值除以步长限制在-1~1，保证action是-1~1
        target_difference_rot = torch.from_numpy(R.from_quat((self.quat_mul(target_world2ee_rot0, self.quat_conjugate(curr_world2ee_rot0))).cpu().numpy()).as_euler("xyz")).to(self.device).float()
        target_difference_pos = target_world2ee_pos0 - curr_world2ee_pos0
        clamp_differece_rot = torch.clamp(target_difference_rot/action_rot_range_value, -1*torch.ones_like(target_difference_rot), 1*torch.ones_like(target_difference_rot))
        clamp_differece_pos = torch.clamp(target_difference_pos/action_pos_range_value, -1*torch.ones_like(target_difference_pos), 1*torch.ones_like(target_difference_pos))
        return np.concatenate([clamp_differece_pos.cpu().numpy(), clamp_differece_rot.cpu().numpy()])
        
    def step(self, action):
        
        print("--------------------------step:", self.steps)
        if self.sequence_random_reset:
            print("current hard level:", self.sequence_random_reset_index)
        if self.control_double_arm:
            initial_obs, state_observation = self.compute_obs()
            initial_tcp_pose0 = jax.device_get(state_observation["rtcp_pose"])
            initial_tcp_pose1 = jax.device_get(state_observation["ltcp_pose"])
        else:
            initial_obs, state_observation = self.compute_obs()
            initial_tcp_pose = jax.device_get(state_observation["tcp_pose"])

        info = {}
        bad_data = False
        action, is_intervened, button0, button1 = self.action(action, initial_tcp_pose, jax.device_get(state_observation["r_force"]))
        success, failure = button0, button1
        if len(self.success_flag) == 10:
            self.success_flag.pop(0)
        if success:
            self.success_flag.append(1)
            # self.next_ep_long_reset = True
        else:
            self.success_flag.append(0)

        if len(self.failure_flag) == 10:
            self.failure_flag.pop(0)
        if failure:
            self.failure_flag.append(1)
        else:
            self.failure_flag.append(0)

        if is_intervened:
            info["intervene_action"] = action
            if self.sequence_random_reset:
                self.intervene_flag = True
        # # 导纳控制
        if self.control_hand:
            left_hand_force = np.array(self.another_hand.get_hand_force())
            self.x_force_positive = left_hand_force[1:4][abs(left_hand_force[1:4]) < 100].mean() if len(left_hand_force[1:4][abs(left_hand_force[1:4]) < 100]) > 0 else copy.deepcopy(self.last_x_force_positive)#x正方向的力
            self.x_force_negative = -left_hand_force[-2] if abs(left_hand_force[-2]) < 100 else copy.deepcopy(self.last_x_force_negative) #x负方向的力
            delta_x_force = np.clip(self.x_force_positive - self.last_x_force_positive + self.x_force_negative - self.last_x_force_negative, -10, 10)
            delta_p = self.admittance_controller.cal_controller_output(delta_x_force, 0, 0)
            print("##########$$$$$$$$$$$$$$$$")
            print("@left_hand_force:", left_hand_force)
            print("@delta_p:", delta_p)
            print("@delta_x_force:", delta_x_force)
        # print("@acc:", acc)
        # print("@v:", v)
        action = action.copy()
        # action[0] += delta_p
        action = np.clip(action, self.action_space.low, self.action_space.high)

        action = torch.from_numpy(action).float().to(self.device)

        if self.control_double_arm:
            initial_baseur2ee_pos0 = torch.from_numpy(initial_tcp_pose0[:3].copy()).to(self.device)
            initial_baseur2ee_quat0 = torch.from_numpy(initial_tcp_pose0[3:].copy()).to(self.device)
            initial_world2ee_pos0 = self.quat_apply(self.r_world2baseur_rot, initial_baseur2ee_pos0) + self.r_world2baseur_pos
            initial_world2ee_quat0 = self.quat_mul(self.r_world2baseur_rot, initial_baseur2ee_quat0)
            target_world2ee_pos0 = initial_world2ee_pos0 + action[:3]*self.action_pos_range_value
            world2ee_euler_err0 = (action[3:6]*self.action_rot_range_value).cpu().numpy()#self.scale(action[3:6], self.action_rot_range_lower, self.action_rot_range_upper).cpu().numpy()
            world2ee_quat_err0 = torch.from_numpy(R.from_euler('xyz', world2ee_euler_err0).as_quat()).float().to(self.device)
            world2ee_quat_dt0 = self.quat_mul(self.quat_mul(self.quat_conjugate(initial_world2ee_quat0), world2ee_quat_err0), initial_world2ee_quat0)
            target_world2ee_quat0 = self.quat_mul(initial_world2ee_quat0, world2ee_quat_dt0)
            
            clamp_target_baseur2ee_pos0, clamp_target_baseur2ee_rotvec0 = self.safety_box(target_world2ee_pos0, target_world2ee_quat0, self.init_world2tcp_quat0, self.init_world2tcp_pos0, self.safety_box_size_min0, self.safety_box_size_max0, self.r_world2baseur_rot, self.r_world2baseur_pos)
            arm_pos = self.cal_ik_real_robot(clamp_target_baseur2ee_pos0.cpu().numpy(), clamp_target_baseur2ee_rotvec0, self.rtde_r.getActualQ())#self.rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos0.cpu().numpy(), baseur2ee_rot0]))
            

            initial_baseur2ee_pos1 = torch.from_numpy(initial_tcp_pose1[:3].copy()).to(self.device)
            initial_baseur2ee_quat1 = torch.from_numpy(initial_tcp_pose1[3:].copy()).to(self.device)
            initial_world2ee_pos1 = self.quat_apply(self.l_world2baseur_rot, initial_baseur2ee_pos1) + self.l_world2baseur_pos
            initial_world2ee_quat1 = self.quat_mul(self.l_world2baseur_rot, initial_baseur2ee_quat1)
            target_world2ee_pos1 = initial_world2ee_pos1 + action[6:9]*self.action_pos_range_value
            world2ee_euler_err1 = (action[9:12]*self.action_rot_range_value).cpu().numpy()#self.scale(action[3:6], self.action_rot_range_lower, self.action_rot_range_upper).cpu().numpy()
            world2ee_quat_err1 = torch.from_numpy(R.from_euler('xyz', world2ee_euler_err1).as_quat()).float().to(self.device)
            world2ee_quat_dt1 = self.quat_mul(self.quat_mul(self.quat_conjugate(initial_world2ee_quat1), world2ee_quat_err1), initial_world2ee_quat1)
            target_world2ee_quat1 = self.quat_mul(initial_world2ee_quat1, world2ee_quat_dt1)
            
            clamp_target_baseur2ee_pos1, clamp_target_baseur2ee_rotvec1 = self.safety_box(target_world2ee_pos1, target_world2ee_quat1, self.init_world2tcp_quat1, self.init_world2tcp_pos1, self.safety_box_size_min1, self.safety_box_size_max1, self.l_world2baseur_rot, self.l_world2baseur_pos)
            another_arm_pos = self.cal_ik_real_robot(clamp_target_baseur2ee_pos1.cpu().numpy(), clamp_target_baseur2ee_rotvec1, self.another_rtde_r.getActualQ())#self.rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos1.cpu().numpy(), baseur2ee_rot1]))
            
            if (arm_pos is None) or (another_arm_pos is None):
                bad_data = True
            else:
                print("initial_dof_pos0:", self.rtde_r.getActualQ())
                print("action0:", arm_pos)
                print("initial_dof_pos1:", self.another_rtde_r.getActualQ())
                print("action1:", another_arm_pos)
                print("original action:", action)
                try:
                    if self.stop_flag == False:
                        self.rtde_c.servoJ(arm_pos, 0.4, 0.1, (1.0 / self.hz), 0.2, 600)
                        self.another_rtde_c.servoJ(another_arm_pos, 0.4, 0.1, (1.0 / self.hz), 0.2, 600)

                except RuntimeError as e:
                    print("step_error:", e)
                    input("把手臂移动一下，脱离安全性停止。。。")
                    self.stop_flag = True
                    self.next_ep_long_reset = True
                    self.reconnect_to_robot()
                time.sleep(1.0 / self.hz)
        else:
            initial_baseur2ee_pos0 = torch.from_numpy(initial_tcp_pose[:3].copy()).to(self.device)
            initial_baseur2ee_quat0 = torch.from_numpy(initial_tcp_pose[3:].copy()).to(self.device)
            initial_world2ee_pos0 = self.quat_apply(self.r_world2baseur_rot, initial_baseur2ee_pos0) + self.r_world2baseur_pos
            initial_world2ee_quat0 = self.quat_mul(self.r_world2baseur_rot, initial_baseur2ee_quat0)
            target_world2ee_pos0 = initial_world2ee_pos0 + action[:3]*self.action_pos_range_value
            if self.control_single_xyz:
                target_world2ee_quat0 = initial_world2ee_quat0.clone()
            else:
                world2ee_euler_err0 = (action[3:]*self.action_rot_range_value).cpu().numpy()#self.scale(action[3:6], self.action_rot_range_lower, self.action_rot_range_upper).cpu().numpy()
                world2ee_quat_err0 = torch.from_numpy(R.from_euler('xyz', world2ee_euler_err0).as_quat()).float().to(self.device)
                world2ee_quat_dt0 = self.quat_mul(self.quat_mul(self.quat_conjugate(initial_world2ee_quat0), world2ee_quat_err0), initial_world2ee_quat0)
                target_world2ee_quat0 = self.quat_mul(initial_world2ee_quat0, world2ee_quat_dt0)
            
            clamp_target_baseur2ee_pos0, clamp_target_baseur2ee_rotvec0 = self.safety_box(target_world2ee_pos0, target_world2ee_quat0, self.init_world2tcp_quat0, self.init_world2tcp_pos0, self.safety_box_size_min0, self.safety_box_size_max0, self.r_world2baseur_rot, self.r_world2baseur_pos)
            arm_pos = self.cal_ik_real_robot(clamp_target_baseur2ee_pos0.cpu().numpy(), clamp_target_baseur2ee_rotvec0, self.rtde_r.getActualQ())#self.rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos0.cpu().numpy(), baseur2ee_rot0]))
            if arm_pos is None:
                bad_data = True
            else:
                print("initial_dof_pos0:", self.rtde_r.getActualQ())
                print("action:", arm_pos)
                print("original action:", action)
                # if self.start_time != -1:
                #     dt = time.time() - self.start_time
                #     time.sleep(max(0, (1.0 / self.hz) - dt))
                #     self.rtde_c.stopJ(1, False)
                #     time.sleep(1)
                # self.rtde_c.moveJ(arm_pos, 0.1, 1, True)
                # self.start_time = time.time()
                try:
                    if self.stop_flag == False:
                        # self.rtde_c.servoJ(arm_pos, 0.4*(1 - np.clip(3*abs(delta_p), 0, 1)), 0.1*(1 - np.clip(3*abs(delta_p), 0, 1)), (1.0 / self.hz), 0.2, 600)
                        self.rtde_c.servoJ(arm_pos, 0.4, 0.1, (1.0 / self.hz), 0.2, 600)
                        # self.rtde_c.moveJ(arm_pos, 0.1, 1)

                except RuntimeError as e:
                    print("step_error:", e)
                    input("把手臂移动一下，脱离安全性停止。。。")
                    self.stop_flag = True
                    self.next_ep_long_reset = True
                    self.reconnect_to_robot()
                time.sleep(1.0 / self.hz)
        next_obs, next_state_observation = self.compute_obs()
        
        self.steps += 1
        self.total_steps += 1
        if self.auto_intervene:
            if self.slide_window_mode:
                if (self.intervened_slide_idx == -1) and (self.collided_idx == -1):
                    self.slide_window.pop(0)
                    self.slide_window.append(min(self.slide_window[-1]+1, len(self.demo_tcp_list)-1))
        if (len(self.success_flag) >= 2) and (self.success_flag[-1] == 1) and (self.success_flag[-2] == 1):
            reward = 1
        else:
            reward = 0
        # if self.auto_intervene:
        success_tcp_pos = np.array([-0.3818, -0.4092,  0.0219])#self.demo_tcp_list[-1][:3]
        next_tcp_pos = jax.device_get(next_state_observation["tcp_pose"])[:3]
        delta_pos = np.linalg.norm(next_tcp_pos - success_tcp_pos)
        if delta_pos < 0.01:
            reward = 1
        if (len(self.failure_flag) >= 2) and (self.failure_flag[-1] == 1) and (self.failure_flag[-2] == 1):
            self.stop_flag = True
        
        if self.classifer:
            done = (self.steps == self.max_episode_steps)
        else:
            done = (self.steps == self.max_episode_steps)|(reward == 1)|(self.stop_flag == True)
        
        self.episode_reward += reward
        info["bad_data"] = bad_data
        
        return next_obs, reward, done, done, info

    def cal_fk_real_robot(self, hand_str, max_retries=100, retry_delay=1):
        for attempt in range(max_retries):
            try:
                if hand_str == "right":
                    baseur2ee_pose = self.rtde_c.getForwardKinematics()
                else:
                    baseur2ee_pose = self.another_rtde_c.getForwardKinematics()
                break  # 成功则跳出循环
                
            except RuntimeError as e:
                print("calfk_error:", e)
                self.stop_flag = True
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
                if (abs(solution[s] - current_qpos[s]) < 2):
                    choose_solution.append(solution[s])
                else:
                    new_s0 = solution[s].copy()
                    new_s1 = solution[s].copy()

                    new_s0 += 6.28
                    if (abs(new_s0 - current_qpos[s]) < 2):
                        choose_solution.append(new_s0)
                    else:
                        new_s1 -= 6.28
                        if (abs(new_s1 - current_qpos[s]) < 2):
                            choose_solution.append(new_s1)

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
    
    def process_image(self, image) -> np.ndarray:
        cropped_rgb = image[:, 80:560, :]
        resized = cv2.resize(cropped_rgb, self.img_size.shape[:2][::-1])
        processed_img = resized#[..., ::-1]
        return processed_img
        
    def get_img(self):
        shelf_img = self.capture_frame(self.pipeline_shelf, self.aligner_shelf)
        if self.save_video:
            self.writer.write(shelf_img)
        ground_img = self.capture_frame(self.pipeline_ground, self.aligner_ground)

        # combined = np.hstack([cv2.cvtColor(shelf_img.reshape(480, 640, 3).astype(np.uint8), cv2.COLOR_BGR2RGB), cv2.cvtColor(ground_img.reshape(480, 640, 3).astype(np.uint8), cv2.COLOR_BGR2RGB)])
        
        processed_shelf_img = self.process_image(shelf_img)
        processed_ground_img = self.process_image(ground_img)
        combined = np.hstack([processed_shelf_img, processed_ground_img])
        
        cv2.imshow("Two Cameras", combined)
        cv2.waitKey(1)

        images = {"shelf": processed_shelf_img, "ground": processed_ground_img}
        return images

    def get_state(self):
        baseur2ee_pose0 = np.array(self.cal_fk_real_robot("right"))
        baseur2ee_pos0 = baseur2ee_pose0[:3].astype(np.float32)
        baseur2ee_rot0 = R.from_rotvec(baseur2ee_pose0[3:6]).as_quat().astype(np.float32)
        tcp_pose0 = jnp.array(np.concatenate([baseur2ee_pos0, baseur2ee_rot0]))

        if self.control_double_arm:
            baseur2ee_pose1 = np.array(self.cal_fk_real_robot("left"))
            baseur2ee_pos1 = baseur2ee_pose1[:3].astype(np.float32)
            baseur2ee_rot1 = R.from_rotvec(baseur2ee_pose1[3:6]).as_quat().astype(np.float32)
            tcp_pose1 = jnp.array(np.concatenate([baseur2ee_pos1, baseur2ee_rot1]))
        r_speed = jnp.array(self.rtde_r.getActualTCPSpeed())

        r_force = jnp.array(self.rtde_r.getActualTCPForce())
        l_force = jnp.array(self.another_rtde_r.getActualTCPForce())
        if self.control_hand:
            r_hand_force = jnp.array(np.clip(self.hand.get_hand_force(), -50, 50))
            l_hand_force = jnp.array(np.clip(self.another_hand.get_hand_force(), -50, 50))

        if self.control_double_arm:
            state_observation = {
                "rtcp_pose": tcp_pose0,
                "ltcp_pose": tcp_pose1,
                # "tcp_vel": r_speed,
                "r_force": r_force,#force&torque
                "l_force": l_force,
                # "r_hand_force": r_hand_force,
                # "l_hand_force": l_hand_force
            }
        else:
            state_observation = {
                "tcp_pose": tcp_pose0,
                # "tcp_vel": r_speed,
                "r_force": r_force,#force&torque
                # "l_force": l_force,
                # "r_hand_force": r_hand_force,
                # "l_hand_force": l_hand_force
            }
        
        return state_observation

    def compute_obs(self):
        images = self.get_img()
        state_observation = self.get_state()

        return copy.deepcopy(dict(images=images, state=state_observation)), state_observation#, jax.device_get(state_observation["r_force"]), jax.device_get(state_observation["tcp_vel"])

    # 输入：随机化的初始目标位姿
    def augment_demo(self, target_baseur2tcp_pos, target_baseur2tcp_quat):
        # input:numpy
        tcp_pose = np.concatenate([target_baseur2tcp_pos, target_baseur2tcp_quat])
        _, _, min_idx = self.select_minidist_point(tcp_pose, self.fix_initial_state_demo_tcp_list)
        aug_trajs = []
        rot_dist = 100
        trans_dist = 100
        curr_baseur2tcp_pos = copy.deepcopy(target_baseur2tcp_pos)
        curr_baseur2tcp_quat = copy.deepcopy(target_baseur2tcp_quat)
        aug_trajs.append(np.concatenate([curr_baseur2tcp_pos, curr_baseur2tcp_quat]))
        mini_demo_point = self.fix_initial_state_demo_tcp_list[min_idx]
        while (rot_dist > 0.1) or (trans_dist > 0.001):
            action = self.ik_solver(mini_demo_point, curr_baseur2tcp_pos, curr_baseur2tcp_quat, 0.00173, self.action_rot_range_value)
            if self.control_single_xyz:
                action = action[:3]
            action = np.clip(action, self.action_space.low, self.action_space.high)
            action = torch.from_numpy(action).float().to(self.device)
            initial_tcp_pose = np.concatenate([curr_baseur2tcp_pos, curr_baseur2tcp_quat])

            initial_baseur2ee_pos0 = torch.from_numpy(initial_tcp_pose[:3].copy()).to(self.device)
            initial_baseur2ee_quat0 = torch.from_numpy(initial_tcp_pose[3:].copy()).to(self.device)
            initial_world2ee_pos0 = self.quat_apply(self.r_world2baseur_rot, initial_baseur2ee_pos0) + self.r_world2baseur_pos
            initial_world2ee_quat0 = self.quat_mul(self.r_world2baseur_rot, initial_baseur2ee_quat0)
            target_world2ee_pos0 = initial_world2ee_pos0 + action[:3]*0.00173#self.action_pos_range_value
            print("delta_world:", torch.norm(target_world2ee_pos0 - initial_world2ee_pos0, p=2, dim=-1).item())
            if self.control_single_xyz:
                target_world2ee_quat0 = initial_world2ee_quat0.clone()
            else:
                world2ee_euler_err0 = (action[3:]*self.action_rot_range_value).cpu().numpy()#self.scale(action[3:6], self.action_rot_range_lower, self.action_rot_range_upper).cpu().numpy()
                world2ee_quat_err0 = torch.from_numpy(R.from_euler('xyz', world2ee_euler_err0).as_quat()).float().to(self.device)
                world2ee_quat_dt0 = self.quat_mul(self.quat_mul(self.quat_conjugate(initial_world2ee_quat0), world2ee_quat_err0), initial_world2ee_quat0)
                target_world2ee_quat0 = self.quat_mul(initial_world2ee_quat0, world2ee_quat_dt0)
            
            clamp_target_baseur2ee_pos0, clamp_target_baseur2ee_rotvec0 = self.safety_box(target_world2ee_pos0, target_world2ee_quat0, self.init_world2tcp_quat0, self.init_world2tcp_pos0, self.safety_box_size_min0, self.safety_box_size_max0, self.r_world2baseur_rot, self.r_world2baseur_pos)
            print("delta_baseur:", torch.norm(clamp_target_baseur2ee_pos0 - initial_baseur2ee_pos0, p=2, dim=-1).item())
            curr_baseur2tcp_pos = clamp_target_baseur2ee_pos0.cpu().numpy()
            curr_baseur2tcp_quat = R.from_rotvec(clamp_target_baseur2ee_rotvec0).as_quat().astype(np.float32)
            aug_trajs.append(np.concatenate([curr_baseur2tcp_pos, curr_baseur2tcp_quat]))
            # rot_dist
            quat_diff = self.quat_mul(torch.from_numpy(curr_baseur2tcp_quat), self.quat_conjugate(torch.from_numpy(mini_demo_point[3:])))
            rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[0:3], p=2, dim=-1), max=1.0))
            # trans_dist
            trans_dist = np.linalg.norm(curr_baseur2tcp_pos - mini_demo_point[:3])
            print("action:", action)
            print("rot_dist:", rot_dist.item(), "trans_dist:", trans_dist)
            print("dist:", np.linalg.norm(aug_trajs[-1][:3] - aug_trajs[-2][:3]))
        for i in range(min_idx+1, len(self.fix_initial_state_demo_tcp_list)):
            aug_trajs.append(self.fix_initial_state_demo_tcp_list[i])
        print("min_idx:", min_idx)
        return np.array(aug_trajs)

    def reset_within_certain_range(self, init_world2tcp_quat, init_world2tcp_pos, world2baseur_rot, world2baseur_pos, rtde_r, rtde_c):
        random_rot_min = 0
        random_rot_max = self.random_rot_range
        random_pos_min = 0
        random_pos_max = self.random_pos_range

        if self.sequence_random_reset:
            if (self.intervene_ep_record_list[-5:] == [0, 0, 0, 0, 0]) and (self.success_ep_record_list[-5:] == [1, 1, 1, 1, 1]):
                self.sequence_random_reset_index = min(4, self.sequence_random_reset_index + 1)
            choice = np.random.choice(len(self.random_weight_list[self.sequence_random_reset_index]), p=self.random_weight_list[self.sequence_random_reset_index])
            random_rot_min = self.random_rot_min_list[choice]
            random_pos_min = self.random_pos_min_list[choice]
            random_rot_max = self.random_rot_max_list[choice]
            random_pos_max = self.random_pos_max_list[choice]
        while True:
            # random_angle = np.random.uniform(low=-self.random_rot_range, high=self.random_rot_range, size=3)
            # 幅值范围 [min, max]
            rot_mag = np.random.uniform(
                low=random_rot_min,
                high=random_rot_max,
                size=3
            )

            # 随机符号 ±1
            rot_sign = np.random.choice([-1.0, 1.0], size=3)

            random_angle = rot_mag * rot_sign

            tcp02tcp1_quat = torch.from_numpy(R.from_euler("xyz", random_angle).as_quat()).float().to(self.device)
            tcp02tcp1_quat_dt = self.quat_mul(self.quat_mul(self.quat_conjugate(init_world2tcp_quat), tcp02tcp1_quat), init_world2tcp_quat)
            
            # tcp02tcp1_pos = ((torch.rand(3) - 0.5) * 2 * self.random_pos_range).to(self.device) 
            # 幅值 [min, max]
            pos_mag = torch.empty(3, device=self.device).uniform_(
                random_pos_min, random_pos_max
            )

            # 随机符号 ±1
            pos_sign = torch.randint(0, 2, (3,), device=self.device) * 2 - 1

            tcp02tcp1_pos = pos_mag * pos_sign
            tcp02tcp1_pos[2] = 0 # 在world坐标系下的z方向上不动，在xy平面上动
            # tcp02tcp1_pos[0] = self.eva_list[self.eva_num][0] # 在world坐标系下的z方向上不动，在xy平面上动
            # tcp02tcp1_pos[1] =  self.eva_list[self.eva_num][1]# 在world坐标系下的z方向上不动，在xy平面上动
            # self.eva_num = (self.eva_num + 1) % 4

            target_world2tcp_pos = (init_world2tcp_pos + tcp02tcp1_pos)
            target_world2tcp_quat = (self.quat_mul(init_world2tcp_quat, tcp02tcp1_quat_dt))
            
            target_baseur2tcp_pos = self.quat_apply(self.quat_conjugate(world2baseur_rot), target_world2tcp_pos - world2baseur_pos).cpu().numpy()
            target_baseur2tcp_quat = self.quat_mul(self.quat_conjugate(world2baseur_rot), target_world2tcp_quat).cpu().numpy()
            target_baseur2tcp_rotvec = R.from_quat(target_baseur2tcp_quat).as_rotvec()

            
            target_dof_pos0 = self.cal_ik_real_robot(target_baseur2tcp_pos, target_baseur2tcp_rotvec, rtde_r.getActualQ())
            if target_dof_pos0 is None:
                continue
            else:
                print("current_dof_pos0:", rtde_r.getActualQ())
                print("随机化初始化target_dof_pos0:", target_dof_pos0)
                rtde_c.moveJ(target_dof_pos0, 0.1, 1)
                # if self.auto_intervene:
                #     if (self.random_pos_range != 0) or (self.random_rot_range != 0):
                #         self.demo_tcp_list = self.augment_demo(target_baseur2tcp_pos, target_baseur2tcp_quat)
                break

    def reset(self, **kwargs):
        if self.control_hand:
            if self.first_reset_flag|self.next_ep_long_reset:
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
                self.first_reset_flag = False
                self.next_ep_long_reset = False
        self.reconnect_to_robot()   
        obs, _ = self.compute_obs() 
        self.rtde_c.moveJ(self.init_dof_pos0, 0.05, 0.5)
        self.another_rtde_c.moveJ(self.init_dof_pos1, 0.05, 0.5)
        # input("please move the robot to the start position, press enter to continue...")

        self.reset_within_certain_range(self.init_world2tcp_quat0, self.init_world2tcp_pos0, self.r_world2baseur_rot, self.r_world2baseur_pos, self.rtde_r, self.rtde_c)
        if self.control_double_arm:
            self.reset_within_certain_range(self.init_world2tcp_quat1, self.init_world2tcp_pos1, self.l_world2baseur_rot, self.l_world2baseur_pos, self.another_rtde_r, self.another_rtde_c)
        obs, _ = self.compute_obs()
        if self.evaluate != 1:
            wandb.log({'mujoco_reward/reward': self.episode_reward}, step=self.total_steps)
        if self.sequence_random_reset:
            if len(self.success_ep_record_list) == 10:
                self.success_ep_record_list.pop(0)
            if self.episode_reward > 0:
                self.success_ep_record_list.append(1)
            else:
                self.success_ep_record_list.append(0)
        self.episode_reward = 0
        self.steps = 0
        print("!!!!!!!!!!!!!!!!!!!!!episode reset!!!!!!!!!!!!!!!!!!!!!")
        self.success_flag = []
        self.failure_flag = []
        self.next_ep_long_reset = False
        self.stop_flag = False
        if self.sequence_random_reset:
            if len(self.intervene_ep_record_list) == 10:
                self.intervene_ep_record_list.pop(0)
            if self.intervene_flag:
                self.intervene_ep_record_list.append(1)
            else:
                self.intervene_ep_record_list.append(0)
            self.intervene_flag = False
        if self.control_hand:
            left_hand_force = np.array(self.another_hand.get_hand_force())
            self.x_force_positive = left_hand_force[1:4][abs(left_hand_force[1:4]) < 100].mean() if len(left_hand_force[1:4][abs(left_hand_force[1:4]) < 100]) > 0 else left_hand_force[1:4].mean()#x正方向的力 
            self.x_force_negative = -left_hand_force[-2] #x负方向的力
            
            self.last_x_force_positive = copy.deepcopy(self.x_force_positive)
            self.last_x_force_negative = copy.deepcopy(self.x_force_negative)
            self.last_left_hand_force = copy.deepcopy(left_hand_force)
        if self.auto_intervene:
            if self.slide_window_mode:
                self.slide_window = [i for i in range(20)]
                self.intervened_slide_idx = -1
                self.collided_idx = -1
            else:
                self.minidist_point_index_list = [-1 for i in range(25)]
                self.intervene_flag_list = [0,0,0]
                self.deviation_cnt = 0
        return obs, {}

    
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
            while flag not in ["4","5","6","7","8"]:
                flag =input("how to reset right hand:")
            flag = int(flag)
            # if flag == 0:
            #     self.control_right_hand0()
            # elif flag == 1:
            #     self.control_left_hand1()
            # elif flag == 2:
            #     self.control_right_hand2()
            # elif flag == 3:
            #     self.control_left_hand3()
            if flag == 4:
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
            while flag not in ["4","5","6","7","8"]:
                flag = input("how to reset left hand:")
            flag = int(flag)
            # if flag == 0:
            #     self.control_right_hand0()
            # elif flag == 1:
            #     self.control_left_hand1()
            # elif flag == 2:
            #     self.control_right_hand2()
            # elif flag == 3:
            #     self.control_left_hand3()
            if flag == 4:
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

    def set_hand_angle(self, str_hand, angle):
        while True:
            try:
                if str_hand == "right":
                    self.hand.set_hand_angle(angle)
                else:
                    self.another_hand.set_hand_angle(angle)
                break
            except RuntimeError as e:
                print(f"[WARN] {str_hand} hand control failed: {e}!!!!!!!!!!!!!!!!!!!!!")
                if str_hand == "right":
                    self.hand = Hand(lower_limit=dof_lower_limits[6:12], upper_limit=dof_upper_limits[6:12], port='/dev/ttyUSB0')
                else:
                    self.another_hand = Hand(lower_limit=dof_lower_limits[18:24], upper_limit=dof_upper_limits[18:24], port='/dev/ttyUSB1')

    # def control_left_hand1(self):
    #     self.another_hand.set_hand_angle(np.array([0, 0, 0, 0, 0, 0]))
    #     time.sleep(0.5)
    #     self.another_hand.set_hand_angle(np.array(self.rh_data[0])[[34, 32, 30, 28, 25, 24]])
    #     self.another_rtde_c.moveJ(self.rh_data[0][18:24], 0.1, 1)
    #     for i, arm_pose in enumerate(self.rh_data[0:]):
    #         # inp = input("press enter to continue")
    #         self.another_rtde_c.servoJ(arm_pose[18:24], 0.8, 0.6, 0.05, 0.2, 600)
    #         # print("left_hand_target_angle:", np.array(arm_pose)[[34, 32, 30, 28, 25, 24]])
    #         self.another_hand.set_hand_angle(np.array(arm_pose)[[34, 32, 30, 28, 25, 24]])
    #         # time.sleep(0.02)
    #     time.sleep(1)
    # def control_right_hand0(self):
    #     self.hand.set_hand_angle(np.array([0, 0, 0, 0, 0, 0]))
    #     time.sleep(0.5)
    #     self.hand.set_hand_angle(np.array(self.rh_data[0])[[16, 14, 12, 10, 7, 6]])
    #     self.rtde_c.moveJ(self.rh_data[0][0:6], 0.1, 1)
    #     for i, arm_pose in enumerate(self.rh_data[0:]):
    #         self.rtde_c.servoJ(arm_pose[0:6], 0.8, 0.6, 0.05, 0.2, 600)
    #         self.hand.set_hand_angle(np.array(arm_pose)[[16, 14, 12, 10, 7, 6]])
    #         # time.sleep(0.02)
    #     time.sleep(1)
    # def control_left_hand3(self):
    #     self.another_rtde_c.moveJ(self.rh_data[-1][18:24], 0.1, 1)#another_rtde_c.servoJ(self.rh_data[799][18:24], 0.2, 0.1, 0.05, 0.2, 600)
    #     # self.another_hand.set_hand_angle(np.array(self.rh_data[799])[[34, 32, 30, 28, 25, 24]])
    # def control_right_hand2(self):
    #     self.rtde_c.moveJ(self.rh_data[-1][0:6], 1, 1)#rtde_c.servoJ(self.rh_data[799][0:6], 0.2, 0.1, 0.05, 0.2, 600)
    #     # self.hand.set_hand_angle(np.array(self.rh_data[799])[[16, 14, 12, 10, 7, 6]])
    def control_left_hand5(self):
        new_pos = self.init_hand_pos.copy()
        new_pos[3] += 0.03
        new_pos[1] += 0.13
        # self.another_hand.set_hand_angle(new_pos+0.05)
        self.set_hand_angle("left", new_pos+0.05)
    def control_right_hand4(self):
        # self.hand.set_hand_angle(self.init_hand_pos+0.05)
        self.set_hand_angle("right", self.init_hand_pos+0.05)
    def control_left_hand7(self):
        self.another_rtde_c.moveJ(self.init_dof_pos1, 1, 1)#another_rtde_c.servoJ(self.rh_data[799][18:24], 0.2, 0.1, 0.05, 0.2, 600)
        new_pos = self.init_hand_pos.copy()
        new_pos -= 0.25
        new_pos[-1] += 0.15
        # self.another_hand.set_hand_angle(new_pos)
        self.set_hand_angle("left", new_pos)
    def control_right_hand6(self):
        self.rtde_c.moveJ(self.init_dof_pos0, 1, 1)#rtde_c.servoJ(self.rh_data[799][0:6], 0.2, 0.1, 0.05, 0.2, 600)
        # self.hand.set_hand_angle(self.init_hand_pos-0.15)
        self.set_hand_angle("right", self.init_hand_pos-0.15)

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
    
    # def sample_points_furthest(self, points, sample_num=1000):
    #     sampled_points_id = pointnet2_utils.furthest_point_sample(points.reshape(1, *points.shape), sample_num)
    #     sampled_points = points.index_select(0, sampled_points_id[0].long())
    #     return sampled_points
    
    def reconnect_to_robot(self, max_retries=100, retry_delay=1.0):
        self.rtde_c.disconnect()
        for attempt in range(max_retries):
            try:
                print(f"[RTDE] 尝试第 {attempt+1} 次连接 UR 机器人...")
                self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
                print("[RTDE] 成功连接 UR 机器人！")
                break
            except RuntimeError as e:
                print("right_reconnect_error:", e)
                input("把手臂移动一下，脱离安全性停止。。。")
    
        self.another_rtde_c.disconnect()
        for attempt in range(max_retries):
            try:
                print(f"[RTDE] 尝试第 {attempt+1} 次连接 UR 机器人...")
                self.another_rtde_c = rtde_control.RTDEControlInterface(self.another_robot_ip)
                print("[RTDE] 成功连接 UR 机器人！")
                break
            except RuntimeError as e:
                print("left_reconnect_error:", e)
                input("把手臂移动一下，脱离安全性停止。。。")
    
    
if __name__ == '__main__':
    """
    !!!!world2lego0_pos: tensor([-0.0881, -0.0357,  1.0922], device='cuda:0')
    !!!!world2lego1_pos: tensor([-0.0969,  0.0316,  1.0872], device='cuda:0')
    initial_dof_pos0: [1.054001808166504, -0.4473608175860804, 0.2894291877746582, 0.9450591802597046, 1.5849055051803589, 2.3439486026763916]
    action0: [-1.81547987 -2.34962344 -1.2982676  -0.34044909  1.43396115 -0.59965008]
    """
    # env = BlockAssemblyEnv(False, 1)
    # while True:
    #     print(env.rtde_r.getActualTCPForce())
    #     time.sleep(0.1)
    #     a = input("press enter to continue:")
    #     if a == "r":
    #         env.reconnect_to_robot()
    # env = BlockAssemblyEnv(False, 1)
    # fk_pose = env.cal_fk_real_robot("right")
    # print("fk_pose:", fk_pose)
    # ik_pos = env.cal_ik_real_robot(fk_pose[:3], fk_pose[3:6], env.rtde_r.getActualQ())
    # print("ik_pos:", ik_pos)
    # print("2fk_pos:", env.rtde_c.getForwardKinematics(ik_pos, [0,0,0,0,0,0]))
    # while True:
    #     print(env.hand.get_hand_force())
    #     time.sleep(0.1)
    # from experiments.block_assembly.spacemouse import pyspacemouse
    # pyspacemouse.open()
    while True:
        state_o = pyspacemouse.read_all()
        state = state_o[0]
        print("xyz0:", state.x, state.y, state.z)
        print("rpy0:", state.roll, state.pitch, state.yaw)
        print("buttion0:", state.buttons)
        state = state_o[1]
        print("xyz1:", state.x, state.y, state.z)
        print("rpy1:", state.roll, state.pitch, state.yaw)
        print("buttion1:", state.buttons)
        print("---------------------------")
        time.sleep(0.1)
    # import pickle as pkl
    # with open("/home/admin01/lyw_2/hil-serl_original/examples/episode_record.pkl", "rb") as f:
    #     data_list = pkl.load(f)
    # mass = 20
    # stiffness = 80
    # damping = 2 * (mass * stiffness) ** 0.5
    # admittance_controller = AdmittanceController(6, mass, damping, stiffness, np.array([0, 0, 1, 0, 0, 0]).astype(bool), 0.1)
    # cnt = 0
    # for data in data_list:
    #     img = data["img"]
    #     initial_r_force = data["initial_r_force"]
    #     initial_tcp_vel = data["initial_tcp_vel"]
    #     action = data["action"]
    #     force = np.linalg.norm([-initial_r_force[1], -initial_r_force[2]])
    #     vel = np.linalg.norm([-initial_tcp_vel[1], -initial_tcp_vel[2]])
    #     delta_p, acc, v = admittance_controller.cal_controller_output(np.ones(6,)*force, action, -0.4*np.ones(6,)-initial_tcp_vel)
    #     print(f"------------{cnt}")
        
    #     # print("delta_p:", delta_p)
    #     # print("acc:", acc)
    #     # print("v:", v)
    #     # print("force:", force)
    #     # print("initial_tcp_vel:", initial_tcp_vel)
    #     # print("action:", action)
    #     # print("initial_r_force:", initial_r_force)
    #     print("initial_tcp_vel:", True if ((initial_tcp_vel[1]<0) and (initial_tcp_vel[2]<0)) else False)
    #     import matplotlib.pyplot as plt
    #     import cv2
    #     # img_rgb = img.reshape(480, 640, 3).astype(np.uint8)
    #     # # plt.imshow(img_rgb)
    #     # # plt.axis('off')
    #     # # plt.show()
    #     # cv2.imshow("img", img_rgb)
    #     # cv2.waitKey(0)
    #     cnt += 1



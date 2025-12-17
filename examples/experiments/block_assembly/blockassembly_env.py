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
import copy
import pyrealsense2 as rs
import cv2
import jax.numpy as jnp
import jax

# params
dof_lower_limits = [-6.2800, -6.2800, -3.1400, -6.2800, -6.2800, -6.2800,  0.0000,  0.0000, 0.0000,  0.0000,  0.0000,  0.2000, -6.2800, -6.2800, -3.1400, -6.2800, -6.2800, -6.2800,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.2000]
dof_upper_limits = [6.2800, 6.2800, 3.1400, 6.2800, 6.2800, 6.2800, 1.7000, 1.7000, 1.7000, 1.7000, 0.5000, 1.3000, 6.2800, 6.2800, 3.1400, 6.2800, 6.2800, 6.2800, 1.7000, 1.7000, 1.7000, 1.7000, 0.5000, 1.3000]

class BlockAssemblyEnv(gym.Env):
    def __init__(self, fake_env, reward_func, evaluate=0, offline_train=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(36,), dtype=np.float32)
        self.observation_space = self.observation_space = gym.spaces.Dict(
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
                        "tcp_vel": gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                        "tcp_force": gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.agent = None
        self.rng = None
        self.queue = np.zeros(50,)
        self.queue_index = 0
        self.evaluate = evaluate
        self.offline_train = offline_train
        
        # important parameters-------------------
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
        self.r_worldold2baseur_pos = self.quat_apply(r_world2basemy_rot, r_basemy2baseur_pos) + r_world2basemy_pos
        self.r_worldold2baseur_rot = self.quat_mul(r_world2basemy_rot, r_basemy2baseur_quat)
        self.r_world2baseur_rot = self.quat_mul(torch.tensor([-0.70710678, 0, 0, 0.70710678]).to(self.device), self.r_worldold2baseur_rot)
        self.r_world2baseur_pos = self.quat_apply(torch.tensor([-0.70710678, 0, 0, 0.70710678]).to(self.device), self.r_worldold2baseur_pos)
        l_basemy2baseur_quat = torch.tensor([-3.60496599e-05, -5.34978704e-06, 7.07114340e-01, -7.07099221e-01]).to(self.device)
        l_basemy2baseur_pos = torch.tensor([-0.02073179, 0.01909007, 0.00689948]).to(self.device)
        l_world2basemy_rot = world2arm_rot_left
        l_world2basemy_pos = world2arm_trans_left
        self.l_world2baseur_pos = self.quat_apply(l_world2basemy_rot, l_basemy2baseur_pos) + l_world2basemy_pos
        self.l_world2baseur_rot = self.quat_mul(l_world2basemy_rot, l_basemy2baseur_quat)

        self.rbaseur2lbaseur_pos = self.quat_apply(self.quat_conjugate(self.r_world2baseur_rot), self.l_world2baseur_pos - self.r_world2baseur_pos)
        self.rbaseur2lbaseur_rot = self.quat_mul(self.quat_conjugate(self.r_world2baseur_rot), self.l_world2baseur_rot)
        
        # 可调参数-------------------------------------
        # 距离完全插入位姿10cm
        self.init_dof_pos0 = np.array([0.987450361251831, -0.934936825429098, 1.25439453125, 0.02481377124786377, 1.4887490272521973, 2.568765163421631])
        self.init_dof_pos1 = np.array([-1.1089, -2.5020, -0.6554, -3.7734, -1.7296,  0.7556])
        self.init_hand_pos = np.array([5.317746399668977e-06, 1.0040700435638428, 0.9746702313423157, 1.01011061668396, 0.10482560843229294, 1.1795626878738403])
        self.init_baseur2tcp_pos0 = np.array([-0.2722, -0.6226,  0.1909])
        self.init_baseur2tcp_quat0 = np.array([ 0.0239, -0.8168,  0.4995,  0.2878])
        self.init_world2tcp_pos0 = self.quat_apply(self.r_world2baseur_rot, torch.from_numpy(self.init_baseur2tcp_pos0).float().to(self.device)) + self.r_world2baseur_pos
        self.init_world2tcp_quat0 = self.quat_mul(self.r_world2baseur_rot, torch.from_numpy(self.init_baseur2tcp_quat0).float().to(self.device))
        # 随机化和范围参数
        self.random_pos_range = 0.03
        self.random_rot_range = 0.175
        # 世界坐标系下的安全范围（y轴向右）
        self.safety_box_size_min = np.array([-self.random_pos_range-0.01, -self.random_pos_range-0.01, -0.1-0.01, -self.random_rot_range-0.05, -self.random_rot_range-0.05, -self.random_rot_range-0.05])
        self.safety_box_size_max = np.array([self.random_pos_range+0.01, self.random_pos_range+0.01, 0.01, self.random_rot_range+0.05, self.random_rot_range+0.05, self.random_rot_range+0.05])
        # 步长# set same as hilserl peg insertion
        self.action_pos_range_value = 0.01
        self.action_rot_range_value = 0.05
        # 控制频率
        self.hz = 10
        # episode_len
        self.max_episode_steps = 200
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
            # listener = keyboard.Listener(on_press=self.on_press)
            # listener.start()
            # reward
            self.reward_func = reward_func
            self.success_flag = 0
            self.stop_flag = False

    def on_press(self, key):
        try:
            if isinstance(key, keyboard.Key):
                if key == keyboard.Key.esc:
                    self.next_ep_long_reset = True
                if key == keyboard.Key.space:
                    self.success_flag = 1
        except AttributeError:
            pass

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

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        success, reset = buttons[0], buttons[-1]
        if success:
            self.success_flag = 1
            self.next_ep_long_reset = True
        if reset:
            self.next_ep_long_reset = True
        intervened = False
        
        if np.linalg.norm(expert_a) > 0.1:#0.001:
            intervened = True

        if intervened:
            return expert_a, True

        return action, False
    
    def safety_box(self, target_world2ee_pos0, target_world2ee_quat0):
        # target_world2ee_pos0 = self.quat_apply(self.r_world2baseur_rot, target_baseur2ee_pos0) + self.r_world2baseur_pos
        delta_world2ee_pos0 = (target_world2ee_pos0 - self.init_world2tcp_pos0).cpu().numpy()
        clamp_delta_world2ee_pos0 = np.clip(delta_world2ee_pos0, self.safety_box_size_min[:3], self.safety_box_size_max[:3])
        clamp_target_world2ee_pos0 = self.init_world2tcp_pos0 + torch.from_numpy(clamp_delta_world2ee_pos0).float().to(self.device)
        clamp_target_baseur2ee_pos0 = self.quat_apply(self.quat_conjugate(self.r_world2baseur_rot), clamp_target_world2ee_pos0 - self.r_world2baseur_pos)
        
        # target_world2ee_quat0 = self.quat_mul(self.r_world2baseur_rot, target_baseur2ee_quat0)
        delta_world2ee_euler0 = R.from_quat((self.quat_mul(target_world2ee_quat0, self.quat_conjugate(self.init_world2tcp_quat0))).cpu().numpy()).as_euler("xyz")
        clamp_delta_world2ee_euler0 = np.clip(delta_world2ee_euler0, self.safety_box_size_min[3:], self.safety_box_size_max[3:])
        clamp_delta_world2ee_quat0 = torch.from_numpy(R.from_euler('xyz', clamp_delta_world2ee_euler0).as_quat()).float().to(self.device)
        clamp_delta_world2ee_quat0_dt = self.quat_mul(self.quat_mul(self.quat_conjugate(self.init_world2tcp_quat0), clamp_delta_world2ee_quat0), self.init_world2tcp_quat0)
        clamp_target_world2ee_quat0 = self.quat_mul(self.init_world2tcp_quat0, clamp_delta_world2ee_quat0_dt)
        clamp_target_baseur2ee_quat0 = self.quat_mul(self.quat_conjugate(self.r_world2baseur_rot), clamp_target_world2ee_quat0)
        clamp_target_baseur2ee_rotvec0 = R.from_quat(clamp_target_baseur2ee_quat0.cpu().numpy()).as_rotvec()
        return clamp_target_baseur2ee_pos0, clamp_target_baseur2ee_rotvec0

    def step(self, action):
        
        print("--------------------------step:", self.steps)
        initial_obs, initial_tcp_pose = self.compute_obs()
        
        info = {}
        bad_data = False
        action, is_intervened = self.action(action)
        if is_intervened:
            info["intervened"] = action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = torch.from_numpy(action).float().to(self.device)
        initial_baseur2ee_pos0 = torch.from_numpy(initial_tcp_pose[:3].copy()).to(self.device)
        initial_baseur2ee_quat0 = torch.from_numpy(initial_tcp_pose[3:].copy()).to(self.device)
        initial_world2ee_pos0 = self.quat_apply(self.r_world2baseur_rot, initial_baseur2ee_pos0) + self.r_world2baseur_pos
        initial_world2ee_quat0 = self.quat_mul(self.r_world2baseur_rot, initial_baseur2ee_quat0)
        target_world2ee_pos0 = initial_world2ee_pos0 + action[:3]*self.action_pos_range_value
        world2ee_euler_err0 = (action[3:]*self.action_rot_range_value).cpu().numpy()#self.scale(action[3:6], self.action_rot_range_lower, self.action_rot_range_upper).cpu().numpy()
        world2ee_quat_err0 = torch.from_numpy(R.from_euler('xyz', world2ee_euler_err0).as_quat()).float().to(self.device)
        world2ee_quat_dt0 = self.quat_mul(self.quat_mul(self.quat_conjugate(initial_world2ee_quat0), world2ee_quat_err0), initial_world2ee_quat0)
        target_world2ee_quat0 = self.quat_mul(initial_world2ee_quat0, world2ee_quat_dt0)
        
        clamp_target_baseur2ee_pos0, clamp_target_baseur2ee_rotvec0 = self.safety_box(target_world2ee_pos0, target_world2ee_quat0)
        arm_pos = self.cal_ik_real_robot(clamp_target_baseur2ee_pos0.cpu().numpy(), clamp_target_baseur2ee_rotvec0, self.rtde_r.getActualQ())#self.rtde_c.getInverseKinematics(np.concatenate([clamp_target_baseur2ee_pos0.cpu().numpy(), baseur2ee_rot0]))
        if arm_pos is None:
            self.rtde_c.stopJ(1, False)
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
                    self.rtde_c.servoJ(arm_pos, 0.8, 0.1, (1.0 / self.hz), 0.2, 600)
            except RuntimeError as e:
                print("step_error:", e)
                input("把手臂移动一下，脱离安全性停止。。。")
                self.stop_flag = True
                self.next_ep_long_reset = True
                self.reconnect_to_robot()
            time.sleep(1.0 / self.hz)
        next_obs, next_tcp_pose = self.compute_obs()
        
        self.steps += 1
        self.total_steps += 1
        reward = copy.deepcopy(self.success_flag)#self.reward_func(next_obs["images"])
        self.success_flag = 0
        print("***************reward:", reward)
        done = (self.steps == self.max_episode_steps)|(reward == 1)
        
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
    
    def process_image(self, image) -> np.ndarray:
        cropped_rgb = image[:, 80:560, :]
        resized = cv2.resize(cropped_rgb, self.img_size.shape[:2][::-1])
        processed_img = resized[..., ::-1]
        return processed_img
        
    def get_img(self):
        shelf_img = self.capture_frame(self.pipeline_shelf, self.aligner_shelf)
        ground_img = self.capture_frame(self.pipeline_ground, self.aligner_ground)

        # combined = np.hstack([cv2.cvtColor(shelf_img.reshape(480, 640, 3).astype(np.uint8), cv2.COLOR_BGR2RGB), cv2.cvtColor(ground_img.reshape(480, 640, 3).astype(np.uint8), cv2.COLOR_BGR2RGB)])
        combined = np.hstack([shelf_img, ground_img])
        
        cv2.imshow("Two Cameras", combined)
        cv2.waitKey(1)

        images = {"shelf": self.process_image(shelf_img), "ground": self.process_image(ground_img)}
        return images

    def get_state(self):
        baseur2ee_pose0 = np.array(self.cal_fk_real_robot("right"))
        baseur2ee_pos0 = baseur2ee_pose0[:3].astype(np.float32)
        baseur2ee_rot0 = R.from_rotvec(baseur2ee_pose0[3:6]).as_quat().astype(np.float32)
        tcp_pose = jnp.array(np.concatenate([baseur2ee_pos0, baseur2ee_rot0]))

        r_speed = jnp.array(self.rtde_r.getActualTCPSpeed())

        r_force = jnp.array(self.rtde_r.getActualTCPForce())

        state_observation = {
            "tcp_pose": tcp_pose,
            "tcp_vel": r_speed,
            "tcp_force": r_force#force&torque
        }
        
        return state_observation

    def compute_obs(self):
        images = self.get_img()
        state_observation = self.get_state()
        return copy.deepcopy(dict(images=images, state=state_observation)), jax.device_get(state_observation["tcp_pose"])

    def reset_within_certain_range(self):
        while True:
            random_angle = np.random.uniform(low=-self.random_rot_range, high=self.random_rot_range, size=3)
            tcp02tcp1_quat = torch.from_numpy(R.from_euler("xyz", random_angle).as_quat()).float().to(self.device)
            tcp02tcp1_quat_dt = self.quat_mul(self.quat_mul(self.quat_conjugate(self.init_world2tcp_quat0), tcp02tcp1_quat), self.init_world2tcp_quat0)
            
            tcp02tcp1_pos = ((torch.rand(3) - 0.5) * 2 * self.random_pos_range).to(self.device) 
            tcp02tcp1_pos[2] = 0 # 在world坐标系下的z方向上不动，在xy平面上动

            target_world2tcp_pos = (self.init_world2tcp_pos0 + tcp02tcp1_pos)
            target_world2tcp_quat = (self.quat_mul(self.init_world2tcp_quat0, tcp02tcp1_quat_dt))
            
            target_baseur2tcp_pos = self.quat_apply(self.quat_conjugate(self.r_world2baseur_rot), target_world2tcp_pos - self.r_world2baseur_pos).cpu().numpy()
            target_baseur2tcp_quat = self.quat_mul(self.quat_conjugate(self.r_world2baseur_rot), target_world2tcp_quat).cpu().numpy()
            target_baseur2tcp_rotvec = R.from_quat(target_baseur2tcp_quat).as_rotvec()

            
            target_dof_pos0 = self.cal_ik_real_robot(target_baseur2tcp_pos, target_baseur2tcp_rotvec, self.rtde_r.getActualQ())
            if target_dof_pos0 is None:
                continue
            else:
                print("current_dof_pos0:", self.rtde_r.getActualQ())
                print("随机化初始化target_dof_pos0:", target_dof_pos0)
                # input("press enter to move robot to random pose...")
                self.rtde_c.moveJ(target_dof_pos0, 0.1, 1)
                break

    def reset(self, **kwargs):
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
        self.rtde_c.moveJ(self.init_dof_pos0, 0.1, 1)
        self.another_rtde_c.moveJ(self.init_dof_pos1, 0.1, 1)
        self.reset_within_certain_range()
        
        obs, _ = self.compute_obs()
        if self.evaluate != 1:
            wandb.log({'mujoco_reward/reward': self.episode_reward}, step=self.total_steps)
        self.episode_reward = 0
        self.steps = 0
        print("!!!!!!!!!!!!!!!!!!!!!episode reset!!!!!!!!!!!!!!!!!!!!!")
        self.success_flag = 0
        self.next_ep_long_reset = False
        self.stop_flag = False
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
        new_pos = self.init_hand_pos.copy()
        new_pos[3] += 0.03
        new_pos[1] += 0.13
        self.another_hand.set_hand_angle(new_pos)
    def control_right_hand4(self):
        self.hand.set_hand_angle(self.init_hand_pos)
    def control_left_hand7(self):
        self.another_rtde_c.moveJ(self.init_dof_pos1, 1, 1)#another_rtde_c.servoJ(self.rh_data[799][18:24], 0.2, 0.1, 0.05, 0.2, 600)
        new_pos = self.init_hand_pos.copy()
        new_pos -= 0.15
        new_pos[-1] += 0.15
        self.another_hand.set_hand_angle(new_pos)
    def control_right_hand6(self):
        self.rtde_c.moveJ(self.init_dof_pos0, 1, 1)#rtde_c.servoJ(self.rh_data[799][0:6], 0.2, 0.1, 0.05, 0.2, 600)
        self.hand.set_hand_angle(self.init_hand_pos-0.15)

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
    
    from examples.experiments.block_assembly.spacemouse import pyspacemouse
    pyspacemouse.open()
    while True:
        state = pyspacemouse.read_all()
        state = state[0]
        print("xyz:", state.x, state.y, state.z)
        print("rpy:", state.roll, state.pitch, state.yaw)
        print("buttion:", state.buttons)
        print("---------------------------")
        time.sleep(0.1)

import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.launcher import make_sac_pixel_agent
from flax.training import checkpoints
import jax
devices = jax.local_devices()
sharding = jax.sharding.PositionalSharding(devices)
FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "blockassembly", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 1, "Number of successful transistions to collect.")
flags.DEFINE_string("checkpoint_path", "/home/admin01/lyw_2/hil-serl_original/checkpoints/20251218_160448", "Path to save checkpoints.")#"/home/admin01/lyw_2/hil-serl_original/checkpoints/"+time.strftime("%Y%m%d_%H%M%S", time.localtime())
flags.DEFINE_integer("eval_checkpoint_step", 0 ,"Step to evaluate the checkpoint.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("record_demo", 1, "Record_demo.")
flags.DEFINE_boolean("control_double_arm", False, "Control double arm.")
"""
人工控制插入，成功时按spacemouse的按键
"""
success_key = False
def on_press(key):
    global success_key
    try:
        if str(key) == 'Key.space':
            success_key = True
    except AttributeError:
        pass

def main(_):
    global success_key
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name](FLAGS.control_double_arm)
    env = config.get_environment(fake_env=False, evaluate=True, classifer=(FLAGS.record_demo==0))

    if FLAGS.eval_checkpoint_step:
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
        ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
                step=FLAGS.eval_checkpoint_step,
            )
        agent = agent.replace(state=ckpt)


    obs, _ = env.reset()
    successes = []
    failures = []
    demos = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)
    sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
    success_cnt = 0
    while len(successes) < success_needed:
        if (FLAGS.eval_checkpoint_step) and (success_cnt == 0) and (FLAGS.record_demo == 0):
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key
                )
            actions = np.asarray(jax.device_get(actions))
        else:
            actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        print("***************reward:", rew)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        obs = next_obs
        if FLAGS.record_demo:
            demos.append(transition)
            if rew == 1.0:
                successes.append(transition)
                pbar.update(1)
                print(f"add {len(demos)} demo transition!")
            
        else:
            if (rew == 1.0) or (success_cnt > 0):
                success_cnt += 1
                
                transition["dones"] = done
                transition["masks"] = 1.0 - done
                transition["rewards"] = 1.0
                successes.append(transition)
                print(f"add {len(successes)} success transition!")
                pbar.update(1)
                success_key = False
                if success_cnt == 10:
                    success_cnt = 0
                    done = True # add by lyw
            else:
                failures.append(transition)

        if done or truncated:
            obs, _ = env.reset()

    if FLAGS.record_demo:
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"/home/admin01/lyw_2/hil-serl_original/data/demos/demo_{curr_time}.pkl", "wb") as f:
            pkl.dump(demos, f)
    else:
        if not os.path.exists("./classifier_data"):
            os.makedirs("./classifier_data")
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if len(successes) > 0:
            file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
            with open(file_name, "wb") as f:
                pkl.dump(successes, f)
                print(f"saved {success_needed} successful transitions to {file_name}")
        if len(failures) > 0:
            file_name = f"./classifier_data/{FLAGS.exp_name}_{len(failures)}_failure_images_{uuid}.pkl"
            with open(file_name, "wb") as f:
                pkl.dump(failures, f)
                print(f"saved {len(failures)} failure transitions to {file_name}")
    
        
if __name__ == "__main__":
    # app.run(main)
    
    # with open("/home/admin01/lyw_2/hil-serl_original/data/demos/demo_2026-01-05_22-33-09.pkl", "rb") as f:
    #     data_list = pkl.load(f)
    #     print(data_list[-1]["observations"]["state"][0][6:])
        # print(data_list[0]["observations"]["state"].shape)
        # for data in data_list:
        #     print(data["observations"]["state"][0][6:])
        #     print("----------------------")
        #     import cv2
        #     import matplotlib.pyplot as plt
        #     print(data["observations"]["shelf"][0].shape)
        #     img_rgb = cv2.cvtColor(data["observations"]["shelf"][0].astype(np.uint8), cv2.COLOR_BGR2RGB)
        #     # plt.imshow(img_rgb)
        #     # plt.axis('off')
        #     # plt.show()
        #     cv2.imshow("img", img_rgb)
        #     cv2.waitKey(0)
    # #     final_list = []
    # #     for data in data_list:
    # #         # 将原始数据结构转换为扁平结构
    # #         img_dict = data["observations"]["images"]
    # #         # 把img_dict里面叫"shelf”的键改名字叫"shelf","ground"的键叫“ground_img”
    # #         # 先做当前 observations["images"] 里的 key 重命名
    # #         # if "shelf" in img_dict:
    # #         #     img_dict["shelf"] = img_dict.pop("shelf")
    # #         # if "ground" in img_dict:
    # #         #     img_dict["ground"] = img_dict.pop("ground")
    # #         state = data["observations"]["state"]
    # #         # 合并字典
    # #         flat_obs = {**img_dict, "state": np.concatenate([state['tcp_pose'], state['tcp_vel'], state['tcp_force']])}
    # #         data["observations"] = flat_obs

    # #         img_dict = data["next_observations"]["images"]
    # #         # if "shelf" in img_dict:
    # #         #     img_dict["shelf"] = img_dict.pop("shelf")
    # #         # if "ground" in img_dict:
    # #         #     img_dict["ground"] = img_dict.pop("ground")
    # #         state = data["next_observations"]["state"]
    # #         # 合并字典
    # #         flat_obs = {**img_dict, "state": np.concatenate([state['tcp_pose'], state['tcp_vel'], state['tcp_force']])}
    # #         data["next_observations"] = flat_obs
    # #         final_list.append(data)
    # # with open("/home/admin01/lyw_2/hil-serl_original/data/demos/demo3.pkl", "wb") as f:
    # #     pkl.dump(final_list, f)
    #     # print(data_list[5]["observations"]["state"])
    #     cnt = 0 
    #     for data in data_list:
    #         cnt += 1
    #         if data["dones"]:
    #             print(cnt)
    #             cnt = 0
    # success = []
    # failure = []
    # for cnt in range(1000, 7000, 1000):
    #     with open(f"/home/admin01/lyw_2/hil-serl_original/checkpoints/20251231_1926/buffer/transitions_{cnt}.pkl", "rb") as f:
    #         data_list = pkl.load(f)
    #         for data in data_list:
    #             if data["rewards"] == 1.0:
    #                 success.append(data)
    #             else:
    #                 failure.append(data)
    
    # curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # with open(f"/home/admin01/lyw_2/hil-serl_original/examples/classifier_data/blockassembly_{len(success)}_success_images_{curr_time}.pkl", "wb") as f:
    #     pkl.dump(success, f)
    # with open(f"/home/admin01/lyw_2/hil-serl_original/examples/classifier_data/blockassembly_{len(failure)}_failure_images_{curr_time}.pkl", "wb") as f:
    #     pkl.dump(failure, f)
    # for cnt in range(1000, 10000, 1000):
    #     with open(f"/home/admin01/lyw_2/hil-serl_original/checkpoints/20251224_1416/demo_buffer/transitions_{cnt}.pkl", "rb") as f:
    #         data_list = pkl.load(f)
    #         print(len(data_list))
    with open("/home/admin01/lyw_2/hil-serl_original/data/demos/demo_2026-01-06_22-31-58_3dof.pkl", "rb") as f:
        new_data_list = pkl.load(f)
        for i in range(len(new_data_list)-1):
            print(np.linalg.norm(new_data_list[i]["observations"]["state"][0][6:9] - new_data_list[i+1]["observations"]["state"][0][6:9]))
    # with open("/home/admin01/lyw_2/hil-serl_original/data/demos/demo_2026-01-07_14-04-33.pkl", "rb") as f:
    #     data_list = pkl.load(f)
    #     cnt = 0
    #     for i in range(len(data_list)):
    #         cnt += 1
    #         if data_list[-2-i]["rewards"] == 1:
    #             break
    #     data_list[-cnt:] = new_data_list
    #     with open("/home/admin01/lyw_2/hil-serl_original/data/demos/demo_2026-01-07_14-04-33_new.pkl", "wb") as f:
    #         pkl.dump(data_list, f)
    #     # for data in data_list:
    #     #     if data["observations"]["state"].shape[1] != 19:
    #     #         print(data["observations"]["state"].shape)
    #     final_list = []
    #     for data in data_list:
    #         data["actions"] = data["actions"][:3]
    #         final_list.append(data)
    #     with open("/home/admin01/lyw_2/hil-serl_original/data/demos/demo_2026-01-06_22-31-58_3dof.pkl", "wb") as f:
    #         pkl.dump(final_list, f)
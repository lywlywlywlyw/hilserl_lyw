import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "blockassembly", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful transistions to collect.")


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
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, evaluate=True)

    obs, _ = env.reset()
    successes = []
    failures = []
    demos = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    
    while len(successes) < success_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
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
        if success_key:
            done = True # add by lyw
            transition["dones"] = done
            transition["masks"] = 1.0 - done
            transition["rewards"] = 1.0
            successes.append(transition)
            pbar.update(1)
            success_key = False
            
        else:
            failures.append(transition)
        demos.append(transition)

        if done or truncated:
            obs, _ = env.reset()

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")
    with open("/home/admin01/lyw_2/hil-serl_original/data/demos/demo.pkl", "wb") as f:
        pkl.dump(demos, f)
        
if __name__ == "__main__":
    # app.run(main)
    with open("/home/admin01/lyw_2/hil-serl_original/data/demos/demo.pkl", "rb") as f:
        data_list = pkl.load(f)
        # import cv2
        # import matplotlib.pyplot as plt
        # img_rgb = cv2.cvtColor(data[0]["observations"]["images"]["shelf"].astype(np.uint8), cv2.COLOR_BGR2RGB)
        # plt.imshow(img_rgb)
        # plt.axis('off')
        # plt.show()
    #     final_list = []
    #     for data in data_list:
    #         # 将原始数据结构转换为扁平结构
    #         img_dict = data["observations"]["images"]
    #         # 把img_dict里面叫"shelf”的键改名字叫"shelf","ground"的键叫“ground_img”
    #         # 先做当前 observations["images"] 里的 key 重命名
    #         # if "shelf" in img_dict:
    #         #     img_dict["shelf"] = img_dict.pop("shelf")
    #         # if "ground" in img_dict:
    #         #     img_dict["ground"] = img_dict.pop("ground")
    #         state = data["observations"]["state"]
    #         # 合并字典
    #         flat_obs = {**img_dict, "state": np.concatenate([state['tcp_pose'], state['tcp_vel'], state['tcp_force']])}
    #         data["observations"] = flat_obs

    #         img_dict = data["next_observations"]["images"]
    #         # if "shelf" in img_dict:
    #         #     img_dict["shelf"] = img_dict.pop("shelf")
    #         # if "ground" in img_dict:
    #         #     img_dict["ground"] = img_dict.pop("ground")
    #         state = data["next_observations"]["state"]
    #         # 合并字典
    #         flat_obs = {**img_dict, "state": np.concatenate([state['tcp_pose'], state['tcp_vel'], state['tcp_force']])}
    #         data["next_observations"] = flat_obs
    #         final_list.append(data)
    # with open("/home/admin01/lyw_2/hil-serl_original/data/demos/demo3.pkl", "wb") as f:
    #     pkl.dump(final_list, f)
        print(data_list[5]["observations"]["state"])
import sys
import numpy as np
import torch
import os
from tqdm import tqdm, trange
sys.path.append("..")

import moviepy.editor as mpy
import env
import gym

import pickle
from gym import spaces
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from config import argparser
from config.motion_planner import add_arguments as mp_add_arguments
from collections import OrderedDict

from termcolor import colored

from bc_visual_args import args
from behavioral_cloning_visual import BC_Visual_Policy, BC_Image_Only, BC_Robot_Only, BC_Visual_Policy_Stochastic

import matplotlib
matplotlib.use('Agg')

# set global seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def retrieve_np_state(raw_state):
    for idx, values in enumerate(raw_state):
        if(idx==0):
            ot = np.array(values)
        else:
            ot = np.concatenate((ot, np.array(values)), axis=0)

    return ot

def get_img_robot_state(obs, env):
    obs_img = torch.from_numpy(obs['image'])
    if env == 'PusherObstacle-v0':
        state_info = list(obs.values())
        state_info = state_info[0:2]
        obs_robot = retrieve_np_state(state_info)
    elif env == 'SawyerPushObstacle-v0' or \
        env == 'SawyerAssemblyObstacle-v0' or \
        env == 'SawyerLiftObstacle-v0':
        obs_robot = np.concatenate((obs['joint_pos'], obs['joint_vel'], obs['gripper_qpos'], obs['gripper_qvel'], obs['eef_pos'], obs['eef_quat']))
    else:
        print('ERROR: Incorrect env name')
    obs_robot = torch.from_numpy(obs_robot).float()
    obs_robot = obs_robot[None, :]
    return obs_img, obs_robot

def visualize_feature_maps(obs_img, policy):
    from matplotlib import pyplot
    out = policy.visualize_third_conv_layer(obs_img)
    square = 16
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(out[0, ix-1, :, :].cpu().detach().numpy(), cmap='gray')
            ix += 1
    # show the figure
    pyplot.savefig('../out/feature_maps_conv3.png')
    breakpoint()
    return None

def run(config):

    os.environ["DISPLAY"] = ":1"

    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.cuda_num)
        assert torch.cuda.is_available()
        device = 'cuda:{}'.format(args.cuda_num)
    else:
        device = torch.device("cpu")

    if args.model == 'BC_Visual_Policy':
        policy = BC_Visual_Policy(robot_state=args.robot_state_size, num_classes=args.action_size, img_size=args.env_image_size)
    elif args.model == 'BC_Image_Only':
        policy = BC_Image_Only(num_classes=args.action_size, img_size=args.env_image_size)
    elif args.model == 'BC_Robot_Only':
        policy = BC_Robot_Only(robot_state=args.robot_state_size, num_classes=args.action_size)
    elif args.model == 'BC_Visual_Policy_Stochastic':
        policy = BC_Visual_Policy_Stochastic(robot_state=args.robot_state_size, num_classes=args.action_size, img_size=args.env_image_size)
    else:
        print(colored('ERROR: Do not support this model {}'.format(args.model)), 'red')
        exit(1)
    print(colored('test model {}'.format(args.model), 'blue'))
    print(policy)

    checkpoint = torch.load(os.path.join(args.model_save_dir, args.checkpoint), map_location='cpu')
    print('loading checkpoint {}'.format(os.path.join(args.model_save_dir, args.checkpoint)))
    
    policy.load_state_dict(checkpoint['state_dict'])
    policy.eval()

    num_success = 0
    num_ep = args.num_eval_ep
    ep_len_success_total = 0
    ep_success_total = 0
    total_episodes = 0
    total_discounted_rewards = 0
    eefs = []

    running_seeds = []
    if args.three_hundred_eval_five_seeds:
        running_seeds = [1234, 200, 500, 2320, 1800]
    else:
        running_seeds = [config.seed]

    for curr_seed in running_seeds:
        _env_eval = gym.make(config.env, **config.__dict__)
        _env_eval.set_seed(curr_seed)
        print(colored('Seed {}'.format(curr_seed), 'blue'))

        for ep in trange(num_ep):
            states = []
            _record_frames = []
            rollout_states = []
            
            obs = _env_eval.reset()

            obs_img, obs_robot = get_img_robot_state(obs, config.env)

            # visualize_feature_maps(obs_img, policy) # DEBUG

            states.append(obs_robot)

            done = False
            ep_len = 0
            ep_rew = 0
            ep_discounted_rew = 0
            
            _store_frame(_env_eval, _record_frames, info={})
            
            # rollout_states.append({"ob": [obs_robot], "ac": []})

            while ep_len < args.eval_bc_max_step and not done:

                if args.model == 'BC_Visual_Policy':
                    action = policy(obs_img, obs_robot)
                elif args.model == 'BC_Image_Only':
                    action = policy(obs_img)
                elif args.model == 'BC_Robot_Only':
                    action = policy(obs_robot)
                elif args.model == 'BC_Visual_Policy_Stochastic':
                    action = policy(obs_img, obs_robot)

                if len(action.shape) == 2:
                    action = action[0]
                obs, reward, done, info = _env_eval.step(action.detach().numpy(), is_bc_policy=True)
                rollout_states.append({"ob": [obs], "ac": [action.detach().numpy()]})
                
                obs_img, obs_robot = get_img_robot_state(obs, config.env)
                
                ep_len += 1
                ep_rew += reward
                # discounted reward based on formula: $\sum_{t=0}^{T-1} \gamma^t R(s_t, a_t)$
                discounted_reward = pow(args.discount_factor, (ep_len-1)) * reward
                ep_discounted_rew += discounted_reward
                _store_frame(_env_eval, _record_frames, info)

                if(ep_len % 100 == 0):
                    print (colored("Current Episode Step: {}, Reward: {}, Discounted Reward: {}".format(ep_len, reward, discounted_reward), "green"))

            if _env_eval._success:
                ep_success = "s"
                ep_len_success_total += ep_len
                ep_success_total += 1
            else:
                ep_success = "f"
            
            fname = "{}_step_{:011d}_{}_r_{}_{}.mp4".format(
                config.env,
                0,
                total_episodes,
                ep_rew,
                ep_success,
            )

            total_episodes += 1
            total_discounted_rewards += ep_discounted_rew

            if(ep_rew>0):
                num_success += 1
            _save_video(fname, _record_frames, config)

            # saving eefs
            cur_eefs = []
            for obj in rollout_states:
                cur_eefs.append(obj['ob'][0]['eef_pos'])
            eefs.append(np.array(cur_eefs))
    
            print("Episode Length: {}, Episode Reward:{}, Episode Discounted Reward:{}.".format(ep_len, ep_rew, ep_discounted_rew), done)
            print(colored("Number of positive reward episodes: " + str(num_success), "red"))

            with open(args.saved_rollouts+"/{}.p".format("bc"), "wb") as f:
                pickle.dump(rollout_states, f)
        print('Finished running seed ', curr_seed)
    
    # saving end-effector positions in numpy
    # with open('bc_eef_positions.npy', 'wb') as f:
    #     np.save(f, eefs)

    print(colored("Average success rate: " + str(ep_success_total/total_episodes*100) + "%", "yellow"))
    print(colored("Average discounted rewards: " + str(total_discounted_rewards/total_episodes), "yellow"))
    print(colored("Average episode length: " + str(ep_len_success_total/ep_success_total), "yellow"))

def _store_frame(env, _record_frames, info={}):
        color = (200, 200, 200)
        geom_colors = {}

        frame = env.render("rgb_array") * 255.0

        _record_frames.append(frame)

def _save_video(fname, frames, config, fps=8.0):
        if(not os.path.exists(args.bc_video_dir)):
            os.mkdir(args.bc_video_dir)
        record_dir = args.bc_video_dir
        path = os.path.join(record_dir, fname)

        def f(t):
            frame_length = len(frames)
            new_fps = 1.0 / (1.0 / fps + 1.0 / frame_length)
            idx = min(int(t * new_fps), frame_length - 1)
            return frames[idx]

        video = mpy.VideoClip(f, duration=len(frames) / fps + 2)

        video.write_videofile(path, fps, verbose=False, logger=None)
        print (colored("[*] Video saved: {}".format(path), "green"))

def overwrite_env_args(env_args):
    env_args.env = args.env
    env_args.env_image_size = args.env_image_size
    env_args.seed = args.env_seed
    env_args.screen_width = args.screen_width
    env_args.screen_height = args.screen_height
    env_args.obs_space = 'all'

if __name__ == "__main__":
    parser = argparser()
    args_mopa, unparsed = parser.parse_known_args()

    if "Pusher" in args.env:
        from config.pusher import add_arguments
    elif "Sawyer" in args.env:
        from config.sawyer import add_arguments
    else:
        raise ValueError("args.env (%s) is not supported" % args_mopa.env)

    add_arguments(parser)
    mp_add_arguments(parser)
    args_mopa, unparsed = parser.parse_known_args()

    # overwrite environment arguments from bc_visual_args.py
    overwrite_env_args(args_mopa)

    if args_mopa.debug:
        args_mopa.rollout_length = 150
        args_mopa.start_steps = 100

    if args_mopa.env == 'PusherObstacle-v0':
        args.action_size = 4
        args.robot_state_size = 14
    elif args_mopa.env == 'SawyerPushObstacle-v0' or \
        args_mopa.env == 'SawyerAssemblyObstacle-v0':
        args.action_size = 7
        args.robot_state_size = 25
    elif args_mopa.env == 'SawyerLiftObstacle-v0':
        args.action_size = 8
        args.robot_state_size = 25
    else:
        print('ERROR: Incorrect env name')
        exit(1)
  
    if len(unparsed):
        logger.error("Unparsed argument is detected:\n%s", unparsed)
    else:
        run(args_mopa)

import copy
import glob
import os
import time
import random
import math

import pickle
import wandb
import shutil
from collections import deque
import gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch.multiprocessing
import matplotlib
import matplotlib.pyplot as plt
import torchvision

import moviepy.editor as mpy
from termcolor import colored

from bc_visual_args import args
import tqdm

# require for evaluation
import sys 
sys.path.append('../')
import env
from tqdm import trange
from config import argparser
from config.motion_planner import add_arguments as mp_add_arguments

from rl.policies.distributions import (
    FixedCategorical,
    FixedNormal,
    MixedDistribution,
    FixedGumbelSoftmax,
)
from collections import OrderedDict

from rl.policies import get_actor_critic_by_name
from util.gym import observation_size, action_size

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.set_num_threads(args.num_threads)

torch.multiprocessing.set_sharing_strategy('file_system') # for RuntimeError: Too many open files. Communication with the workers is no longer possible.

class BC_Visual_Policy(nn.Module): 
    def __init__(self, robot_state=0, num_classes=256, img_size=128):
        super(BC_Visual_Policy, self).__init__()

        first_linear_layer_size = int(256 * math.floor(img_size / 8) * math.floor(img_size / 8))
 
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(first_linear_layer_size + robot_state, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # Defining the forward pass    
    def forward(self, x, robot_state):
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        # concatenate img features with robot_state's information
        x = torch.cat([x, robot_state], dim=1)
        x = self.linear_layers(x)
        return x
    
    def visualize_third_conv_layer(self, x):
        x = self.cnn_layers(x)
        return x

class BC_Visual_Policy_Stochastic(nn.Module): 
    def __init__(self, robot_state=0, num_classes=256, img_size=128):
        super(BC_Visual_Policy_Stochastic, self).__init__()

        first_linear_layer_size = int(256 * math.floor(img_size / 8) * math.floor(img_size / 8))
 
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(first_linear_layer_size + robot_state, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
        )

        self._activation_fn = getattr(F, 'relu')
    
        self.fc_means = nn.Sequential(
            nn.Linear(256, num_classes),
        )

        self.fc_log_stds = nn.Sequential(
            nn.Linear(256, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # Defining the forward pass    
    def forward(self, x, robot_state):
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        # concatenate img features with robot_state's information
        x = torch.cat([x, robot_state], dim=1)
        x = self.linear_layers(x)

        x = self._activation_fn(x)

        means = self.fc_means(x)
        log_std = self.fc_log_stds(x)
        log_std = torch.clamp(log_std, -10, 2)
        stds = torch.exp(log_std.double())
        means = OrderedDict([("default", means)])
        stds = OrderedDict([("default", stds)])
        
        z = FixedNormal(means['default'], stds['default']).rsample()

        action = torch.tanh(z)
        
        return action


class BC_Image_Only(nn.Module): 
    def __init__(self, num_classes=256, img_size=128):
        super(BC_Image_Only, self).__init__()

        first_linear_layer_size = int(256 * math.floor(img_size / 8) * math.floor(img_size / 8))

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(first_linear_layer_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x

class BC_Robot_Only(nn.Module): 
    def __init__(self, robot_state=0, num_classes=256):
        super(BC_Robot_Only, self).__init__()

        self.linear_layers = nn.Sequential(
            nn.Linear(robot_state, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # Defining the forward pass    
    def forward(self, x):
        x = self.linear_layers(x)
        return x

class StateImageActionDataset(Dataset):
    def __init__(self, config, pickle_file, transform=None):
        self.config = config
        rollout_file = open(pickle_file, 'rb')
        self.data = pickle.load(rollout_file)
        rollout_file.close()
        self.transform = transform

    def __len__(self):
        return len(self.data['obs'])

    def random_crop_and_pad(self, img, crop=84):
        """
            source: https://github.com/MishaLaskin/rad/blob/master/data_augs.py
            args:
            img: np.array shape (C,H,W)
            crop: crop size (e.g. 84)
            returns np.array
        """
        data_aug_prob = random.uniform(0, 1)
        if data_aug_prob < 0.5:
            c, h, w = img.shape
            crop_max = h - crop + 1
            w1 = np.random.randint(0, crop_max)
            h1 = np.random.randint(0, crop_max)
            cropped = np.zeros((c, h, w), dtype=img.dtype)
            cropped[:, h1:h1 + crop, w1:w1 + crop] = img[:, h1:h1 + crop, w1:w1 + crop]            
            return cropped
        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ob, ac, img_filepath, full_state_ob = self.data["obs"][idx], self.data["acs"][idx], self.data['imgs'][idx], self.data["full_state_obs"][idx]
        img = np.load('../' + img_filepath)

        # customized data augmentations
        if self.config.img_aug :
            if self.config.random_crop:
                img = self.random_crop_and_pad(img, self.config.random_crop_size)

        ob = torch.from_numpy(ob)
        ac = torch.from_numpy(ac)
        img = torch.from_numpy(img)
        full_state_ob = torch.from_numpy(full_state_ob)

        # torchvision style data augmentations
        if self.transform:
            img = self.transform(img)

        out = {'ob': ob, 'ac': ac, 'img': img, 'full_state_ob': full_state_ob}
        return out

class Evaluation:
    def __init__(self, bc_visual_args, args_mopa):
        os.environ["DISPLAY"] = ":1"
        self.eval_seeds = [52, 93, 156, 377, 1000, 1234]
        self.args_mopa = args_mopa
        self.args_mopa.gpu= int(bc_visual_args.cuda_num)
        self.bc_visual_args = bc_visual_args
        self.model = bc_visual_args.model
        self.robot_state_size = bc_visual_args.robot_state_size
        self.action_size = bc_visual_args.action_size
        self._env_eval = gym.make(self.args_mopa.env, **self.args_mopa.__dict__)

    def get_img_robot_state(self, obs):
        obs_img = torch.from_numpy(obs['image'])
        if self.args_mopa.env == 'PusherObstacle-v0':
            state_info = list(obs.values())
            state_info = state_info[0:2]
            obs_robot = retrieve_np_state(state_info)
        elif self.args_mopa.env == 'SawyerPushObstacle-v0' or \
            self.args_mopa.env == 'SawyerAssemblyObstacle-v0' or \
            self.args_mopa.env == 'SawyerLiftObstacle-v0':
            obs_robot = np.concatenate((obs['joint_pos'], obs['joint_vel'], obs['gripper_qpos'], obs['gripper_qvel'], obs['eef_pos'], obs['eef_quat']))

        obs_robot = torch.from_numpy(obs_robot).float()
        obs_robot = obs_robot[None, :]
        return obs_img.cuda(), obs_robot.cuda()

    def evaluate(self, checkpoint):
        if self.model == 'BC_Visual_Policy':
            policy_eval = BC_Visual_Policy(robot_state=self.robot_state_size, num_classes=self.action_size, img_size=self.bc_visual_args.env_image_size)
        elif self.model == 'BC_Image_Only':
            policy_eval = BC_Image_Only(num_classes=self.action_size, img_size=self.bc_visual_args.env_image_size)
        elif self.model == 'BC_Robot_Only':
            policy_eval = BC_Robot_Only(robot_state=self.robot_state_size, num_classes=self.action_size)
        elif self.model == 'BC_Visual_Policy_Stochastic' or self.model == 'BC_Visual_Stochastic_w_Critics':
            policy_eval = BC_Visual_Policy_Stochastic(robot_state=self.robot_state_size, num_classes=self.action_size, img_size=self.bc_visual_args.env_image_size)

        policy_eval.cuda()
        policy_eval.load_state_dict(checkpoint['state_dict'])
        policy_eval.eval()
        
        num_ep = self.bc_visual_args.num_eval_ep_validation_per_seed

        total_success, total_rewards = 0, 0
        for eval_seed in tqdm.tqdm(self.eval_seeds):
            self._env_eval.set_seed(eval_seed)
            print("\n", colored("Running seed {}".format(eval_seed), "blue"))
            for ep in range(num_ep):            
                obs = self._env_eval.reset()

                obs_img, obs_robot = self.get_img_robot_state(obs)

                done = False
                ep_len = 0
                ep_rew = 0
                
                while ep_len < self.bc_visual_args.eval_bc_max_step_validation and not done:
                    if self.model == 'BC_Visual_Policy':
                        action = policy_eval(obs_img, obs_robot)
                    elif self.model == 'BC_Image_Only':
                        action = policy_eval(obs_img)
                    elif self.model == 'BC_Robot_Only':
                        action = policy_eval(obs_robot)
                    elif self.model == 'BC_Visual_Policy_Stochastic' or self.model == 'BC_Visual_Stochastic_w_Critics':
                        action = policy_eval(obs_img, obs_robot)

                    if len(action.shape) == 2:
                        action = action[0]                    
                    obs, reward, done, info = self._env_eval.step(action.detach().cpu().numpy(), is_bc_policy=True)
                    obs_img, obs_robot = self.get_img_robot_state(obs)
                    ep_len += 1
                    ep_rew += reward
                    if(ep_len % 100 == 0):
                        print(colored("Current Episode Step: {}, Reward: {}".format(ep_len, reward), "green"))

                print(colored("Current Episode Total Rewards: {}".format(ep_rew), "yellow"))
                if self._env_eval._success:
                    print(colored("Success!", "yellow"), "\n")
                    total_success += 1
                total_rewards += ep_rew
        del policy_eval
        return total_success, total_rewards
        
def retrieve_np_state(raw_state):
    for idx, values in enumerate(raw_state):
        if(idx==0):
            ot = np.array(values)
        else:
            ot = np.concatenate((ot, np.array(values)), axis=0)

    return ot

def get_arguments_from_mopa():
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
    return args_mopa

def overwrite_env_args(env_args):
    env_args.env = args.env
    env_args.env_image_size = args.env_image_size
    env_args.seed = args.env_seed
    env_args.screen_width = args.screen_width
    env_args.screen_height = args.screen_height
    env_args.obs_space = 'all'

def get_mopa_rl_agent(args_mopa, device, ckpt):
    mopa_config = copy.deepcopy(args_mopa)
    mopa_config.policy = 'mlp'
    mopa_config.obs_space = 'state'
    expert_actor, expert_critic = get_actor_critic_by_name(mopa_config.policy)

    ckpt = torch.load(ckpt)
    env = gym.make(mopa_config.env, **mopa_config.__dict__)
    ob_space = env.observation_space
    ac_space = env.action_space

    critic1 = expert_critic(mopa_config, ob_space, ac_space)    
    critic1.load_state_dict(ckpt["agent"]["critic1_state_dict"])
    critic1.to(device)
    critic1.eval()

    critic2 = expert_critic(mopa_config, ob_space, ac_space)    
    critic2.load_state_dict(ckpt["agent"]["critic2_state_dict"])
    critic2.to(device)
    critic2.eval()

    return critic1, critic2

def main():
    torch.set_num_threads(1)
    global val_loss_best
    val_loss_best = 1e10

    device = torch.device("cuda:0")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_num)

    args_mopa = get_arguments_from_mopa()

    mse_loss = nn.MSELoss()
    if args.model == 'BC_Visual_Stochastic_w_Critics':
        mse_critic1_loss = nn.MSELoss()
        mse_critic2_loss = nn.MSELoss()
        mopa_critic1, mopa_critic2 = get_mopa_rl_agent(args_mopa, device, args.mopa_ckpt)
    else:
        critic1, critic2 = None, None

    if args_mopa.env == 'PusherObstacle-v0':
        args.action_size = 4
        args.robot_state_size = 14
    elif args_mopa.env == 'SawyerPushObstacle-v0' or \
        args_mopa.env == 'SawyerAssemblyObstacle-v0':
        args.action_size = 7
        args.robot_state_size = 25
        args.env_state_size = 40
    elif args_mopa.env == 'SawyerLiftObstacle-v0':
        args.action_size = 8
        args.robot_state_size = 25
        args.env_state_size = 40
    else:
        print('ERROR: Incorrect env name')
        exit(1)

    # image augmentations
    transform = None
    # torchvision data augmentations style
    # if args.img_aug:
    #     img_augs = [] 
    #     if args.random_crop:
    #         img_augs.append(torchvision.transforms.RandomCrop(size=args.random_crop_size))
    #         args.env_image_size = args.random_crop_size
    #     transform = torchvision.transforms.Compose(img_augs)
    #     print('Applying data augmentations on images...')
    
    if args.model == 'BC_Visual_Policy':
        policy = BC_Visual_Policy(robot_state=args.robot_state_size, num_classes=args.action_size, img_size=args.env_image_size)
    elif args.model == 'BC_Image_Only':
        policy = BC_Image_Only(num_classes=args.action_size, img_size=args.env_image_size)
    elif args.model == 'BC_Robot_Only':
        policy = BC_Robot_Only(robot_state=args.robot_state_size, num_classes=args.action_size)
    elif args.model == 'BC_Visual_Policy_Stochastic':
        policy = BC_Visual_Policy_Stochastic(robot_state=args.robot_state_size, num_classes=args.action_size, img_size=args.env_image_size)
    elif args.model == 'BC_Visual_Stochastic_w_Critics':
        policy = BC_Visual_Policy_Stochastic(robot_state=args.robot_state_size, num_classes=args.action_size, img_size=args.env_image_size)
        critic1 = BC_Visual_Policy(robot_state=args.action_size + args.robot_state_size, num_classes=1, img_size=args.env_image_size)
        critic2 = BC_Visual_Policy(robot_state=args.action_size + args.robot_state_size, num_classes=1, img_size=args.env_image_size)
    else:
        print(colored('ERROR: Do not support this model {}'.format(args.model)), 'red')
        exit(1)

    if args.wandb:
        wandb.init(
            project="mopa-rl-bc-visual",
            config={k: v for k, v in args.__dict__.items()}
        )
        wandb.watch(policy)

    print(colored('Training model {}'.format(args.model), 'blue'))
    print(policy)

    if args.model == 'BC_Visual_Stochastic_w_Critics':
        critic1_optimizer = optim.Adam(list(critic1.parameters()), lr = args.lrate, betas = (args.beta1, args.beta2))
        critic2_optimizer = optim.Adam(list(critic2.parameters()), lr = args.lrate, betas = (args.beta1, args.beta2))
        scheduler_critic1 = optim.lr_scheduler.StepLR(critic1_optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
        scheduler_critic2 = optim.lr_scheduler.StepLR(critic2_optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    optimizer = optim.Adam(list(policy.parameters()), lr = args.lrate, betas = (args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    # load from checkpoint
    if args.load_saved:
        checkpoint = torch.load(os.path.join(args.model_save_dir, args.checkpoint), map_location='cuda:0')
        start_epoch = checkpoint['epoch']
        policy.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # fix for bug: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
        # https://github.com/pytorch/pytorch/issues/2830#issuecomment-336031198
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        scheduler.load_state_dict(checkpoint['scheduler'])
        val_loss_best = checkpoint['val_loss_best']

        if args.model == 'BC_Visual_Stochastic_w_Critics':
            critic1.load_state_dict(checkpoint['critic1_state_dict'])
            critic2.load_state_dict(checkpoint['critic2_state_dict'])
    else:
        start_epoch = args.start_epoch

    dataset = StateImageActionDataset(args, args.saved_rollouts_file, transform=transform)
    dataset_length = len(dataset)
    train_size = int(args.train_data_ratio * dataset_length)
    test_size = dataset_length - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    train_loss = []
    val_loss = []

    policy.cuda()
    mse_loss.cuda()
    if args.model == 'BC_Visual_Stochastic_w_Critics':
        critic1.cuda()
        critic2.cuda()    
        mse_critic1_loss.cuda()
        mse_critic2_loss.cuda()
    
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    
    evaluation_obj = Evaluation(args, args_mopa)

    print('Total number of state-action pairs: ', dataset_length)
    print('Number of training state-action pairs: ', len(train_dataset))
    print('Number of test state-action pairs: ', len(test_dataset))
    outer = tqdm.tqdm(total=args.end_epoch-start_epoch, desc='Epoch', position=start_epoch)
    for epoch in range(start_epoch, args.end_epoch):
        total_loss = 0.0
        total_critic1_loss = 0.0
        total_critic2_loss = 0.0
        validation_loss = 0.0
        
        policy.train()

        print('\nprocessing training batch...')
        for i, batch in enumerate(dataloader_train):
            ob, ac, img = batch['ob'], batch['ac'], batch['img']
            ob = ob.float().cuda()
            ac = ac.float().cuda()
            img = img.float().cuda()

            if args.model == 'BC_Visual_Policy':
                ac_pred = policy(img, ob)
            elif args.model == 'BC_Image_Only':
                ac_pred = policy(img)
            elif args.model == 'BC_Robot_Only':
                ac_pred = policy(ob)
            elif args.model == 'BC_Visual_Policy_Stochastic':
                ac_pred = policy(img, ob)
            elif args.model == 'BC_Visual_Stochastic_w_Critics':
                ac_pred = policy(img, ob)

            if args.model == 'BC_Visual_Stochastic_w_Critics':
                ac_predictor_loss = mse_loss(ac_pred, ac)

                ac_pred_detached = ac_pred.detach()
                inp_robot_ac_state = torch.cat([ob, ac_pred_detached], axis=1)
                critic1_pred = critic1(img, inp_robot_ac_state)
                critic2_pred = critic2(img, inp_robot_ac_state)

                full_state_ob = batch['full_state_ob'].float().cuda()
                critic1_data = mopa_critic1.forward_tensor(full_state_ob, ac_pred_detached)
                critic2_data = mopa_critic2.forward_tensor(full_state_ob, ac_pred_detached)

                critic1_prediction_loss = mse_critic1_loss(critic1_pred, critic1_data)
                critic2_prediction_loss = mse_critic2_loss(critic2_pred, critic2_data)

                loss =  ac_predictor_loss + critic1_prediction_loss + critic2_prediction_loss

                optimizer.zero_grad()
                critic1_optimizer.zero_grad()
                critic2_optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                critic1_optimizer.step()
                critic2_optimizer.step()

                total_loss += ac_predictor_loss.data.item()
                total_critic1_loss += critic1_prediction_loss.data.item()
                total_critic2_loss += critic2_prediction_loss.data.item()
            else:
                # ac mse
                ac_predictor_loss = mse_loss(ac_pred, ac)
                optimizer.zero_grad()
                ac_predictor_loss.backward()
                optimizer.step()
                total_loss += ac_predictor_loss.data.item()

        training_loss = total_loss / (args.batch_size*len(dataloader_train))
        train_loss.append(training_loss)
        critic1_loss = total_critic1_loss / (args.batch_size*len(dataloader_train))
        critic2_loss = total_critic2_loss / (args.batch_size*len(dataloader_train))

        print('')
        print('----------------------------------------------------------------------')
        print('Epoch #' + str(epoch))
        print('Action Prediction Loss (Train): ' + str(training_loss))
        print('Critic1 loss: ', str(critic1_loss))
        print('Critic2 loss: ', str(critic2_loss))
        print('----------------------------------------------------------------------')
        
        # evaluating on test set
        policy.eval()
        
        action_predictor_loss_val = 0.
        
        print('\nprocessing test batch...')
        for i, batch in enumerate(dataloader_test):
            ob, ac, img = batch['ob'], batch['ac'], batch['img']
            ob = ob.float().cuda()
            ac = ac.float().cuda()
            img = img.float().cuda()

            if args.model == 'BC_Visual_Policy':
                ac_pred = policy(img, ob)
            elif args.model == 'BC_Image_Only':
                ac_pred = policy(img)
            elif args.model == 'BC_Robot_Only':
                ac_pred = policy(ob)
            elif args.model == 'BC_Visual_Policy_Stochastic':
                ac_pred = policy(img, ob)

            action_predictor_loss_val = mse_loss(ac_pred, ac)
            validation_loss += action_predictor_loss_val.data.item()
        
        validation_loss /= (args.batch_size * len(dataloader_test))
        val_loss.append(validation_loss)
            
        print('')
        print('**********************************************************************')
        print('Epoch #' + str(epoch))
        print('')
        print('Action Prediction Loss (Test): ' + str(validation_loss))
        print()
        print('**********************************************************************')

        scheduler.step()
        if args.model == 'BC_Visual_Stochastic_w_Critics':
            scheduler_critic1.step()
            scheduler_critic2.step()

        if(validation_loss<val_loss_best):
            val_loss_best = validation_loss
            print(colored("BEST VAL LOSS: {}".format(val_loss_best), "yellow"))
   
        # arrange/save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': policy.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss_best': val_loss_best,
        }
        if critic1 and critic2:
            checkpoint['critic1_state_dict'] = critic1.state_dict()
            checkpoint['critic2_state_dict'] = critic2.state_dict()
        torch.save(checkpoint, os.path.join(args.model_save_dir, 'epoch_{}.pth'.format(epoch)))

        # perform validation
        if epoch % args.eval_interval == 0:
            total_success, total_rewards = evaluation_obj.evaluate(checkpoint)
        else:
            total_success, total_rewards = -1, -1

        # wandb logging 
        if args.wandb:
            wandb.log({
                "Epoch": epoch,
                "Total Success": total_success,
                "Total Rewards": total_rewards,
                "Action Prediction Loss (Train)": training_loss,
                "Action Prediction Loss (Test)": validation_loss,
                "Critic1 Loss": critic1_loss,
                "Critic2 Loss": critic2_loss,
            })
        else:
            plt.plot(train_loss, label="train loss")
            plt.plot(val_loss, label="validation loss")
            plt.legend()
            plt.savefig(os.path.join(args.bc_video_dir, 'train_loss_plots.png'))
            plt.close()
        outer.update(1)

if __name__ == "__main__":
    main()

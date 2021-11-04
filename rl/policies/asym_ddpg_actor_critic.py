from collections import OrderedDict

import torch
import torch.nn as nn
from gym import spaces

from rl.policies.utils import  MLP, BC_Visual_Policy
from rl.policies.simple_actor_critic import Actor, Critic
from util.gym import observation_size, action_size, goal_size, box_size, robot_state_size, image_size
import numpy as np
from util.pytorch import to_tensor

class AsymDDPGActor(Actor):
    def __init__(
        self,
        config,
        ob_space,
        ac_space,
        tanh_policy,
        ac_scale,
        deterministic=False,
        activation="relu",
        rl_hid_size=None,
    ):
        super().__init__()
        self._config = config
        self._ac_space = ac_space
        self._deterministic = deterministic
        self.env = config.env
        self._activation_fn = torch.tanh
        self._ac_scale = ac_scale

        if config.env == 'PusherObstacle-v0':
            # observation (excluding goal information)
            input_dim = observation_size(ob_space) - goal_size(ob_space) - box_size(ob_space)
        elif 'Sawyer' in config.env:
            input_dim = robot_state_size(ob_space)
        else:
            raise NotImplementedError

        self.cnn = BC_Visual_Policy(robot_state=input_dim, num_classes=action_size(ac_space), img_size=config.env_image_size)
        # self.cnn.load_pretrained(config.bc_checkpoint)
        # print('load pretrained BC weights to the actor from {}'.format(config.bc_checkpoint))

    def forward(self, ob):
        if self.env == 'PusherObstacle-v0':
            inp = list(ob.values())
            if self._config.obs_space == "all":
                inp_robot_state = inp[:2]
                if len(inp[0].shape) == 1: 
                    inp_robot_state = [x.unsqueeze(0) for x in inp_robot_state]
                inp_robot_state = torch.cat(inp_robot_state, dim=-1)
                inp_img = inp[5]
                if len(inp_img.shape) == 5:
                    inp_img = inp_img.squeeze(1) # remove unnecessary dimension
            else:
                raise NotImplementedError
        elif 'Sawyer' in self.env:
            if len(ob['joint_pos'].shape) == 1:
                inp_robot_state = torch.cat([ob['joint_pos'], ob['joint_vel'], ob['gripper_qpos'], ob['gripper_qvel'], ob['eef_pos'], ob['eef_quat']])
                inp_robot_state = inp_robot_state[None, :]
                inp_img = ob['image']
            elif len(ob['joint_pos'].shape) == 2:
                inp_robot_state = torch.cat([ob['joint_pos'], ob['joint_vel'], ob['gripper_qpos'], ob['gripper_qvel'], ob['eef_pos'], ob['eef_quat']], axis=1)
                inp_img = ob['image']
            if len(inp_img.shape) == 5:
                inp_img = inp_img.squeeze(1) # remove unnecessary dimension
        else:
            raise NotImplementedError

        out = self._activation_fn(self.cnn(inp_img, inp_robot_state))
        out = torch.reshape(out, (out.shape[0], -1))
        out = torch.clip(out, -self._ac_scale, self._ac_scale)
        return out

    def act(self, ob, act_noise=None):
        ob_copy = ob.copy()
        if 'image' in ob.keys() and isinstance(ob['image'], str):
            ob_copy['image'] = np.load(ob_copy['image'])
        ob_copy = to_tensor(ob_copy, self._config.device)
        actions_vals = self.forward(ob_copy)
        actions_vals = actions_vals.detach().cpu().numpy().squeeze(0)
        if act_noise:
            actions_vals += act_noise * np.random.randn(action_size(self._ac_space))
        ob_copy.clear()
        return actions_vals


class AsymDDPGCritic(Critic):
    def __init__(
        self, config, ob_space, ac_space=None, activation="relu", rl_hid_size=None
    ):
        super().__init__()
        self._config = config
        self.env = config.env

        if config.env == 'PusherObstacle-v0':
            # observation (including goal information)
            input_dim = observation_size(ob_space)
        elif 'Sawyer' in config.env and 'image' not in ob_space:
            input_dim = observation_size(ob_space)
        elif 'Sawyer' in config.env and 'image' in ob_space:
            input_dim = observation_size(ob_space) - image_size(ob_space)
        else:
            raise NotImplementedErro

        if ac_space is not None:
            input_dim += action_size(ac_space)

        if rl_hid_size == None:
            rl_hid_size = config.rl_hid_size

        self.fc = MLP(config, input_dim, 1, [rl_hid_size] * 2, activation=activation)

    def forward(self, ob, ac=None):
        inp = list(ob.values())
        if self.env == 'PusherObstacle-v0':
            if self._config.obs_space == "all":
                # only use robot and environment state (env. image is not used) 
                inp = inp[:5]
            else:
                raise NotImplementedError
        elif 'Sawyer' in self.env:
            inp = inp[:-1]
        else:
            raise NotImplementedError

        if len(inp[0].shape) == 1:
            inp = [x.unsqueeze(0) for x in inp]
        if ac is not None:
            if isinstance(ac, OrderedDict):
                ac = list(ac.values())
            else:
                ac = [ac]
            if len(ac[0].shape) == 1:
                ac = [x.unsqueeze(0) for x in ac]
            inp.extend(ac)
        
        out = self.fc(torch.cat(inp, dim=-1))
        out = torch.reshape(out, (out.shape[0], 1))

        return out

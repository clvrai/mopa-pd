from collections import OrderedDict

import torch
import torch.nn as nn
from gym import spaces

from rl.policies.utils import  MLP, BC_Visual_Policy
from rl.policies.actor_critic import Actor, Critic
from util.gym import observation_size, action_size, goal_size, box_size, robot_state_size, image_size
import numpy as np

from rl.policies.distributions import (
    FixedCategorical,
    FixedNormal,
    MixedDistribution,
    FixedGumbelSoftmax,
)
from util.pytorch import to_tensor
import torch.nn.functional as F

class ImageActor(Actor):
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
        super().__init__(config, ob_space, ac_space, tanh_policy, ac_scale)

        self._ac_space = ac_space
        self._deterministic = deterministic
        self.env = config.env
        self._ac_scale = ac_scale

        if rl_hid_size == None:
            rl_hid_size = config.rl_hid_size

        if config.env == 'PusherObstacle-v0':
            # observation (excluding goal information)
            input_dim = observation_size(ob_space) - goal_size(ob_space) - box_size(ob_space)
        elif 'Sawyer' in config.env:
            input_dim = robot_state_size(ob_space)
        else:
            raise NotImplementedError

        self.cnn = BC_Visual_Policy(robot_state=input_dim, num_classes=256, img_size=config.env_image_size)

        self.fc_means = nn.ModuleDict()
        self.fc_log_stds = nn.ModuleDict()

        for k, space in ac_space.spaces.items():
            if isinstance(space, spaces.Box):
                self.fc_means.update(
                    {
                        k: MLP(
                            config,
                            rl_hid_size,
                            action_size(space),
                            activation=activation,
                        )
                    }
                )
                if not self._deterministic:
                    self.fc_log_stds.update(
                        {
                            k: MLP(
                                config,
                                rl_hid_size,
                                action_size(space),
                                activation=activation,
                            )
                        }
                    )
            elif isinstance(space, spaces.Discrete):
                self.fc_means.update(
                    {k: MLP(config, rl_hid_size, space.n, activation=activation)}
                )
            else:
                self.fc_means.update(
                    {k: MLP(config, rl_hid_size, space, activation=activation)}
                )

    def forward(self, ob, deterministic=False):
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
                out = self._activation_fn(self.cnn(inp_img, inp_robot_state))
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
            out = self._activation_fn(self.cnn(inp_img, inp_robot_state))
        else:
            raise NotImplementedError

        out = torch.reshape(out, (out.shape[0], -1))

        means, stds = OrderedDict(), OrderedDict()

        for k, space in self._ac_space.spaces.items():
            mean = self.fc_means[k](out)
            if isinstance(space, spaces.Box) and not self._deterministic:
                if self._config.algo == "ppo":
                    zeros = torch.zeros(mean.size()).to(self._config.device)
                    log_std = self.fc_log_stds[k](zeros)
                else:
                    log_std = self.fc_log_stds[k](out)
                    log_std = torch.clamp(log_std, -10, 2)
                std = torch.exp(log_std.double())
            else:
                std = None
            means[k] = mean
            stds[k] = std
        return means, stds

    def act(self, ob, is_train=True, return_log_prob=False, return_stds=False):
        ob_copy = ob.copy()
        if 'image' in ob.keys() and isinstance(ob['image'], str):
            ob_copy['image'] = np.load(ob_copy['image'])
        ob_copy = to_tensor(ob_copy, self._config.device)
        means, stds = self.forward(ob_copy, self._deterministic)

        ob_copy.clear()

        dists = OrderedDict()
        for k, space in self._ac_space.spaces.items():
            if isinstance(space, spaces.Box):
                if self._deterministic:
                    stds[k] = torch.zeros_like(means[k])
                dists[k] = FixedNormal(means[k], stds[k])
            else:
                if self._config.algo == "sac" or "aac" in self._config.algo:
                    dists[k] = FixedGumbelSoftmax(
                        torch.tensor(self._config.temperature), logits=means[k]
                    )
                else:
                    dists[k] = FixedCategorical(logits=means[k])

        actions = OrderedDict()
        mixed_dist = MixedDistribution(dists)
        if not is_train or self._deterministic:
            activations = mixed_dist.mode()
        else:
            activations = mixed_dist.sample()

        if return_log_prob:
            log_probs = mixed_dist.log_probs(activations)

        for k, space in self._ac_space.spaces.items():
            z = activations[k]
            if self._tanh and isinstance(space, spaces.Box):
                action = torch.tanh(z)
                if return_log_prob:
                    # follow the Appendix C. Enforcing Action Bounds
                    log_det_jacobian = 2 * (np.log(2.0) - z - F.softplus(-2.0 * z)).sum(
                        dim=-1, keepdim=True
                    )
                    log_probs[k] = log_probs[k] - log_det_jacobian
            else:
                action = z

            action = torch.clip(action, -self._ac_scale, self._ac_scale)
            actions[k] = action.detach().cpu().numpy().squeeze(0)
            activations[k] = z.detach().cpu().numpy().squeeze(0)

        if return_log_prob:
            log_probs_ = torch.cat(list(log_probs.values()), -1).sum(-1, keepdim=True)
            # if log_probs_.min() < -100:
            #     print('sampling an action with a probability of 1e-100')
            #     import ipdb; ipdb.set_trace()

            log_probs_ = log_probs_.detach().cpu().numpy().squeeze(0)
            return actions, activations, log_probs_

        elif return_stds:
            return actions, activations, stds
        else:
            return actions, activations

    def act_log(self, ob, activations=None):
        means, stds = self.forward(ob)

        dists = OrderedDict()
        actions = OrderedDict()
        for k, space in self._ac_space.spaces.items():
            if isinstance(space, spaces.Box):
                if self._deterministic:
                    stds[k] = torch.zeros_like(means[k])
                dists[k] = FixedNormal(means[k], stds[k])
            else:
                if self._config.algo == "sac" or "aac" in self._config.algo:
                    dists[k] = FixedGumbelSoftmax(
                        torch.tensor(self._config.temperature), logits=means[k]
                    )
                else:
                    dists[k] = FixedCategorical(logits=means[k])

        mixed_dist = MixedDistribution(dists)

        activations_ = mixed_dist.rsample() if activations is None else activations
        for k in activations_.keys():
            if len(activations_[k].shape) == 1:
                activations_[k] = activations_[k].unsqueeze(0)
        log_probs = mixed_dist.log_probs(activations_)

        for k, space in self._ac_space.spaces.items():
            z = activations_[k]
            if self._tanh and isinstance(space, spaces.Box):
                action = torch.tanh(z)
                # follow the Appendix C. Enforcing Action Bounds
                log_det_jacobian = 2 * (np.log(2.0) - z - F.softplus(-2.0 * z)).sum(
                    dim=-1, keepdim=True
                )
                log_probs[k] = log_probs[k] - log_det_jacobian
            else:
                action = z

            action = torch.clip(action, -self._ac_scale, self._ac_scale)
            actions[k] = action

        log_probs_ = torch.cat(list(log_probs.values()), -1).sum(-1, keepdim=True)
        # if log_probs_.min() < -100:
        #     print(ob)
        #     print(log_probs_.min())
        #     import ipdb; ipdb.set_trace()
        if activations is None:
            return actions, log_probs_
        else:
            ents = mixed_dist.entropy()
            return log_probs_, ents

    def load_partial_layers(self, state_dict):
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in self.state_dict().keys():
                filtered_dict[k] = v
        self.load_state_dict(filtered_dict, strict=False)
        return None

    def load_state_dict_processed(self, state_dict):
        processed_dict = {}
        for k, v in state_dict.items():
            if 'cnn' in k or 'linear_layers' in k:
                k = 'cnn.' + k
                assert k in self.state_dict().keys()
            elif 'fc' in k:
                tokens = k.split('.')
                k = tokens[0] + '.default.fc.' + tokens[1] + '.' + tokens[2]
                assert k in self.state_dict().keys()
            else:
                print('incorrect checkpoint')
                exit(1)
            processed_dict[k] = v
        self.load_state_dict(processed_dict)
        return True

class ImageCritic(Critic):
    def __init__(
        self, config, ob_space, ac_space=None, activation="relu", rl_hid_size=None
    ):
        super().__init__(config)

        self.env = config.env

        if config.env == 'PusherObstacle-v0':
            # observation (excluding goal information)
            input_dim = observation_size(ob_space) - goal_size(ob_space) - box_size(ob_space)
        elif 'Sawyer' in config.env:
            input_dim = robot_state_size(ob_space)
        else:
            raise NotImplementedError

        if ac_space is not None:
            input_dim += action_size(ac_space)

        self.cnn = BC_Visual_Policy(robot_state=input_dim, num_classes=1, img_size=config.env_image_size)

    def forward(self, ob, ac=None):
        if ac is not None:
            ac = list(ac.values())
            if len(ac[0].shape) == 1:
                ac = [x.unsqueeze(0) for x in ac]

        if 'Sawyer' in self.env:
            if len(ob['joint_pos'].shape) == 1:
                inp_robot_ac_state = torch.cat([ob['joint_pos'], ob['joint_vel'], ob['gripper_qpos'], ob['gripper_qvel'], ob['eef_pos'], ob['eef_quat'], ac[0]])
                inp_robot_ac_state = inp_robot_ac_state[None, :]
                inp_img = ob['image']
            elif len(ob['joint_pos'].shape) == 2:
                inp_robot_ac_state = torch.cat([ob['joint_pos'], ob['joint_vel'], ob['gripper_qpos'], ob['gripper_qvel'], ob['eef_pos'], ob['eef_quat'], ac[0]], axis=1)
                inp_img = ob['image']
            if len(inp_img.shape) == 5:
                inp_img = inp_img.squeeze(1) # remove unnecessary dimension
            out = self.cnn(inp_img, inp_robot_ac_state)
        else:
            raise NotImplementedError

        return out

    def load_state_dict_processed(self, state_dict):
        processed_dict = {}
        for k, v in state_dict.items():
            if 'cnn' in k or 'linear_layers' in k:
                k = 'cnn.' + k
                assert k in self.state_dict().keys()
            else:
                print('incorrect checkpoint')
                exit(1)
            processed_dict[k] = v
        self.load_state_dict(processed_dict)
        return True

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

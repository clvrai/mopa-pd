from collections import OrderedDict

import torch
import torch.nn as nn
from gym import spaces

from rl.policies.utils import CNN, MLP, resnet18, mobilenet_v3_small, SimpleCNN, BC_Visual_Policy
from rl.policies.actor_critic import Actor, Critic
from util.gym import observation_size, action_size


class MlpActor(Actor):
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
        super().__init__(config, ob_space, ac_space, tanh_policy)

        self._ac_space = ac_space
        self._deterministic = deterministic
        if rl_hid_size == None:
            rl_hid_size = config.rl_hid_size

        # observation
        input_dim = observation_size(ob_space)

        if self._config.obs_space == "image": 
            # self.cnn = resnet18(robot_state=input_dim, num_classes=256)
            # self.cnn = mobilenet_v3_small(robot_state=input_dim, num_classes=256)
            # self.cnn = SimpleCNN(robot_state=input_dim, num_classes=256)
            self.cnn = BC_Visual_Policy(robot_state=input_dim, num_classes=256)

            if self._config.load_pretrained:
                self.cnn.load_pretrained(self._config.checkpoint_path)
                print('load pretrained weights for actor from {}'.format(self._config.checkpoint_path))
        else:
            self.fc = MLP(
                config,
                input_dim,
                rl_hid_size,
                [rl_hid_size] * config.actor_num_hid_layers,
                activation=activation,
            )
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

        if self._config.load_pretrained and self._config.bc_loss:
            self.bc_visual_model = BC_Visual_Policy(robot_state=input_dim, num_classes=action_size(ac_space))
            self.bc_visual_model.load_pretrained(self._config.checkpoint_path)
            for param in self.bc_visual_model.parameters():
                param.requires_grad = False
            self.bc_visual_model.eval()

    def forward(self, ob, deterministic=False):
        inp = list(ob.values())
        if self._config.obs_space == "image":
            inp_robot_state = inp[:2]
            if len(inp[0].shape) == 1: 
                inp_robot_state = [x.unsqueeze(0) for x in inp_robot_state]
            inp_robot_state = torch.cat(inp_robot_state, dim=-1)
            inp_img = inp[2]
            if len(inp_img.shape) == 5:
                inp_img = inp_img.squeeze(1) # remove unnecessary dimension
            out = self._activation_fn(self.cnn(inp_img, inp_robot_state))
        else:
            if len(inp[0].shape) == 1:
                inp = [x.unsqueeze(0) for x in inp]
            out = self._activation_fn(self.fc(torch.cat(inp, dim=-1)))
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

    def process_data_for_mopa_rl(self, ob):
        if self._config.env == 'PusherObstacle-v0':
            # convert ob format to obs_space==state in _get_obs of pusher_obstacle.py
            new_ob = OrderedDict(
                [
                    ("default", torch.cat([ob['default'][:, :-4], ob['box_qpos'], ob['default'][:, -4:], ob['box_vel']], dim=1)),
                    ("fingertip", ob['fingertip']),
                    ("goal", ob['goal']),
                ]
            )
            return new_ob
        elif 'Sawyer' in self._config.env and 'image' in ob:
            ob_copy = ob.copy()
            del ob_copy['image']
            return ob_copy
        return ob

    def get_predicted_ac(self, ob):
        ob_copy = self.process_data_for_mopa_rl(ob)
        ac, _ = self.act_log(ob_copy)
        return ac['default']

class MlpCritic(Critic):
    def __init__(
        self, config, ob_space, ac_space=None, activation="relu", rl_hid_size=None
    ):
        super().__init__(config)

        input_dim = observation_size(ob_space)
        if ac_space is not None:
            input_dim += action_size(ac_space)

        if rl_hid_size == None:
            rl_hid_size = config.rl_hid_size

        if self._config.obs_space == "image": 
            # self.cnn = resnet18(robot_state=input_dim, num_classes=1)
            # self.cnn = mobilenet_v3_small(robot_state=input_dim, num_classes=1)
            # self.cnn = SimpleCNN(robot_state=input_dim, num_classes=1)
            self.cnn = BC_Visual_Policy(robot_state=input_dim, num_classes=1)

            if self._config.load_pretrained:
                self.cnn.load_pretrained(self._config.checkpoint_path)
                print('load pretrained weights for critic from {}'.format(self._config.checkpoint_path))
        else:
            self.fc = MLP(config, input_dim, 1, [rl_hid_size] * 2, activation=activation)

    def forward(self, ob, ac=None):
        inp = list(ob.values())
        if self._config.obs_space == "image": 
            inp_robot_state = inp[:2]
            if len(inp[0].shape) == 1:
                inp_robot_state = [x.unsqueeze(0) for x in inp_robot_state]
            inp_img = inp[2]
            if len(inp_img.shape) == 5:
                inp_img = inp_img.squeeze(1) # remove unnecessary dimension
            if ac is not None:
                ac = list(ac.values())
                if len(ac[0].shape) == 1:
                    ac = [x.unsqueeze(0) for x in ac]
                inp_robot_state.extend(ac)
            inp_robot_state = torch.cat(inp_robot_state, dim=-1)
            out = self.cnn(inp_img, inp_robot_state)
        else:
            if len(inp[0].shape) == 1:
                inp = [x.unsqueeze(0) for x in inp]
            if ac is not None:
                ac = list(ac.values())
                if len(ac[0].shape) == 1:
                    ac = [x.unsqueeze(0) for x in ac]
                inp.extend(ac)

            out = self.fc(torch.cat(inp, dim=-1))
        out = torch.reshape(out, (out.shape[0], 1))

        return out

    def forward_tensor(self, ob, ac):
        ob_tensor = torch.cat([ob, ac], axis=1)
        out = self.fc(ob_tensor)
        return out

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

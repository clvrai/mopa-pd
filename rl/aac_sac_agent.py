# SAC training code reference
# https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py

import copy
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym import spaces

from rl.dataset import ReplayBuffer, RandomSampler, ReplayBufferIterableDatasetDisk, ReplayBufferIterableDataset
from rl.base_agent import BaseAgent
from rl.planner_agent import PlannerAgent
from util.logger import logger
from util.mpi import mpi_average
from util.pytorch import (
    optimizer_cuda,
    count_parameters,
    sync_networks,
    sync_grads,
    to_tensor,
)
from util.gym import action_size, observation_size
from rl.policies.utils import BC_Visual_Policy
from util.gym import observation_size, action_size, goal_size, box_size, robot_state_size



class AACSACAgent(BaseAgent):
    def __init__(
        self,
        config,
        ob_space,
        ac_space,
        actor,
        critic,
        non_limited_idx=None,
        ref_joint_pos_indexes=None,
        joint_space=None,
        is_jnt_limited=None,
        jnt_indices=None,
    ):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space
        self._jnt_indices = jnt_indices
        self._ref_joint_pos_indexes = ref_joint_pos_indexes
        self._log_alpha = torch.tensor(
            np.log(config.alpha), requires_grad=True, device=config.device
        )
        self._alpha_optim = optim.Adam([self._log_alpha], lr=config.lr_actor, eps=1e-04, weight_decay=1e-04)
        self._joint_space = joint_space
        self._is_jnt_limited = is_jnt_limited
        if joint_space is not None:
            self._jnt_minimum = joint_space["default"].low
            self._jnt_maximum = joint_space["default"].high

        self.bc_mse_loss = nn.MSELoss()

        # build up networks
        self._build_actor(actor)
        self._build_critic(critic)
        self._network_cuda(config.device)

        self._target_entropy = -action_size(self._actor._ac_space)

        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_actor, eps=1e-04, weight_decay=1e-04)
        self._critic1_optim = optim.Adam(
            self._critic1.parameters(), lr=config.lr_critic, eps=1e-04, weight_decay=0.001
        )
        self._critic2_optim = optim.Adam(
            self._critic2.parameters(), lr=config.lr_critic, eps=1e-04, weight_decay=0.001
        )

        sampler = RandomSampler()
        buffer_keys = ["ob", "ac", "meta_ac", "done", "rew"]
        if config.mopa or config.expand_ac_space:
            buffer_keys.append("intra_steps")

        if config.parallel_dataloading:
            if config.parallel_dataloading_mode == 'disk':
                def collate_transitions(batch):
                    transitions, ob_images, ob_next_images = zip(*batch)
                    transitions[0]['ob']['image'] = ob_images[0]
                    transitions[0]['ob_next']['image'] = ob_next_images[0]
                    return transitions[0]
                self._buffer = ReplayBufferIterableDatasetDisk(buffer_keys, config.buffer_size, config.batch_size)
                self._buffer_loader = torch.utils.data.DataLoader(self._buffer, batch_size=1, num_workers=0, collate_fn=collate_transitions)
            elif config.parallel_dataloading_mode == 'ram':
                self._buffer = ReplayBufferIterableDataset(buffer_keys, config.buffer_size, config.batch_size)
                self._buffer_loader = torch.utils.data.DataLoader(self._buffer, batch_size=1, num_workers=0, collate_fn=lambda batch: batch[0])
            else:
                raise NotImplementedError
        else:
            self._buffer = ReplayBuffer(
                buffer_keys, config.buffer_size, sampler.sample_func
            )

        self._log_creation()

        self._planner = None
        self._is_planner_initialized = False

        if config.env == 'PusherObstacle-v0':
            # observation (excluding goal information)
            input_dim = observation_size(ob_space) - goal_size(ob_space) - box_size(ob_space)
        elif 'Sawyer' in config.env:
            input_dim = robot_state_size(ob_space)
        else:
            raise NotImplementedError
        self.expert_policy = BC_Visual_Policy(robot_state=input_dim, num_classes=action_size(ac_space), img_size=config.env_image_size, device=config.device, env=config.env)
        self.expert_policy.load_bc_weights(config.bc_checkpoint)
        self.expert_policy.eval()
        self.expert_policy.cuda()
        print('load pretrained BC weights to BC model from {}'.format(self._config.bc_checkpoint))

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("creating a AAC agent")
            logger.info("the actor has %d parameters", count_parameters(self._actor))
            logger.info(
                "the critic1 has %d parameters", count_parameters(self._critic1)
            )
            logger.info(
                "the critic2 has %d parameters", count_parameters(self._critic2)
            )

    def _build_actor(self, actor):
        self._actor = actor(
            self._config,
            self._ob_space,
            self._ac_space,
            self._config.tanh_policy,
        )

    def _build_critic(self, critic):
        config = self._config
        self._critic1 = critic(config, self._ob_space, self._ac_space)
        self._critic2 = critic(config, self._ob_space, self._ac_space)

        # build up target networks
        self._critic1_target = critic(config, self._ob_space, self._ac_space)
        self._critic2_target = critic(config, self._ob_space, self._ac_space)
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts)

    def valid_action(self, ac):
        return np.all(ac["default"] >= -1.0) and np.all(ac["default"] <= 1.0)

    def clip_qpos(self, curr_qpos):
        tmp_pos = curr_qpos.copy()
        if np.any(
            curr_qpos[self._is_jnt_limited[self._jnt_indices]]
            < self._jnt_minimum[self._jnt_indices][
                self._is_jnt_limited[self._jnt_indices]
            ]
        ) or np.any(
            curr_qpos[self._is_jnt_limited[self._jnt_indices]]
            > self._jnt_maximum[self._jnt_indices][
                self._is_jnt_limited[self._jnt_indices]
            ]
        ):
            new_curr_qpos = np.clip(
                curr_qpos.copy(),
                self._jnt_minimum[self._jnt_indices] + self._config.joint_margin,
                self._jnt_maximum[self._jnt_indices] - self._config.joint_margin,
            )
            new_curr_qpos[np.invert(self._is_jnt_limited[self._jnt_indices])] = tmp_pos[
                np.invert(self._is_jnt_limited[self._jnt_indices])
            ]
            curr_qpos = new_curr_qpos
        return curr_qpos

    def state_dict(self):
        return {
            "log_alpha": self._log_alpha.cpu().detach().numpy(),
            "actor_state_dict": self._actor.state_dict(),
            "critic1_state_dict": self._critic1.state_dict(),
            "critic2_state_dict": self._critic2.state_dict(),
            "alpha_optim_state_dict": self._alpha_optim.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
            "critic1_optim_state_dict": self._critic1_optim.state_dict(),
            "critic2_optim_state_dict": self._critic2_optim.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._log_alpha.data = torch.tensor(
            ckpt["log_alpha"], requires_grad=True, device=self._config.device
        )
        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._critic1.load_state_dict(ckpt["critic1_state_dict"])
        self._critic2.load_state_dict(ckpt["critic2_state_dict"])

        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())

        self._network_cuda(self._config.device)

        self._alpha_optim.load_state_dict(ckpt["alpha_optim_state_dict"])
        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self._critic1_optim.load_state_dict(ckpt["critic1_optim_state_dict"])
        self._critic2_optim.load_state_dict(ckpt["critic2_optim_state_dict"])

        optimizer_cuda(self._alpha_optim, self._config.device)
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic1_optim, self._config.device)
        optimizer_cuda(self._critic2_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._critic1.to(device)
        self._critic2.to(device)
        self._critic1_target.to(device)
        self._critic2_target.to(device)

    def sync_networks(self):
        if self._config.is_mpi:
            sync_networks(self._actor)
            sync_networks(self._critic2)
            sync_networks(self._critic2)

    def train(self):
        for i in range(self._config.num_batches):            
            if self._config.parallel_dataloading:
                transitions = next(iter(self._buffer_loader))
            else:
                transitions = self._buffer.sample(self._config.batch_size)

            train_info = self._update_network(transitions, i)
            self._soft_update_target_network(
                self._critic1_target, self._critic1, self._config.polyak
            )
            self._soft_update_target_network(
                self._critic2_target, self._critic2, self._config.polyak
            )
            del transitions['ob']['image']
            transitions['ob']['image'] = ''
            del transitions['ob_next']['image']
            transitions['ob_next']['image'] = ''
            transitions.clear()
        return train_info

    def act_log(self, ob):
        return self._actor.act_log(ob)

    def load_env_image_if_needed(self, o):
        img_arr = []
        if 'image' in o.keys() and isinstance(o['image'][0], str):
            for img_path in o['image']:
                img_arr.append(np.load(img_path)[0])
            o['image'] = np.array(img_arr)
        return None

    def _update_network(self, transitions, step=0):
        info = {}

        # pre-process observations
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o, o_next = transitions["ob"], transitions["ob_next"]
        bs = len(transitions["done"])

        if not self._config.parallel_dataloading:        
            self.load_env_image_if_needed(o)
            self.load_env_image_if_needed(o_next)

        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions["ac"])

        if "intra_steps" in transitions.keys() and self._config.use_smdp_update:
            intra_steps = _to_tensor(transitions["intra_steps"])

        done = _to_tensor(transitions["done"]).reshape(bs, 1)
        rew = _to_tensor(transitions["rew"]).reshape(bs, 1)

        actions_real, log_pi = self.act_log(o)
        alpha_loss = -(
            self._log_alpha.exp() * (log_pi + self._target_entropy).detach()
        ).mean()

        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()
        alpha = self._log_alpha.exp()
        info["alpha_loss"] = alpha_loss.cpu().item()
        info["entropy_alpha"] = alpha.cpu().item()
        alpha = self._log_alpha.exp()

        # the actor loss
        entropy_loss = (alpha * log_pi).mean()
        actor_loss = -torch.min(
            self._critic1(o, actions_real), self._critic2(o, actions_real)
        ).mean()
        info["log_pi"] = log_pi.mean().cpu().item()
        info["entropy_loss"] = entropy_loss.cpu().item()
        info["actor_loss"] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        # BC mse loss between expert BC visual policy and the actor
        bc_visual_expert_ac = self.expert_policy.get_predicted_ac(o)
        bc_mse_loss = self.bc_mse_loss(actions_real['default'], bc_visual_expert_ac)
        actor_loss += bc_mse_loss
        info["bc_mse_loss"] = bc_mse_loss.cpu().item()

        # calculate the target Q value function
        with torch.no_grad():
            actions_next, log_pi_next = self.act_log(o_next)
            q_next_value1 = self._critic1_target(o_next, actions_next)
            q_next_value2 = self._critic2_target(o_next, actions_next)
            q_next_value = torch.min(q_next_value1, q_next_value2) - alpha * log_pi_next
            if self._config.use_smdp_update:
                target_q_value = (
                    self._config.reward_scale * rew
                    + (1 - done)
                    * (self._config.discount_factor ** (intra_steps + 1))
                    * q_next_value
                )
            else:
                target_q_value = (
                    self._config.reward_scale * rew
                    + (1 - done) * self._config.discount_factor * q_next_value
                )
            target_q_value = target_q_value.detach()

        # the q loss
        for k, space in self._ac_space.spaces.items():
            if isinstance(space, spaces.Discrete):
                ac[k] = (
                    F.one_hot(ac[k].long(), action_size(self._ac_space[k]))
                    .float()
                    .squeeze(1)
                )
        real_q_value1 = self._critic1(o, ac)
        real_q_value2 = self._critic2(o, ac)
        critic1_loss = 0.5 * (target_q_value - real_q_value1).pow(2).mean()
        critic2_loss = 0.5 * (target_q_value - real_q_value2).pow(2).mean()

        if critic1_loss > 10000:
            print('Critic1 loss > 10000....Exploding loss!!!!!!!!!!!!!')
            exit(1)

        info["min_target_q"] = target_q_value.min().cpu().item()
        info["target_q"] = target_q_value.mean().cpu().item()
        info["min_real1_q"] = real_q_value1.min().cpu().item()
        info["min_real2_q"] = real_q_value2.min().cpu().item()
        info["real1_q"] = real_q_value1.mean().cpu().item()
        info["real2_q"] = real_q_value2.mean().cpu().item()
        info["critic1_loss"] = critic1_loss.cpu().item()
        info["critic2_loss"] = critic2_loss.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        if self._config.is_mpi:
            sync_grads(self._actor)
        self._actor_optim.step()

        # update the critic
        self._critic1_optim.zero_grad()
        critic1_loss.backward()
        if self._config.is_mpi:
            sync_grads(self._critic1)
        self._critic1_optim.step()

        self._critic2_optim.zero_grad()
        critic2_loss.backward()
        if self._config.is_mpi:
            sync_grads(self, _critic2)
        self._critic2_optim.step()

        if self._config.is_mpi:
            return mpi_average(info)
        else:
            return info

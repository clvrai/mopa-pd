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
from rl.policies.utils import BC_Visual_Policy, BC_Visual_Policy_Stochastic
from util.gym import observation_size, action_size, goal_size, box_size, robot_state_size
from collections import OrderedDict
import random

class AACAgent(BaseAgent):
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
        ac_scale=None,
        expert_agent=None,
    ):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space
        self._jnt_indices = jnt_indices
        self._ref_joint_pos_indexes = ref_joint_pos_indexes
        self._ac_scale = ac_scale
        self._log_alpha = torch.tensor(
            np.log(config.alpha), requires_grad=True, device=config.device
        )
        self._alpha_optim = optim.Adam([self._log_alpha], lr=config.lr_actor)
        self._joint_space = joint_space
        self._is_jnt_limited = is_jnt_limited
        if joint_space is not None:
            self._jnt_minimum = joint_space["default"].low
            self._jnt_maximum = joint_space["default"].high

        # build up networks
        self._build_actor(actor)
        self._build_critic(critic)
        self._network_cuda(config.device)

        self._target_entropy = -action_size(self._actor._ac_space)

        if config.expert_mode == 'mopa-sota' or config.expert_mode == 'bc-sota':
            self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_actor, weight_decay=1e-05)
            self._critic1_optim = optim.Adam(
                self._critic1.parameters(), lr=config.lr_critic, weight_decay=1e-05
            )
            self._critic2_optim = optim.Adam(
                self._critic2.parameters(), lr=config.lr_critic, weight_decay=1e-05
            )
            self.bc_mse_loss = nn.MSELoss()
        else:
            self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_actor)
            self._critic1_optim = optim.Adam(
                self._critic1.parameters(), lr=config.lr_critic
            )
            self._critic2_optim = optim.Adam(
                self._critic2.parameters(), lr=config.lr_critic
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
                self._expert_buffer = ReplayBufferIterableDatasetDisk(buffer_keys, config.buffer_size, config.batch_size)
                self._buffer_loader = torch.utils.data.DataLoader(self._buffer, batch_size=1, num_workers=0, collate_fn=collate_transitions)
                self._expert_buffer_loader = torch.utils.data.DataLoader(self._expert_buffer, batch_size=1, num_workers=0, collate_fn=collate_transitions)
            elif config.parallel_dataloading_mode == 'ram':
                self._buffer = ReplayBufferIterableDataset(buffer_keys, config.buffer_size, config.batch_size)
                self._expert_buffer = ReplayBufferIterableDataset(buffer_keys, config.buffer_size, config.batch_size)
                self._buffer_loader = torch.utils.data.DataLoader(self._buffer, batch_size=1, num_workers=0, collate_fn=lambda batch: batch[0])
                self._expert_buffer_loader = torch.utils.data.DataLoader(self._expert_buffer, batch_size=1, num_workers=0, collate_fn=lambda batch: batch[0])
        else:
            self._buffer = ReplayBuffer(
                buffer_keys, config.buffer_size, sampler.sample_func
            )
            self._expert_buffer = ReplayBuffer(
                buffer_keys, config.buffer_size, sampler.sample_func
            )

        self._log_creation()

        self._planner = None
        self._is_planner_initialized = False

        if config.expert_mode == 'bc':
            if config.env == 'PusherObstacle-v0':
                # observation (excluding goal information)
                input_dim = observation_size(ob_space) - goal_size(ob_space) - box_size(ob_space)
            elif 'Sawyer' in config.env:
                input_dim = robot_state_size(ob_space)
            else:
                raise NotImplementedError
            self.expert_policy = BC_Visual_Policy(robot_state=input_dim, num_classes=action_size(ac_space), img_size=config.env_image_size, device=config.device, env=config.env)
            self.expert_policy.load_bc_weights(config.bc_checkpoint)
            print('load pretrained BC weights to BC policy from {}'.format(self._config.bc_checkpoint))
           
            bc_checkpoint = torch.load(config.bc_checkpoint, map_location=config.device)
            self._actor.load_partial_layers(bc_checkpoint['state_dict'])
            print("load partial BC Visual Policy weights to this policy's actor.")
            
            mopa_rl_checkpoint = torch.load(config.mopa_checkpoint, map_location=config.device)
            self._critic1.load_state_dict(mopa_rl_checkpoint['agent']['critic1_state_dict'])
            self._critic2.load_state_dict(mopa_rl_checkpoint['agent']['critic2_state_dict'])
            self._critic1_target.load_state_dict(mopa_rl_checkpoint['agent']['critic1_state_dict'])
            self._critic2_target.load_state_dict(mopa_rl_checkpoint['agent']['critic2_state_dict'])
            print("load pretrained MoPA-RL critic weights to this policy's critic.")
        elif config.expert_mode == 'bc-stochastic':
            self.expert_policy_mopa_rl = expert_agent
            self.mse_loss = nn.MSELoss()

            if config.env == 'PusherObstacle-v0':
                # observation (excluding goal information)
                input_dim = observation_size(ob_space) - goal_size(ob_space) - box_size(ob_space)
            elif 'Sawyer' in config.env:
                input_dim = robot_state_size(ob_space)
            else:
                raise NotImplementedError
           
            bc_checkpoint = torch.load(config.bc_checkpoint, map_location=config.device)
            self._actor.load_state_dict_processed(bc_checkpoint['state_dict'])
            print("load BC Visual Policy Stochastic weights to this policy's actor.")
            
            # temporarily removed: using pre-trained weights for critics
            # if config.policy == 'full-image':
            #     # loading other checkpoint for critics
            #     # critics_checkpoint = torch.load('checkpoints/bc_with_critics_sawyer_push_32px_v3/epoch_135.pth', map_location=config.device)
            #     self._critic1.load_state_dict_processed(bc_checkpoint['critic1_state_dict'])
            #     self._critic2.load_state_dict_processed(bc_checkpoint['critic2_state_dict'])
            #     self._critic1_target.load_state_dict_processed(bc_checkpoint['critic1_state_dict'])
            #     self._critic2_target.load_state_dict_processed(bc_checkpoint['critic2_state_dict'])
            #     print("load BC critics' weights to critics")
            # else:
            #     self._critic1.load_state_dict(expert_agent._critic1.state_dict())
            #     self._critic2.load_state_dict(expert_agent._critic2.state_dict())
            #     self._critic1_target.load_state_dict(expert_agent._critic1.state_dict())
            #     self._critic2_target.load_state_dict(expert_agent._critic2.state_dict())
            #     print("load MoPA-RL' critics weights to critics")
            self._critic1.load_state_dict(expert_agent._critic1.state_dict())
            self._critic2.load_state_dict(expert_agent._critic2.state_dict())
            self._critic1_target.load_state_dict(expert_agent._critic1.state_dict())
            self._critic2_target.load_state_dict(expert_agent._critic2.state_dict())
            print("load MoPA-RL' critics weights to critics")

            self._log_alpha.data = torch.tensor(
                config.log_alpha, requires_grad=True, device=self._config.device
            )
            print("load pretrained MoPA-RL critic weights to this policy's critic.")

            self.expert_policy = BC_Visual_Policy_Stochastic(robot_state=input_dim, num_classes=action_size(ac_space), img_size=config.env_image_size, device=config.device, env=config.env)
            self.expert_policy.load_bc_weights(config.bc_checkpoint)
            print('load pretrained BC weights to BC policy from {}'.format(self._config.bc_checkpoint))
        elif config.expert_mode == 'bc-stochastic-randweights':
            self.expert_policy = None
        elif config.expert_mode == 'mopa' or config.expert_mode == 'bc-stochastic-mopa':
            self.expert_policy = expert_agent._actor

            # self._actor.load_partial_layers(expert_agent._actor.state_dict())
            # print("load partial MoPA-RL actor weights to this policy's actor.")            
            
            self._critic1.load_state_dict(expert_agent._critic1.state_dict())
            self._critic2.load_state_dict(expert_agent._critic2.state_dict())
            self._critic1_target.load_state_dict(expert_agent._critic1_target.state_dict())
            self._critic2_target.load_state_dict(expert_agent._critic2_target.state_dict())
            print("load pretrained MoPA-RL critic weights to this policy's critic.")
        elif config.expert_mode == 'mopa-sota':
            self.expert_policy = expert_agent._actor
        elif config.expert_mode == 'bc-sota':
            if config.env == 'PusherObstacle-v0':
                # observation (excluding goal information)
                input_dim = observation_size(ob_space) - goal_size(ob_space) - box_size(ob_space)
            elif 'Sawyer' in config.env:
                input_dim = robot_state_size(ob_space)
            else:
                raise NotImplementedError
            self.expert_policy = BC_Visual_Policy(robot_state=input_dim, num_classes=action_size(ac_space), img_size=config.env_image_size, device=config.device, env=config.env)
            self.expert_policy.load_bc_weights(config.bc_checkpoint)
            print('load pretrained BC weights to BC policy from {}'.format(self._config.bc_checkpoint))            
        else:
            raise NotImplementedError
        
        if self.expert_policy is not None:
            self.expert_policy.eval()
            self.expert_policy.cuda()
        
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

    def policy_replay_buffer(self):
        return self._buffer.state_dict()
    
    def expert_policy_replay_buffer(self):
        return self._expert_buffer.state_dict()

    def load_replay_buffers(self, policy_state_dict, expert_state_dict):
        self._buffer.load_state_dict(policy_state_dict)
        self._expert_buffer.load_state_dict(expert_state_dict)

    def _build_actor(self, actor):
        self._actor = actor(
            self._config,
            self._ob_space,
            self._ac_space,
            self._config.tanh_policy,
            self._ac_scale,
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

    def freeze_actor_layers(self):
        for param in self._actor.parameters():
            param.requires_grad = False
        return True

    def unfreeze_actor_layers(self):
        for param in self._actor.parameters():
            param.requires_grad = True
        return True

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts)
    
    def store_episode_expert(self, rollouts):
        self._expert_buffer.store_episode(rollouts)

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
    
    def concat_transitions_helper(self, t1, t2, batch_size_expert, batch_size_policy):
        if isinstance(t1, dict):
            sub_out = dict()
            for k, v in t1.items():
                sub_out[k] = self.concat_transitions_helper(t1[k], t2[k], batch_size_expert, batch_size_policy)
            return sub_out
        elif type(t1) is np.ndarray:
            if batch_size_expert is not None and batch_size_policy is not None:
                return np.concatenate((t1[:batch_size_expert], t2[:batch_size_policy]), axis=0)
            else:
                return np.concatenate((t1, t2), axis=0) 
        else:
            raise NotImplementedError

    def concat_transitions(self, t1, t2, batch_size_expert=None, batch_size_policy=None):
        out = dict()
        # dict_keys(['ob', 'ac', 'meta_ac', 'done', 'rew', 'intra_steps', 'ob_next'])
        for k, v in t1.items():
            out[k] = self.concat_transitions_helper(t1[k], t2[k], batch_size_expert, batch_size_policy)
        return out

    def train(self):
        # config.percent_expert_batch_size from expert trajectories and [1-config.percent_expert_batch_size] from policy
        batch_size_expert = int(self._config.batch_size * self._config.percent_expert_batch_size)
        batch_size_policy = self._config.batch_size - batch_size_expert
        for i in range(self._config.num_batches):
            if self._config.parallel_dataloading:
                transitions_expert = next(iter(self._expert_buffer_loader))
                transitions_policy = next(iter(self._buffer_loader))
                transitions = self.concat_transitions(transitions_expert, transitions_policy, batch_size_expert=batch_size_expert, batch_size_policy=batch_size_policy)
            else:
                transitions_expert = self._expert_buffer.sample(batch_size_expert)
                transitions_policy = self._buffer.sample(batch_size_policy)
                transitions = self.concat_transitions(transitions_expert, transitions_policy)

            train_info = self._update_network(transitions, i)
            self._soft_update_target_network(
                self._critic1_target, self._critic1, self._config.polyak
            )
            self._soft_update_target_network(
                self._critic2_target, self._critic2, self._config.polyak
            )
            transitions_expert.clear()
            transitions_policy.clear()
            transitions.clear()
        return train_info

    def train_expert(self):        
        for i in range(self._config.num_batches):
            if self._config.parallel_dataloading:
                transitions = next(iter(self._expert_buffer_loader))
            else:
                transitions = self._expert_buffer.sample(self._config.batch_size)
            train_info = self._update_network(transitions, i)
            self._soft_update_target_network(
                self._critic1_target, self._critic1, self._config.polyak
            )
            self._soft_update_target_network(
                self._critic2_target, self._critic2, self._config.polyak
            )
            transitions.clear()
        return train_info

    def train_policy(self):        
        for i in range(self._config.num_batches):
            if self._config.parallel_dataloading:
                transitions = next(iter(self._buffer_loader))
            else:
                transitions = self._buffer.sample(self._config.batch_size)

            if self._config.random_crop:
                transitions['ob']['image'] = self.random_crop_and_pad(transitions['ob']['image'], 24)

            train_info = self._update_network(transitions, i)
            self._soft_update_target_network(
                self._critic1_target, self._critic1, self._config.polyak
            )
            self._soft_update_target_network(
                self._critic2_target, self._critic2, self._config.polyak
            )
            transitions.clear()
        return train_info

    def pretrain_policy(self):        
        for i in range(self._config.num_batches):
            if self._config.parallel_dataloading:
                transitions = next(iter(self._buffer_loader))
            else:
                transitions = self._buffer.sample(self._config.batch_size)
            train_info = self._update_critics_supervised(transitions, i)
            self._soft_update_target_network(
                self._critic1_target, self._critic1, self._config.polyak
            )
            self._soft_update_target_network(
                self._critic2_target, self._critic2, self._config.polyak
            )
            transitions.clear()
        return train_info

    def random_crop_and_pad(self, imgs, crop=84):
        """
            source: https://github.com/MishaLaskin/rad/blob/master/data_augs.py
            args:
            imgs: np.array shape (B,1, C,H,W)
            crop: output size (e.g. 84)
            returns np.array
        """
        prob = random.uniform(0, 1)
        if prob < 0.5:
            n, extra, c, h, w = imgs.shape
            crop_max = h - crop + 1
            w1 = np.random.randint(0, crop_max, n)
            h1 = np.random.randint(0, crop_max, n)
            cropped = np.zeros((n, 1, c, h, w), dtype=imgs.dtype)
            for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
                cropped[i, 0, :, h11:h11 + crop, w11:w11 + crop] = img[0, :, h11:h11 + crop, w11:w11 + crop]
            return cropped
        return imgs

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

        if self._config.expert_mode == 'mopa-sota' or self._config.expert_mode == 'bc-sota':
            # BC mse loss between expert BC visual policy and the actor
            bc_visual_expert_ac = self.expert_policy.get_predicted_ac(o)
            bc_mse_loss = self.bc_mse_loss(actions_real['default'], bc_visual_expert_ac)
            actor_loss += bc_mse_loss
            info["bc_mse_loss"] = bc_mse_loss.cpu().item()
            info['policy_predicted_actions_mean'] = actions_real['default'].mean().cpu().item()
            info['expert_predicted_actions_mean'] = bc_visual_expert_ac.mean().cpu().item()

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
            sync_grads(self._critic2)
        self._critic2_optim.step()

        if self._config.is_mpi:
            return mpi_average(info)
        else:
            return info

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

    def _update_critics_supervised(self, transitions, step=0):
        info = {}

        # pre-process observations
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = transitions["ob"]

        if not self._config.parallel_dataloading:        
            self.load_env_image_if_needed(o)

        o = _to_tensor(o)
        o_no_img = o.copy()
        o_no_img = self.process_data_for_mopa_rl(o_no_img)

        actions_real, log_pi = self.act_log(o)
        alpha_loss = -(
            self._log_alpha.exp() * (log_pi + self._target_entropy).detach()
        )
        alpha_loss_mean = alpha_loss.mean()

        actions_real_expert, log_pi_expert = self.expert_policy_mopa_rl.act_log(o_no_img)
        alpha_loss_expert = -(
            self.expert_policy_mopa_rl._log_alpha.exp() * (log_pi_expert + self.expert_policy_mopa_rl._target_entropy).detach()
        )
        alpha_mse_loss = self.mse_loss(alpha_loss, alpha_loss_expert)

        self._alpha_optim.zero_grad()
        alpha_mse_loss.backward()
        self._alpha_optim.step()
        alpha = self._log_alpha.exp()
        info["alpha_mse_loss"] = alpha_mse_loss.cpu().item()
        info["alpha_loss"] = alpha_loss_mean.cpu().item()
        info["entropy_alpha"] = alpha.cpu().item()

        if self._config.is_mpi:
            return mpi_average(info)
        else:
            return info
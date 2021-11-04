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


class AAC_DDPG_Agent:
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
    ):
        self._config = config
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._jnt_indices = jnt_indices
        self._ref_joint_pos_indexes = ref_joint_pos_indexes
        self._joint_space = joint_space
        self._is_jnt_limited = is_jnt_limited
        self._ac_scale = ac_scale
        if joint_space is not None:
            self._jnt_minimum = joint_space["default"].low
            self._jnt_maximum = joint_space["default"].high

        self.bc_mse_loss = nn.MSELoss()
        self.q_mse_loss = nn.MSELoss()

        # build up networks
        self._build_actor(actor)
        self._build_critic(critic)
        self._network_cuda(config.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self._actor_target.parameters():
            p.requires_grad = False
        for p in self._critic_target.parameters():
            p.requires_grad = False

        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_actor)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=config.lr_critic)

        # parameters specific to DDPG
        self.gamma = config.gamma

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
            sampler = RandomSampler()
            self._buffer = ReplayBuffer(
                buffer_keys, config.buffer_size, sampler.sample_func
            )
            self._expert_buffer = ReplayBuffer(
                buffer_keys, config.buffer_size, sampler.sample_func
            )

        self._log_creation()

        self._planner = None
        self._is_planner_initialized = False

        # create and load expert BC Visual policy
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
            logger.info("creating a AAC-DDPG agent")
            logger.info("the actor has %d parameters", count_parameters(self._actor))
            logger.info("the critic has %d parameters", count_parameters(self._critic))

    def _build_actor(self, actor):
        self._actor = actor(self._config, self._ob_space, self._ac_space, self._config.tanh_policy, self._ac_scale)
        # build up target network
        self._actor_target = actor(self._config, self._ob_space, self._ac_space, self._config.tanh_policy, self._ac_scale)
        self._actor_target.load_state_dict(self._actor.state_dict())

    def _build_critic(self, critic):
        self._critic = critic(self._config, self._ob_space, self._ac_space)
        # build up target networks
        self._critic_target = critic(self._config, self._ob_space, self._ac_space)
        self._critic_target.load_state_dict(self._critic.state_dict())

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
            "actor_state_dict": self._actor.state_dict(),
            "critic_state_dict": self._critic.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
            "critic_optim_state_dict": self._critic_optim.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._critic.load_state_dict(ckpt["critic_state_dict"])

        self._actor_target.load_state_dict(self._actor.state_dict())
        self._critic_target.load_state_dict(self._critic.state_dict())

        self._network_cuda(self._config.device)

        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self._critic_optim.load_state_dict(ckpt["critic_optim_state_dict"])

        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._critic.to(device)
        self._actor_target.to(device)
        self._critic_target.to(device)
    
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
            transitions.clear()
        return train_info

    def act(self, ob, is_train=True, return_stds=False, random_exploration=False, collect_expert_trajectories=False):
        if collect_expert_trajectories:
            ac = self.expert_policy.act_expert(ob)
            return ac, None, None
     
        if is_train:
            ac = self._actor.act(ob, self._config.act_noise)
        else:
            ac = self._actor.act(ob)

        return ac, None, None

    def load_env_image_if_needed(self, o):
        img_arr = []
        if 'image' in o.keys() and isinstance(o['image'][0], str):
            for img_path in o['image']:
                img_arr.append(np.load(img_path)[0])
            o['image'] = np.array(img_arr)
        return None

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, o, o_next, ac, done, rew, info):
        q = self._critic(o, ac)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self._critic_target(o_next, self._actor_target(o_next))
            backup = rew + self.gamma * (1 - done) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = self.q_mse_loss(q, backup)

        info['sum_rewards'] = torch.sum(rew).cpu().item()
        info['loss_q'] =  loss_q.cpu().item()

        return loss_q

    def compute_loss_pi(self, o):
        q_pi = self._critic(o, self._actor(o))
        return -q_pi.mean()

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
        done = _to_tensor(transitions["done"]).reshape(bs, 1)
        rew = _to_tensor(transitions["rew"]).reshape(bs, 1)

        # First run one gradient descent step for Q.
        self._critic_optim.zero_grad()
        # calculate q loss
        loss_q = self.compute_loss_q(o, o_next, ac, done, rew, info)
        loss_q.backward()
        # torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 1) # clip gradient
        self._critic_optim.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self._critic.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self._actor_optim.zero_grad()
        loss_pi = self.compute_loss_pi(o)

        # BC mse loss between expert BC visual policy and the actor
        bc_visual_expert_ac = self.expert_policy.get_predicted_ac(o)
        actor_prediced_ac = self._actor(o)
        bc_mse_loss = self.bc_mse_loss(actor_prediced_ac, bc_visual_expert_ac)
        info["loss_pi"] = loss_pi.cpu().item()
        info["bc_mse_loss"] = bc_mse_loss.cpu().item()
        loss_pi_bc = loss_pi + bc_mse_loss
        loss_pi_bc.backward()
        # torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 1) # clip gradient
        self._actor_optim.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self._critic.parameters():
            p.requires_grad = True        

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self._actor.parameters(), self._actor_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self._config.polyak)
                p_targ.data.add_((1 - self._config.polyak) * p.data)
            
            for p, p_targ in zip(self._critic.parameters(), self._critic_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self._config.polyak)
                p_targ.data.add_((1 - self._config.polyak) * p.data)

        return info

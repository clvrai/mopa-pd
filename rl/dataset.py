from collections import defaultdict
from time import time

import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset


class ReplayBufferIterableDatasetDisk(IterableDataset):
    '''
    This implementation fixes the memory leak issue with ReplayBufferIterableDataset;
    however, the processing speed has reduced significantly from 10it/s to 5.3it/s.
    Direct image loading approach is around 5.3it/s.
    '''
    def __init__(self, keys, buffer_size, batch_size):
        self._size = buffer_size
        self._batch_size = batch_size

        # memory management
        self._idx = 0
        self._current_size = 0

        # create the buffer to store info
        self._keys = keys
        self._buffers = defaultdict(list)

    # store the episode
    def store_episode(self, rollout):
        idx = self._idx = (self._idx + 1) % self._size
        self._current_size += 1

        if self._current_size > self._size:
            for k in self._keys:
                self._buffers[k][idx] = rollout[k]
        else:
            for k in self._keys:
                self._buffers[k].append(rollout[k])

    def state_dict(self):
        return self._buffers
    
    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        self._current_size = len(self._buffers["ac"])

    def _sample_func(self, start_index, end_index):
        episode_batch = self._buffers
        episode_idxs = torch.randint(start_index, end_index, (self._batch_size,))
        t_samples = [
            torch.randint(0, len(episode_batch["ac"][episode_idx]), (1,)).numpy()[0]
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [
                episode_batch[key][episode_idx][t]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        if "ob_next" not in episode_batch.keys():
            transitions["ob_next"] = [
                episode_batch["ob"][episode_idx][t + 1]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        ob_images = []
        ob_next_images = []
        for i in range(self._batch_size):
            ob_images.append(np.load(transitions['ob'][i]['image'])[0])
            ob_next_images.append(np.load(transitions['ob_next'][i]['image'])[0])
        ob_images = np.array(ob_images)
        ob_next_images = np.array(ob_next_images)
        transitions.clear()
        return new_transitions, ob_images, ob_next_images

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: 
            iter_start = 0
            iter_end = len(self._buffers["ac"])
        else:
            per_worker = int(len(self._buffers["ac"]) / float(worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self._buffers["ac"]))
        yield self._sample_func(iter_start, iter_end)


class ReplayBufferIterableDataset(IterableDataset):
    '''
    This implementation uses IterableDataset. Although this implementation was designed to
    save memory by only loading a batch of images in RAM, the following code has memory leak
    issue (loaded images in RAM are not freed): 

    'transitions[key][i]['image'] = np.load(transitions[key][i]['image'])[0]'

    This problem is actually not well documented on the Internet, but there is a person who
    has the same issue when he assigns np.array to a dictionary in the Dataloader:
    https://discuss.pytorch.org/t/memory-leak-in-dataloader/66617.

    del, gc.collect, .clear() have all been tried but to no effect.

    Under the same environment and machine, this implementation can process around 10it/s.
    '''
    def __init__(self, keys, buffer_size, batch_size):
        self._size = buffer_size
        self._batch_size = batch_size

        # memory management
        self._idx = 0
        self._current_size = 0

        # create the buffer to store info
        self._keys = keys
        self._buffers = defaultdict(list)

    # store the episode
    def store_episode(self, rollout):
        idx = self._idx = (self._idx + 1) % self._size
        self._current_size += 1

        if self._current_size > self._size:
            for k in self._keys:
                self._buffers[k][idx] = rollout[k]
        else:
            for k in self._keys:
                self._buffers[k].append(rollout[k])

    def state_dict(self):
        return self._buffers
    
    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        self._current_size = len(self._buffers["ac"])

    def _sample_func(self, start_index, end_index):
        episode_batch = self._buffers
        episode_idxs = torch.randint(start_index, end_index, (self._batch_size,))
        t_samples = [
            torch.randint(0, len(episode_batch["ac"][episode_idx]), (1,)).numpy()[0]
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [
                episode_batch[key][episode_idx][t]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]
            if key == 'ob':
                for i in range(len(transitions[key])):
                    if isinstance(transitions[key][i]['image'], str):
                        transitions[key][i]['image'] = np.load(transitions[key][i]['image'])[0]

        if "ob_next" not in episode_batch.keys():
            transitions["ob_next"] = [
                episode_batch["ob"][episode_idx][t + 1]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]
            for i in range(len(transitions["ob_next"])):
                if isinstance(transitions["ob_next"][i]['image'], str):
                    transitions["ob_next"][i]['image'] = np.load(transitions["ob_next"][i]['image'])[0]

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        transitions.clear()
        return new_transitions

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: 
            iter_start = 0
            iter_end = len(self._buffers["ac"])
        else:
            per_worker = int(len(self._buffers["ac"]) / float(worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self._buffers["ac"]))
        yield self._sample_func(iter_start, iter_end)


class ReplayBuffer:
    def __init__(self, keys, buffer_size, sample_func):
        self._size = buffer_size

        # memory management
        self._idx = 0
        self._current_size = 0
        self._sample_func = sample_func

        # create the buffer to store info
        self._keys = keys
        self._buffers = defaultdict(list)

    def clear(self):
        self._idx = 0
        self._current_size = 0
        self._buffers = defaultdict(list)

    # store the episode
    def store_episode(self, rollout):
        idx = self._idx = (self._idx + 1) % self._size
        self._current_size += 1

        if self._current_size > self._size:
            for k in self._keys:
                self._buffers[k][idx] = rollout[k]
        else:
            for k in self._keys:
                self._buffers[k].append(rollout[k])

    # sample the data from the replay buffer
    def sample(self, batch_size):
        # sample transitions
        transitions = self._sample_func(self._buffers, batch_size)
        return transitions

    def state_dict(self):
        return self._buffers

    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        self._current_size = len(self._buffers["ac"])


class RandomSampler:
    def sample_func(self, episode_batch, batch_size_in_transitions):
        rollout_batch_size = len(episode_batch["ac"])
        batch_size = batch_size_in_transitions
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = [
            np.random.randint(len(episode_batch["ac"][episode_idx]))
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [
                episode_batch[key][episode_idx][t]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        if "ob_next" not in episode_batch.keys():
            transitions["ob_next"] = [
                episode_batch["ob"][episode_idx][t + 1]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        return new_transitions


class HERSampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        if self.replay_strategy == "future":
            self.future_p = 1 - (1.0 / 1 + replay_k)
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_func(self, episode_batch, batch_size_in_transitions):
        rollout_batch_size = len(episode_batch["ac"])
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = [
            np.random.randint(len(episode_batch["ac"][episode_idx]))
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [
                episode_batch[key][episode_idx][t]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        transitions["ob_next"] = [
            episode_batch["ob"][episode_idx][t + 1]
            for episode_idx, t in zip(episode_idxs, t_samples)
        ]
        transitions["r"] = np.zeros((batch_size,))

        # hindsight experience replay
        for i, (episode_idx, t) in enumerate(zip(episode_idxs, t_samples)):
            replace_goal = np.random.uniform() < self.future_p
            if replace_goal:
                future_t = np.random.randint(
                    t + 1, len(episode_batch["ac"][episode_idx]) + 1
                )
                future_ag = episode_batch["ag"][episode_idx][future_t]
                if (
                    self.reward_func(
                        episode_batch["ag"][episode_idx][t], future_ag, None
                    )
                    < 0
                ):
                    transitions["g"][i] = future_ag

            transitions["r"][i] = self.reward_func(
                episode_batch["ag"][episode_idx][t + 1], transitions["g"][i], None
            )

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        return new_transitions

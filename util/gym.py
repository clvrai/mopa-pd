import numpy as np
from gym import spaces
import cv2


def observation_size(observation_space):
    if isinstance(observation_space, spaces.Dict):
        return sum(
            [observation_size(value) for key, value in observation_space.spaces.items()]
        )
    elif isinstance(observation_space, spaces.Box):
        return np.product(observation_space.shape)


def action_size(action_space):
    if isinstance(action_space, spaces.Dict):
        return sum([action_size(value) for key, value in action_space.spaces.items()])
    elif isinstance(action_space, spaces.Box):
        return np.product(action_space.shape)
    elif isinstance(action_space, spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, spaces.MultiDiscrete):
        return np.product(action_space.nvec)
    elif isinstance(action_space, spaces.MultiBinary):
        return action_space.n

def goal_size(observation_space):
    if isinstance(observation_space, spaces.Dict):
        return observation_space['goal'].shape[0]
    elif isinstance(observation_space, spaces.Box):
        raise NotImplementedError

def box_size(observation_space):
    if isinstance(observation_space, spaces.Dict):
        return observation_space['box'].shape[0]
    elif isinstance(observation_space, spaces.Box):
        raise NotImplementedError

def robot_state_size(observation_space):
    if isinstance(observation_space, spaces.Dict):
        return observation_space['joint_pos'].shape[0] + \
                observation_space['joint_vel'].shape[0] + \
                observation_space['gripper_qpos'].shape[0] + \
                observation_space['gripper_qvel'].shape[0] + \
                observation_space['eef_pos'].shape[0] + \
                observation_space['eef_quat'].shape[0]
    elif isinstance(observation_space, spaces.Box):
        raise NotImplementedError

def image_size(observation_space):
    if isinstance(observation_space, spaces.Dict):
        return np.prod(observation_space['image'].shape)
    elif isinstance(observation_space, spaces.Box):
        raise NotImplementedError
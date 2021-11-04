from collections import OrderedDict

import numpy as np
import cv2
import gym
import torch

from env.inverse_kinematics import qpos_from_site_pose_sampling, qpos_from_site_pose
from util.logger import logger
from util.env import joint_convert, mat2quat, quat_mul, quat_inv
from util.info import Info
from rl.rollouts import Rollout
import random


class BCRLRolloutRunner(object):
    def __init__(self, config, env, env_eval, pi):
        random.seed(config.seed)
        print('Seed number ', config.seed)
        self._config = config
        self._env = env
        self._env_eval = env_eval
        self._pi = pi

    def gaussian_noise(self, intended_action, cov):
        action_np = intended_action["default"]
        action_sampled = np.random.multivariate_normal(action_np, cov)
        intended_action["default"] = action_sampled
        return intended_action

    def run(
        self,
        max_step=10000,
        is_train=True,
        random_exploration=False,
        every_steps=None,
        every_episodes=None,
        collect_expert_trajectories=False
    ):
        """
        Collects trajectories and yield every @every_steps/@every_episodes.
        Args:
            is_train: whether rollout is for training or evaluation.
            every_steps: if not None, returns rollouts @every_steps
            every_episodes: if not None, returns rollouts @every_epiosdes
        """
        if every_steps is None and every_episodes is None:
            raise ValueError("Both every_steps and every_episodes cannot be None")
        config = self._config
        device = config.device
        env = self._env if is_train else self._env_eval
        pi = self._pi

        rollout = Rollout()
        reward_info = Info()
        ep_info = Info()

        step = 0
        episode = 0
        while True:
            done = False
            ep_len = 0
            ep_rew = 0
            ep_discounted_rew = 0
            mp_path_len = 0
            interpolation_path_len = 0
            ob = env.reset()

            # run rollout
            meta_ac = None
            counter = {
                "mp": 0,
                "rl": 0,
                "interpolation": 0,
                "mp_fail": 0,
                "approximate": 0,
                "invalid": 0,
            }
            while not done and ep_len < max_step:
                env_step = 0

                ll_ob = ob.copy()
                ac, ac_before_activation, stds = pi.act(
                    ll_ob,
                    is_train=is_train,
                    return_stds=True,
                    random_exploration=random_exploration,
                    collect_expert_trajectories=collect_expert_trajectories,
                )

                if torch.is_tensor(ac):
                    ac = OrderedDict([("default", ac.cpu().numpy())])
                elif isinstance(ac, np.ndarray):
                    ac = OrderedDict([("default", ac)])

                curr_qpos = env.sim.data.qpos.copy()
                prev_qpos = env.sim.data.qpos.copy()
                target_qpos = curr_qpos.copy()
                prev_ob = ob.copy()
                ll_ob = ob.copy()
                rollout.add(
                    {
                        "ob": ll_ob,
                        "meta_ac": meta_ac,
                        "ac": ac,
                        "ac_before_activation": ac_before_activation,
                    }
                )
                counter["rl"] += 1

                ob, reward, done, info = env.step(ac, is_mopa_rl=False)

                rollout.add({"done": done, "rew": reward, "intra_steps": 0})
                ep_len += 1
                step += 1
                ep_rew += reward
                ep_discounted_rew += pow(config.gamma, (ep_len-1)) * reward
                env_step += 1
                reward_info.add(info)
                if every_steps is not None and step % every_steps == 0:
                    # last frame
                    ll_ob = ob.copy()
                    rollout.add({"ob": ll_ob, "meta_ac": meta_ac})
                    ep_info.add({"env_step": env_step})
                    env_step = 0
                    yield rollout.get(), ep_info.get_dict(only_scalar=True)
                env.reset_prev_state()
            ep_info.add({"len": ep_len, "rew": ep_rew, "rew_discounted": ep_discounted_rew})
            ep_info.add(counter)
            reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
            ep_info.add(reward_info_dict)
            logger.info(
                "Ep %d rollout: %s %s",
                episode,
                {
                    k: v
                    for k, v in reward_info_dict.items()
                    if not "qpos" in k and np.isscalar(v)
                },
                {k: v for k, v in counter.items()},
            )
            episode += 1

    def run_episode(
        self, max_step=10000, is_train=True, record=False, random_exploration=False
    ):
        config = self._config
        device = config.device
        env = self._env if is_train else self._env_eval
        pi = self._pi

        rollout = Rollout()
        reward_info = Info()
        ep_info = Info()

        done = False
        ep_len = 0
        ep_rew = 0
        ep_discounted_rew = 0
        ob = env.reset()
        self._record_frames = []
        if record:
            self._store_frame(env)

        # buffer to save qpos
        saved_qpos = []

        if config.stochastic_eval and not is_train:
            is_train = True

        stochastic = is_train or not config.stochastic_eval
        # run rollout
        meta_ac = None
        total_contact_force = 0.0
        counter = {
            "mp": 0,
            "rl": 0,
            "interpolation": 0,
            "mp_fail": 0,
            "approximate": 0,
            "invalid": 0,
        }
        action_dim = env.action_space['default'].shape[0]
        cov = np.random.normal(0, 0.05, (action_dim, action_dim))
        cov = cov.T.dot(cov)
        cov = cov / np.trace(cov)

        while not done and ep_len < max_step:
            p_noise = random.uniform(0, 1)
            ll_ob = ob.copy()
            ac, ac_before_activation, stds = pi.act(
                ll_ob,
                is_train=is_train,
                return_stds=True,
                random_exploration=random_exploration,
            )

            if(config.eval_noise==True and p_noise <= 0.01):
                ac = self.gaussian_noise(ac, cov)
                cov = np.random.normal(0, 0.05, (action_dim, action_dim))
                cov = cov.T.dot(cov)
                cov = cov / np.trace(cov)

            curr_qpos = env.sim.data.qpos.copy()
            prev_qpos = env.sim.data.qpos.copy()
            prev_joint_qpos = curr_qpos[env.ref_joint_pos_indexes]
            target_qpos = env.sim.data.qpos.copy()

            ll_ob = ob.copy()
            ll_img = env.get_env_image()
            counter["rl"] += 1

            ob, reward, done, info = env.step(ac, is_mopa_rl=False)
            contact_force = env.get_contact_force()
            total_contact_force += contact_force

            ep_len += 1
            ep_rew += reward
            ep_discounted_rew += pow(config.gamma, (ep_len-1)) * reward
            reward_info.add(info)
            rollout.add(
                {
                    "ob": ll_ob,
                    "img": ll_img,
                    "meta_ac": meta_ac,
                    "ac": ac,
                    "ac_before_activation": None,
                }
            )

            rollout.add(
                {
                    "done": done,
                    "rew": reward,
                }
            )
            if record:
                frame_info = info.copy()
                frame_info["ac"] = ac["default"]
                frame_info["contact_force"] = contact_force
                frame_info["std"] = np.array(stds["default"].detach().cpu())[0]
                env.reset_visualized_indicator()
                self._store_frame(env, frame_info)
            env.reset_prev_state()
            env.reset_visualized_indicator()

            # last frame
            ll_ob = ob.copy()
            ll_img = env.get_env_image()
            rollout.add({"ob": ll_ob, "img": ll_img,"meta_ac": meta_ac})
        ep_info.add(
            {
                "len": ep_len,
                "rew": ep_rew,
                "rew_discounted": ep_discounted_rew,
                "contact_force": total_contact_force,
                "avg_conntact_force": total_contact_force / ep_len,
            }
        )
        ep_info.add(counter)
        reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
        ep_info.add(reward_info_dict)
        # last frame
        return rollout.get(), ep_info.get_dict(only_scalar=True), self._record_frames

    def _cart2dispalcement(self, env, ik_env, ac, curr_qpos, target_cart):
        if len(env.min_world_size) == 2:
            target_cart = np.concatenate(
                (
                    target_cart,
                    np.array([env.sim.data.get_site_xpos(config.ik_target)[2]]),
                )
            )
        if "quat" in ac.keys():
            target_quat = mat2quat(env.sim.data.get_site_xmat(self._config.ik_target))
            target_quat = target_quat[[3, 0, 1, 1]]
            target_quat = quat_mul(
                target_quat,
                (ac["quat"] / np.linalg.norm(ac["quat"])).astype(np.float64),
            )
        else:
            target_quat = None
        ik_env.set_state(curr_qpos.copy(), env.data.qvel.copy())
        result = qpos_from_site_pose(
            ik_env,
            self._config.ik_target,
            target_pos=target_cart,
            target_quat=target_quat,
            joint_names=env.robot_joints,
            max_steps=100,
            tol=1e-2,
        )
        target_qpos = env.sim.data.qpos.copy()
        target_qpos[env.ref_joint_pos_indexes] = result.qpos[
            env.ref_joint_pos_indexes
        ].copy()
        target_qpos = np.clip(
            target_qpos,
            env._jnt_minimum[env.jnt_indices],
            env._jnt_maximum[env.jnt_indices],
        )
        displacement = OrderedDict(
            [
                (
                    "default",
                    target_qpos[env.ref_joint_pos_indexes]
                    - curr_qpos[env.ref_joint_pos_indexes],
                )
            ]
        )
        return displacement

    def _store_frame(self, env, info={}, planner=False):
        color = (200, 200, 200)

        text = "{:4} {}".format(env.episode_length, env.episode_reward)

        geom_colors = {}

        frame = env.render("rgb_array", is_eval=True) * 255.0

        if self._config.vis_info:
            if planner:
                for geom_idx, color in geom_colors.items():
                    env.sim.model.geom_rgba[geom_idx] = color

            fheight, fwidth = frame.shape[:2]
            frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

            if self._config.record_caption:
                font_size = 0.4
                thickness = 1
                offset = 12
                x, y = 5, fheight + 10
                cv2.putText(
                    frame,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (255, 255, 0),
                    thickness,
                    cv2.LINE_AA,
                )
                for i, k in enumerate(info.keys()):
                    v = info[k]
                    key_text = "{}: ".format(k)
                    (key_width, _), _ = cv2.getTextSize(
                        key_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness
                    )

                    cv2.putText(
                        frame,
                        key_text,
                        (x, y + offset * (i + 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_size,
                        (66, 133, 244),
                        thickness,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        frame,
                        str(v),
                        (x + key_width, y + offset * (i + 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_size,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )

        self._record_frames.append(frame)

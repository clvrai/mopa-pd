from collections import OrderedDict
import json
import os

from copy import deepcopy
import wandb
import gym
from gym.envs.mujoco import mujoco_env
import numpy as np
import mujoco_py
from mujoco_py.modder import CameraModder, LightModder, TextureModder

import env
from env.sawyer.sawyer import SawyerEnv
from env.sawyer.sawyer_push_obstacle import SawyerPushObstacleEnv
from env.sawyer.sawyer_lift_obstacle import SawyerLiftObstacleEnv
from env.sawyer.sawyer_assembly_obstacle import SawyerAssemblyObstacleEnv

from env.sawyer.domain_randomization import (
    sample,
    sample_light_dir,
    look_at,
    jitter_angle,
    ImgTextureModder,
)

class SettableStateWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        from_pixels=False,
        height=100,
        width=100,
        camera_id=None,
        channels_first=True,
    ):
        super().__init__(env)
        self.env = env
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._channels_first = channels_first

    ### Make state settable for testing image/action translations ###

    def set_state_from_ob (self, qpos, qvel, goal=None, reset=False):
        # InvertedPendulum

        if goal is not None:
            self.set_state(qpos, qvel, goal=goal)
        else:
            self.set_state(qpos, qvel)

        if self._from_pixels:
            ob = self.render(
                mode="rgb_array",
                height=self._height,
                width=self._width,
                camera_id=self._camera_id,
            )
            if reset:
                ob = self.render(
                    mode="rgb_array",
                    height=self._height,
                    width=self._width,
                    camera_id=self._camera_id,
                )
            if self._channels_first:
                ob = ob.transpose(2, 0, 1).copy()
        else:
            raise NotImplementedError
        return ob

    def step_from_state(self, qpos, qvel, action):
        self.set_state(qpos, qvel)
        self.env.step(action)
        return self.env.sim.data.qpos.copy(), self.env.sim.data.qvel.copy()

class DRWrapper(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.env = env
        self._config = config
        self.sim = env.sim
        self.model = self.sim.model
        with open("./env/sawyer/dr_config.json") as f:
            self._params = json.load(f)[config.dr_params_set]
        
        # init for each modder
        if self._params["tex_mod"]["active"]:
            self.tex_modder = ImgTextureModder(self.sim, modes=self._params["tex_mod"]["modes"])
            if self.model.mat_rgba is not None:
                self.tex_modder.whiten_materials()

        if self._params["camera_mod"]["active"]:
            self.cam_modder = CameraModder(self.sim)
            if self._params["camera_mod"]["use_default"]:
                self.unwrapped._get_viewer(mode="rgb_array")
                self.original_lookat = self.viewer.cam.lookat.copy()
            else:
                self.original_pos = self.cam_modder.get_pos(self._params["camera_mod"]['cam_name']).copy()
                self.original_quat = self.cam_modder.get_quat(self._params["camera_mod"]['cam_name']).copy()
                self._cam_name = self._params["camera_mod"]['cam_name']

        if self._params["light_mod"]["active"]:
            self.light_modder = LightModder(self.sim)

        if self._params["dynamics_mod"]['active']:
            dynamics_params = self._params["dynamics_mod"]
            if "body_mass_multiplier" in dynamics_params:
                self._original_intertia = self.model.body_inertia.copy()
                self._original_mass = self.model.body_mass.copy()
            if "armature_multiplier" in dynamics_params:
                self._original_armature = self.model.dof_armature.copy()
            if "mass_targets" in dynamics_params:
                self.inertia_mass_one = np.zeros_like(self.model.body_inertia)
                for body, param in dynamics_params["mass_targets"].items():
                    body_id = self.model.body_name2id(body)
                    # body inertia when mass is 1
                    self.inertia_mass_one[body_id,:] = self.model.body_inertia[body_id] /  self.model.body_mass[body_id]

    def _rand_textures(self):
        """Randomize all the textures in the scene, including the skybox"""
        for name in self._params["tex_mod"]["geoms"]:
            self.tex_modder.rand_all(name)

    def _rand_colors(self):
        if "partial_body_change" in self._params["color_mod"]:
            for part in self._params["color_mod"]["partial_body_change"]["parts"]:
                self.model.geom_rgba[part, 0] = np.random.uniform(0,1)
                self.model.geom_rgba[part, 1] = np.random.uniform(0,1)
                self.model.geom_rgba[part, 2] = np.random.uniform(0,1)
        elif "geoms_change" in self._params["color_mod"]:
            for geom in self._params["color_mod"]["geoms_change"]:
                geom_id = self.model.geom_name2id(geom)
                self.model.geom_rgba[geom_id, :3] = sample([[0,1]]*3)
        elif "full_body_change" in self._params["color_mod"]:
            self.model.geom_rgba[:, 0] = np.random.uniform(0,1)
            self.model.geom_rgba[:, 1] = np.random.uniform(0,1)
            self.model.geom_rgba[:, 2] = np.random.uniform(0,1)
        else:
            a = self.model.geom_rgba.T[3, :].copy()
            self.model.geom_rgba[:] = np.random.rand(*self.model.geom_rgba.shape)
            self.model.geom_rgba.T[3, :] = a

    def _rand_camera(self):
        # Params
        cam_params = self._params["camera_mod"]
        if cam_params["use_default"]:
            # when default free camera is used
            # only works with rgb_array mode
            self.model.vis.global_.fovy = np.random.uniform(*cam_params["fovy_range"])
            self.unwrapped._viewers['rgb_array'].__init__(self.sim, -1)
            try:
                self.viewer_setup()
            except:
                self.unwrapped._viewer_setup()
            for key, value in cam_params['veiwer_cam_param_targets'].items():
                if isinstance(value, np.ndarray):
                    getattr(self.viewer.cam, key)[:] = sample(value)
                else:
                    setattr(self.viewer.cam, key, np.random.uniform(*value))
        else:
            # when camera is defined
            # Look approximately at the robot, but then randomize the orientation around that
            cam_pos = self.original_pos + sample(cam_params["pos_change_range"])
            self.cam_modder.set_pos(cam_params["cam_name"], cam_pos)
            self.sim.set_constants()

            quat = self.original_quat
            if "camera_focus" in cam_params:
                cam_id = self.cam_modder.get_camid(cam_params["cam_name"]) 
                target_id = self.model.body_name2id(cam_params["camera_focus"])
                quat = look_at(self.model.cam_pos[cam_id], self.sim.data.body_xpos[target_id])
            if "ang_jitter_range" in cam_params:
                quat = jitter_angle(quat, cam_params["ang_jitter_range"])
            
            self.cam_modder.set_quat(cam_params["cam_name"], quat)

            self.cam_modder.set_fovy(
                cam_params["cam_name"], np.random.uniform(*cam_params["fovy_range"])
            )

    def _rand_lights(self):
        """Randomize pos, direction, and lights"""
        # adjusting user defined lights
        light_params = self._params["light_mod"]
        if self.model.light_pos is not None:
            # pick light that is guaranteed to be on
            # other lights has 20% chance to be turned off
            always_on = np.random.choice(len(self.model.light_pos))
            for lightid in range(len(self.model.light_pos)):
                self.model.light_dir[lightid] = sample_light_dir()
                self.model.light_pos[lightid] = sample(light_params["pos_range"])

                if "color_range" in light_params:
                    color = np.array(sample(light_params["color_range"]))
                else:
                    color = np.ones(3)
                spec = np.random.uniform(*light_params['spec_range'])
                diffuse = np.random.uniform(*light_params['diffuse_range'])
                ambient = np.random.uniform(*light_params['ambient_range'])

                self.model.light_specular[lightid] = spec * color
                self.model.light_diffuse[lightid] = diffuse * color
                self.model.light_ambient[lightid] = ambient * color
                self.model.light_castshadow[lightid] = np.random.uniform(0, 1) < 0.5
        if light_params["head_light"]:
            if "color_range" in light_params:
                color = np.array(sample(light_params["color_range"]))
            else:
                color = np.ones(3)
            spec = np.random.uniform(*light_params['spec_range'])
            diffuse = np.random.uniform(*light_params['diffuse_range'])
            ambient = np.random.uniform(*light_params['ambient_range'])
            # adjust headlight
            self.model.vis.headlight.diffuse[:] = spec * color
            self.model.vis.headlight.ambient[:] = diffuse * color
            self.model.vis.headlight.specular[:] = diffuse * color

    def _rand_dynamics(self):
        dynamics_params = self._params["dynamics_mod"]
        if "action_mod" in dynamics_params:
            theta_degree = np.random.uniform(*dynamics_params["action_mod"]["theta"])
            theta = theta_degree * np.pi / 180 # pi/2 = 90 deg
            c, s = np.cos(theta), np.sin(theta)
            self.unwrapped.rot = np.array(((c, -s), (s, c)))
            self.unwrapped.bias = np.zeros(4)
            self.unwrapped.bias[2] = np.random.uniform(*dynamics_params["action_mod"]["bias"])

        if "body_mass_multiplier" in dynamics_params:
            multiplier = np.random.uniform(*dynamics_params["body_mass_multiplier"])
            self.model.body_inertia[:,:] = self._original_intertia * multiplier
            self.model.body_mass[:] =  self._original_mass * multiplier
        
        if "armature_multiplier" in dynamics_params:
            multiplier = np.random.uniform(*dynamics_params["armature_multiplier"])
            self.model.dof_armature[:] = self._original_armature * multiplier

        if "friction_targets" in dynamics_params:
            for geom, param in dynamics_params["friction_targets"].items():
                new_friction = param["range"][0] + np.random.rand() * (
                    param["range"][1] - param["range"][0]
                )
                geom_id = self.model.geom_name2id(geom)
                self.model.geom_friction[geom_id][0] = new_friction
        
        if "mass_targets" in dynamics_params:
            for body, param in dynamics_params["mass_targets"].items():
                new_mass = param["range"][0] + np.random.rand() * (
                    param["range"][1] - param["range"][0]
                )
                body_id = self.model.body_name2id(body)
                self.model.body_mass[body_id] = new_mass
                if param["inerta_reset"]:
                    self.model.body_inertia[body_id,:] = self.inertia_mass_one[body_id] * new_mass
        
        self.sim.set_constants()

    def reset(self):
        if self._params["tex_mod"]["active"]:
            self._rand_textures()
        if self._params["camera_mod"]["active"]:
            self._rand_camera()
        if self._params["light_mod"]["active"]:
            self._rand_lights()
        if self._params["color_mod"]["active"]:
            self._rand_colors()
        if self._params["dynamics_mod"]["active"]:
            self._rand_dynamics()

        return self.env.reset()

    def step(self, action, is_planner=False, is_mopa_rl=True):
        if isinstance(action, list):
            action = {key: val for ac_i in action for key, val in ac_i.items()}
        if isinstance(action, OrderedDict):
            action = np.concatenate(
                [
                    action[key]
                    for key in self.action_space.spaces.keys()
                    if key is not "ac_type"
                ]
            )

        self.env._pre_action(action)
        ob, reward, done, info = self.env._step(action, is_planner, is_mopa_rl)
        done, info, penalty = self.env._after_step(reward, done, info)
        return ob, reward + penalty, done, info

    def get_env_image(self):
        img = (super().render(mode='rgb_array')).astype('float32')
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        if self._config.save_img_to_disk:
            img_filepath = os.path.join(self.img_folder, 'img_{}.npy'.format(self.img_counter))
            np.save(img_filepath, img)
            self.img_counter += 1
            return img_filepath

        return img

    def _reset(self):
        if(self._config.env=="SawyerPushObstacle-v0"):
            self._set_camera_position(2, [-0.5, -0.4, 2.0]) 
            self._set_camera_rotation(2, [2.0, 0.5, 0])

            # Cam Position: Candidate 2
            # self._set_camera_position(2, [1.9, 0., 1.5]) # default camera position
            # self._set_camera_rotation(2, [0, 0, 0.7])

            init_qpos = (
                self.init_qpos + self.np_random.randn(self.init_qpos.shape[0]) * 0.02 # default configuration
            )

            self.sim.data.qpos[self.ref_joint_pos_indexes] = init_qpos
            self.sim.data.qvel[self.ref_joint_vel_indexes] = 0.0
            init_target_qpos = self.sim.data.qpos[self.ref_target_pos_indexes]
            init_target_qpos += self.np_random.uniform(
                low=-0.01, high=0.01, size=init_target_qpos.shape[0] # default configuration
            )
            self.goal = init_target_qpos
            self.sim.data.qpos[self.ref_target_pos_indexes] = self.goal
            self.sim.data.qvel[self.ref_joint_vel_indexes] = 0.0
            self.sim.forward()

            return self._get_obs()
        
        elif(self._config.env=="SawyerLiftObstacle-v0"):
            self._set_camera_position(2, [1.16, 0., 2.85]) # default camera position

            # self._set_camera_position(2, [0.07, 0., 1.9]) # candidate 1 camera position
            # self._set_camera_rotation(2, [0.5, -0.2, 0.99])

            # self._set_camera_position(2, [0.3, -0.6, 2.35]) # candidate 2 camera position
            # self._set_camera_rotation(2, [0.35, -0.2, 0.99])

            # self._set_camera_position(2, [0.71, -0.39, 1.4]) # candidate 3 camera position
            # self._set_camera_rotation(2, [0, 0.1, 0])

            init_qpos = (
                self.init_qpos + self.np_random.randn(self.init_qpos.shape[0]) * 0.02
            )
            self.sim.data.qpos[self.ref_joint_pos_indexes] = init_qpos
            self.sim.data.qvel[self.ref_joint_vel_indexes] = 0.0
            self.sim.data.qvel[self.ref_joint_vel_indexes] = 0.0
            self.sim.forward()

            return self._get_obs()
        elif(self._config.env=="SawyerAssemblyObstacle-v0"):
            # default configuration
            init_qpos = (
                self.init_qpos + self.np_random.randn(self.init_qpos.shape[0]) * 0.02
            )
            self.sim.data.qpos[self.ref_joint_pos_indexes] = init_qpos
            self.sim.data.qvel[self.ref_joint_vel_indexes] = 0.0
            self.sim.forward()


            return self._get_obs()


    def compute_reward(self, action):
        if(self._config.env=="SawyerPushObstacle-v0"):
            reward_type = self._config.reward_type
            info = {}
            reward = 0

            right_gripper, left_gripper = (
                self.sim.data.get_site_xpos("right_eef"),
                self.sim.data.get_site_xpos("left_eef"),
            )
            gripper_site_pos = (right_gripper + left_gripper) / 2.0
            cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
            target_pos = self.sim.data.body_xpos[self.target_id]
            gripper_to_cube = np.linalg.norm(cube_pos - gripper_site_pos)
            cube_to_target = np.linalg.norm(cube_pos[:2] - target_pos[:2])
            reward_push = 0.0
            reward_reach = 0.0
            if gripper_to_cube < 0.1:
                reward_reach += 0.1 * (1 - np.tanh(10 * gripper_to_cube))

            if cube_to_target < 0.1:
                reward_push += 0.5 * (1 - np.tanh(5 * cube_to_target))
            reward += reward_push + reward_reach
            info = dict(reward_reach=reward_reach, reward_push=reward_push)

            if cube_to_target < self._config.distance_threshold:
                reward += self._config.success_reward
                self._success = True
                self._terminal = True

            return reward, info
        
        elif(self._config.env=="SawyerLiftObstacle-v0"):
            reward_type = self._config.reward_type
            info = {}
            reward = 0

            reach_mult = 0.1
            grasp_mult = 0.35
            lift_mult = 0.5
            hover_mult = 0.7

            reward_reach = 0.0
            gripper_site_pos = self.sim.data.get_site_xpos("grip_site")
            cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
            gripper_to_cube = np.linalg.norm(cube_pos - gripper_site_pos)
            reward_reach = (1 - np.tanh(10 * gripper_to_cube)) * reach_mult

            touch_left_finger = False
            touch_right_finger = False
            for i in range(self.sim.data.ncon):
                c = self.sim.data.contact[i]
                if c.geom1 == self.cube_geom_id:
                    if c.geom2 in self.l_finger_geom_ids:
                        touch_left_finger = True
                    if c.geom2 in self.r_finger_geom_ids:
                        touch_right_finger = True
                elif c.geom2 == self.cube_geom_id:
                    if c.geom1 in self.l_finger_geom_ids:
                        touch_left_finger = True
                    if c.geom1 in self.r_finger_geom_ids:
                        touch_right_finger = True
            has_grasp = touch_right_finger and touch_left_finger
            reward_grasp = int(has_grasp) * grasp_mult

            reward_lift = 0.0
            object_z_locs = self.sim.data.body_xpos[self.cube_body_id][2]
            if reward_grasp > 0.0:
                z_target = self.get_pos("bin1")[2] + 0.45
                z_dist = np.maximum(z_target - object_z_locs, 0.0)
                reward_lift = grasp_mult + (1 - np.tanh(15 * z_dist)) * (
                    lift_mult - grasp_mult
                )

            reward += max(reward_reach, reward_grasp, reward_lift)
            info = dict(
                reward_reach=reward_reach,
                reward_grasp=reward_grasp,
                reward_lift=reward_lift,
            )

            if reward_grasp > 0.0 and np.abs(object_z_locs - z_target) < 0.05:
                reward += self._config.success_reward
                self._success = True
                self._terminal = True
            else:
                self._success = False

            return reward, info
        
        elif(self._config.env=="SawyerAssemblyObstacle-v0"):
            info = {}
            reward = 0
            reward_type = self._config.reward_type
            pegHeadPos = self.sim.data.get_site_xpos("pegHead")
            hole = self.sim.data.get_site_xpos("hole")
            dist = np.linalg.norm(pegHeadPos - hole)
            hole_bottom = self.sim.data.get_site_xpos("hole_bottom")
            dist_to_hole_bottom = np.linalg.norm(pegHeadPos - hole_bottom)
            dist_to_hole = np.linalg.norm(pegHeadPos - hole)
            reward_reach = 0
            if dist < 0.3:
                reward_reach += 0.4 * (1 - np.tanh(15 * dist_to_hole))
            reward += reward_reach
            if dist_to_hole_bottom < 0.025:
                reward += self._config.success_reward
                self._success = True
                self._terminal = True

            return reward, info

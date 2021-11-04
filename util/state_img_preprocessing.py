import argparse
import os
import numpy as np
import tqdm
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("--rollout_folder", type=str, default="saved_rollouts/sawyer-push-46files-32px-withcritics", help="path to rollout folder")
parser.add_argument("--task", type=str, default="sawyer-push", choices=["2d-push", "sawyer-push", "sawyer-assembly", "sawyer-lift"], help="choice of task")

args = parser.parse_args()
rollout_folder = args.rollout_folder
task = args.task

obs = []
acs = []
imgs = []
full_state_obs = []

# create image folder for storing image files
img_folder = os.path.join(rollout_folder, 'images')
if not os.path.exists(img_folder):
    os.mkdir(img_folder)

img_id = 1
pickle_files = os.listdir(rollout_folder)
tqdm_outer = tqdm.tqdm(total=len(pickle_files), desc='Pickle Files', position=1)
for i, pickle_file in enumerate(pickle_files):
    pickle_filepath = os.path.join(rollout_folder, pickle_file)
    if os.path.isdir(pickle_filepath):
        tqdm_outer.update(1)
        continue
    rollout_file = open(pickle_filepath, 'rb')
    rollout_data = pickle.load(rollout_file)
    rollout_file.close()

    tqdm_inner = tqdm.tqdm(total=len(rollout_data), desc='Number of rollouts', position=1)
    for rollout in rollout_data:
        observations, actions, imgs_traj = rollout["ob"], rollout["ac"], rollout["img"]
        for i in range(len(actions)):
            if task == '2d-push':
                # ['default'][0:8] robot joints np.cos(theata) and np.sin(theta) 
                # ['default'][10:14] robot joints velocity
                ot = np.concatenate((observations[i]['default'][0:8], observations[i]['default'][10:14], observations[i]['fingertip'])) # concatenate into a (14,) single vector
            elif task == 'sawyer-push' or task == 'sawyer-assembly' or task == 'sawyer-lift':
                # concatenate into a (25,) single vector
                ot = np.concatenate((observations[i]['joint_pos'], observations[i]['joint_vel'], observations[i]['gripper_qpos'], observations[i]['gripper_qvel'], observations[i]['eef_pos'], observations[i]['eef_quat']))
            else:
                print('ERROR: no matching task...')
                exit(1)
            at = actions[i]["default"]
            obs.append(ot)
            acs.append(at)
            full_state_ob = [item for item in observations[i].values()]
            full_state_ob = np.concatenate(full_state_ob)
            full_state_obs.append(full_state_ob)
            img_filepath = os.path.join(img_folder, 'img_{}.npy'.format(img_id))
            np.save(img_filepath, imgs_traj[i][0])
            imgs.append(img_filepath)
            img_id += 1
        tqdm_inner.update(1)
    tqdm_outer.update(1)

result = {
    'obs': obs,
    'acs': acs,
    'imgs': imgs,
    'full_state_obs': full_state_obs
}
with open(os.path.join(rollout_folder, 'combined.pickle'), 'wb') as f:
    pickle.dump(result, f)
print('combined.pickle saved sucessfully in {}'.format(rollout_folder))
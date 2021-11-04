import argparse
import getpass

parser = argparse.ArgumentParser()

## training
parser.add_argument('--start_epoch', type=int, default=0, help="starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=1000, help="ending epoch for training")
parser.add_argument('--lrate', type=float, default=0.0005, help="initial learning rate for the policy network update")
parser.add_argument('--beta1', type=float, default=0.95, help="betas for Adam Optimizer")
parser.add_argument('--beta2', type=float, default=0.9, help="betas for Adam Optimizer")
parser.add_argument('--batch_size', type=int, default=512, help="batch size for model training")
parser.add_argument('--load_saved', type=bool, default=False, help="load weights from the saved model")
parser.add_argument('--model_save_dir', type=str, default='../checkpoints/sawyer_assembly_32px_0.4million_checkpoint', help="directory for saving trained model weights")
parser.add_argument('--checkpoint', type=str, default='epoch_12.pth', help="checkpoint file")
parser.add_argument('--saved_rollouts', type=str, default='../saved_rollouts', help="directory to load saved expert demonstrations from")
parser.add_argument('--saved_rollouts_file', type=str, default='../saved_rollouts/sawyer-assembly-21files-0.4million/combined.pickle', help="file to load saved expert demonstrations from")
parser.add_argument('--saved_rollouts_vis', type=str, default='../saved_rollouts', help="directory to save visualization of the covered states from saved bc data")
parser.add_argument('--seed', type=int, default=1234, help="torch seed value")
parser.add_argument('--num_threads', type=int, default=1, help="number of threads for execution")
parser.add_argument('--train_data_ratio', type=float, default=0.90, help="ratio for training data for train-test split")
parser.add_argument("--model", type=str, default="BC_Visual_Policy_Stochastic", choices=["BC_Visual_Policy", "BC_Image_Only", "BC_Robot_Only", "BC_Visual_Policy_Stochastic", "BC_Visual_Stochastic_w_Critics"], help="choice of model")

## overwrite env. arguments
parser.add_argument("--env", type=str, default="SawyerAssemblyObstacle-v0", choices=["PusherObstacle-v0", "SawyerPushObstacle-v0", "SawyerAssemblyObstacle-v0", "SawyerLiftObstacle-v0"], help="environment name")
parser.add_argument('--env_image_size', type=int, default=32, help="batch size for model training")
parser.add_argument("--env_seed", type=int, default=1234, help="random seed")
parser.add_argument("--screen_width", type=int, default=32, help="width of camera image")
parser.add_argument("--screen_height", type=int, default=32, help="height of camera image")


## data augmentation
parser.add_argument('--img_aug', type=bool, default=False, help="whether to use data augmentations on images")
# random crop
parser.add_argument('--random_crop', type=bool, default=True, help="whether to use random crop")
parser.add_argument('--random_crop_size', type=int, default=24, help="random crop size")


## MoPA-RL specific params
parser.add_argument('--mopa_ckpt', type=str, default='../out/mopa_rl_sawyer_push_checkpoint/ckpt_03000000.pt', help="path to MoPA-RL checkpoint")


## scheduler
parser.add_argument('--scheduler_step_size', type=int, default=5, help="step size for optimizer scheduler")
parser.add_argument('--scheduler_gamma', type=float, default=0.99, help="decay rate for optimizer scheduler")

## cuda
parser.add_argument('--cuda_num', type=str, default='1', help="use gpu for computation")

## logs
parser.add_argument('--wandb', type=bool, default=True, help="learning curves logged on weights and biases")
parser.add_argument('--print_iteration', type=int, default=1000, help="iteration interval for displaying current loss values")

## validation arguments
parser.add_argument('--num_eval_ep_validation_per_seed', type=int, default=5, help="number of episodes to run during evaluation")
parser.add_argument('--eval_bc_max_step_validation', type=int, default=400, help="maximum steps during evaluations of learnt bc policy")
parser.add_argument('--eval_interval', type=int, default=1, help="evaluation_interval")

## bc args
parser.add_argument('--image_rollouts', type=bool, default=False, help="whether the bc observations are state based or image based")
parser.add_argument('--stacked_states', type=bool, default=False, help="whether to use stacked frames as observations or individually")
parser.add_argument('--num_stack_frames', type=int, default=4, help="number of frames to be stacked for each observation")
parser.add_argument('--action_size', type=int, default=4, help="dimension of the action space")
parser.add_argument('--robot_state_size', type=int, default=14, help="dimension of the observation space")
parser.add_argument('--env_state_size', type=int, default=0, help="dimension of the environment space")
parser.add_argument('--bc_video_dir', type=str, default='../bc_visual_videos', help="directory to store behavioral cloning video simulations")
parser.add_argument('--eval_bc_max_step', type=int, default=1000, help="maximum steps during evaluations of learnt bc policy")
parser.add_argument('--num_eval_ep', type=int, default=100, help="number of episodes to run during evaluation")
parser.add_argument('--three_hundred_eval_five_seeds', type=bool, default=True, help="500 evaluations (100 for each random seed [1234, 200, 500, 2320, 1800])")
parser.add_argument('--discount_factor', type=float, default=0.99, help="discount factor for calculating discounted rewards")


args = parser.parse_args()

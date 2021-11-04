import argparse

from util import str2bool, str2list


def argparser():
    parser = argparse.ArgumentParser(
        "MoPA-RL", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # parser.add_argument("--date", type=str, default='01.01', help="experiment date") # for eval
    parser.add_argument("--date", type=str, default=None, help="experiment date")

    parser.add_argument("--log_dir", type=str, default=None, help="directly specify the log directory") # this will overrule "run_name" and use this value instead if not None

    # environment
    parser.add_argument(
        "--env", type=str, default="reacher-obstacle-v0", help="environment name"
    )

    parser.add_argument(
        "--obs_space", type=str, default="state", choices=["state", "image", "all"], help="observations as state or image" # 'state' for eval 
    )

    # domain randomization
    parser.add_argument(
        "--dr", type=str2bool, default=False, help="use domain randomization during training" # 'state' for eval 
    )

    parser.add_argument(
        "--dr_params_set", type=str, default="default", choices=["default", "sawyer_push", "sawyer_lift", "sawyer_assembly"], help="set to the required env dr params" # 'state' for eval 
    )

    parser.add_argument(
        "--dr_eval", type=str2bool, default="False", help="set to True for evaluating the domain randomization model" # 'state' for eval 
    )

    # training algorithm
    parser.add_argument(
        "--algo", type=str, default="sac", choices=["sac", "td3", "aac", "aac-sac", "aac-ddpg"], help="RL algorithm"
    )
    parser.add_argument(
        "--policy", type=str, default="mlp", choices=["mlp", "asym", "asym-ddpg", "full-image"], help="policy type"
    )
    parser.add_argument("--mopa", type=str2bool, default=False, help="enable MoPA")
    parser.add_argument(
        "--ac_space_type",
        type=str,
        default="piecewise",
        choices=["normal", "piecewise"],
        help="Action space type for MoPA",
    )
    parser.add_argument(
        "--use_ik_target",
        type=str2bool,
        default=False,
        help="Enable cartasian action space for inverse kienmatics",
    )
    parser.add_argument(
        "--ik_target",
        type=str,
        default="fingertip",
        help="reference location for inverse kinematics",
    )
    parser.add_argument(
        "--expand_ac_space",
        type=str2bool,
        default=False,
        help="enable larger action space for SAC",
    )

    # data augmentations
    parser.add_argument(
        "--random_crop", type=str2bool, default=False, help="whether to apply random crop with padding to ob. images"
    )

    # vanilla rl
    parser.add_argument("--rl_hid_size", type=int, default=256, help="hidden unit size")
    parser.add_argument(
        "--rl_activation",
        type=str,
        default="relu",
        choices=["relu", "elu", "tanh"],
        help="activation function",
    )
    parser.add_argument(
        "--tanh_policy", type=str2bool, default=True, help="enable tanh policy"
    )
    parser.add_argument(
        "--actor_num_hid_layers",
        type=int,
        default=2,
        help="number of hidden layer in an actor",
    )

    # motion planning
    parser.add_argument(
        "--invalid_target_handling",
        type=str2bool,
        default=False,
        help="enable invalid target handling",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="number of trials for invalid target handling",
    )
    parser.add_argument(
        "--interpolation",
        type=str2bool,
        default=True,
        help="enable interpolation for motion planner",
    )

    # BC + RL
    parser.add_argument(
        "--expert_mode", type=str, default="bc", choices=["mopa", "bc", "bc-stochastic", "bc-stochastic-randweights", "bc-stochastic-mopa", "mopa-sota", "bc-sota"], help="Choice of expert policy"
    )
    parser.add_argument(
        "--bc_rl_sampling_mode", type=str, default="two-buffers", choices=["two-buffers", "one-buffer"], help="Choice of bc+rl sampling methods: sampling from two buffers or sampling from only one buffer"
    )
    parser.add_argument(
        "--bc_checkpoint", type=str, default='/home/arthurliu/Documents/link_to_ssd_workspace/mopa-rl/out/bc_visual_policy_sawyer_assembly_32px_checkpoint_29out30/epoch_7.pth', help="path to checkpoint file"
    )
    parser.add_argument(
        "--mopa_checkpoint", type=str, default='/home/arthurliu/Documents/link_to_ssd_workspace/mopa-rl/out/bc_visual_policy_sawyer_assembly_32px_checkpoint_29out30/epoch_7.pth', help="path to checkpoint file"
    )
    parser.add_argument(
        "--expert_num_trajectories",
        type=int,
        default=2e4,
        # default=100, # for debugging
        help="number of expert trajectories collected and stored in an expert memory buffer before training",
    )
    parser.add_argument(
        "--initial_policy_num_trajectories",
        type=int,
        default=1e4,
        # default=1e2, # for debugging
        help="number of expert trajectories collected and stored in an expert memory buffer before training",
    )
    parser.add_argument(
        "--pretrain_step",
        type=int,
        default=int(1000),
        # default=int(10), # for debugging
        help="number of time steps to update networks using expert trajectories during pretraining phase",
    )
    parser.add_argument(
        "--save_img_to_disk", type=str2bool, default=False, help="enable saving/loading env. image to/from disk (RAM is used if parallel_dataloading_mode==ram)"
    )
    parser.add_argument(
        "--save_img_folder", type=str, default='out/test-img_folder', help="path to image folder that saves/loads env. images" # for debugging
    )
    parser.add_argument(
        "--parallel_dataloading", type=str2bool, default=False, help="enable parallel dataloading to increase training speed"
    )
    parser.add_argument(
        "--parallel_dataloading_mode", type=str, default="disk", choices=["ram", "disk"], help="RL algorithm"
    )
    parser.add_argument(
        "--percent_expert_batch_size",
        type=float,
        default=0.25,
        help="this value * batch_size = expert_batch_size ; policy_batch_size = batch_size - expert_batch_size",
    )
    parser.add_argument(
        "--pretraining_expert_trajectories_collection_interval", type=int, default=1000, help="interval for sampling expert trajectories during pre-training"
    )
    parser.add_argument(
        "--expert_trajectories_collection_interval", type=int, default=10000, help="interval for sampling expert trajectories during main training phase"
    )
    parser.add_argument(
        "--pretrain_evaluate_interval", type=int, default=10000, help="interval for evaluation"
    )
    parser.add_argument(
        "--step_switch_policy_only", type=int, default=2000000, help="training step where expert trajectories are no longer being used and only policy's are used"
    )
    parser.add_argument(
        "--env_image_size",
        type=int,
        default=int(32),
        help="environment image size",
    )
    parser.add_argument(
        "--log_alpha", type=float, default=1.0, help="log alpha"
    )

    # BC + RL DDPG specific
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor. (Always between 0 and 1.)"
    )
    parser.add_argument(
        "--act_noise", type=float, default=0.1, help="Stddev for Gaussian exploration noise added to policy at training time. (At test time, no noise is added.)"
    )

    # MoPA
    parser.add_argument(
        "--omega",
        type=float,
        default=1.0,
        help="threshold of direct action execution and motion planner",
    )
    parser.add_argument(
        "--reuse_data", type=str2bool, default=False, help="enable reuse of data"
    )
    parser.add_argument(
        "--max_reuse_data",
        type=int,
        default=30,
        help="maximum number of reused samples",
    )
    parser.add_argument(
        "--action_range", type=float, default=1.0, help="range of radian"
    )
    parser.add_argument(
        "--discrete_action",
        type=str2bool,
        default=False,
        help="enable discrete action to choose motion planner or direct action execution",
    )
    parser.add_argument(
        "--stochastic_eval",
        type=str2bool,
        default=False,
        help="eval an agent with a stochastic policy",
    )

    # off-policy rl
    parser.add_argument(
        # "--buffer_size", type=int, default=int(1e6), help="the size of the buffer" # default buffer size
        "--buffer_size", type=int, default=int(5e5), help="the size of the buffer"
        # "--buffer_size", type=int, default=int(4e5), help="the size of the buffer"
        # "--buffer_size", type=int, default=int(1e5), help="the size of the buffer"
    )
    parser.add_argument(
        "--discount_factor", type=float, default=0.99, help="the discount factor"
    )
    parser.add_argument(
        # "--lr_actor", type=float, default=0.00001, help="the learning rate of the actor"
        "--lr_actor", type=float, default=0.0003, help="the learning rate of the actor" # default learning rate
    )
    parser.add_argument(
        # "--lr_critic", type=float, default=0.00001, help="the learning rate of the critic"
        "--lr_critic", type=float, default=0.0003, help="the learning rate of the critic" # default learning rate
    )
    parser.add_argument(
        "--polyak", type=float, default=0.995, help="the average coefficient"
    )
    parser.add_argument(
        "--actor_update_freq", type=int, default=1, help="frequency of actor update"
    )

    # training
    parser.add_argument(
        "--is_train", type=str2bool, default=True, help="enable training"  # False for eval
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help="the times to update the network per epoch",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="the sample batch size" # default batch size
        # "--batch_size", type=int, default=64, help="the sample batch size"
    )
    parser.add_argument(
        "--max_global_step",
        type=int,
        default=int(3e6),
        help="maximum number of time steps",
    )
    parser.add_argument("--gpu", type=int, default=None, help="gpu id")

    parser.add_argument(
        "--load_pretrained", type=str2bool, default=False, help="load pretrained weights obtained from BC visual policy"
    )

    parser.add_argument(
        "--checkpoint_path", type=str, default='/home/arthurliu/Documents/link_to_ssd_workspace/mopa-rl/out/best_bc_visual_checkpoint/64px/epoch_23.pth', help="path to checkpoint file"
    )

    parser.add_argument(
        "--bc_loss", type=str2bool, default=False, help="enable BC loss using pretrained BC visual model"
    )

    # sac
    parser.add_argument(
        "--start_steps",
        type=int,
        default=1e4,
        # default=1e1, # for debugging
        help="number of samples collected before training",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for Gumbel Softmax"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="inverse of reward scale"
    )
    parser.add_argument(
        "--reward_scale", type=float, default=1.0, help="reward scale for SAC"
    )
    parser.add_argument(
        "--use_smdp_update",
        type=str2bool,
        default=False,
        help="update a policy under semi-markov decision process",
    )

    # td3
    parser.add_argument(
        "--target_noise", type=float, default=0.2, help="target noise for TD3"
    )
    parser.add_argument(
        "--action_noise", type=float, default=0.1, help="action noise for TD3"
    )
    parser.add_argument(
        "--noise_clip", type=float, default=0.5, help="noise clip for TD3"
    )

    # log
    parser.add_argument(
        "--log_interval", type=int, default=1000, help="interval for logging"
    )
    parser.add_argument(
        "--vis_replay_interval",
        type=int,
        default=10000,
        help="interval for visualization of replay buffer",
    )
    parser.add_argument(
        "--evaluate_interval", type=int, default=40000, help="interval for evaluation"
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=100000,
        help="interval for saving a checkpoint file",
    )
    parser.add_argument(
        "--log_root_dir", type=str, default="log", help="root directory for logging"
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=False, # False for eval
        help="set it True if you want to use wandb",
    )
    parser.add_argument(
        "--entity", type=str, default="", help="Set an entity name for wandb"
    )
    parser.add_argument(
        "--project", type=str, default="", help="set a project name for wandb"
    )
    parser.add_argument("--group", type=str, default=None, help="group for wandb")
    parser.add_argument(
        "--vis_replay",
        type=str2bool,
        default=True,
        help="enable visualization of replay buffer",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="2d",
        choices=["2d", "3d"],
        help="plot type for replay buffer visualization",
    )
    parser.add_argument(
        "--vis_info",
        type=str2bool,
        default=True,
        help="enable visualization information of rollout in a video",
    )

    # evaluation
    # parser.add_argument("--ckpt_num", type=int, default=3000000, help="checkpoint nubmer") # for eval
    parser.add_argument("--ckpt_num", type=int, default=None, help="checkpoint nubmer")
    parser.add_argument(
        # "--num_eval", type=int, default=10, help="number of evaluations"
        "--num_eval", type=int, default=500, help="number of evaluations"
    )
    parser.add_argument(
        "--save_rollout",
        type=str2bool,
        default=False, # True for eval
        help="save rollout information during evaluation",
    )
    parser.add_argument(
        "--record", type=str2bool, default=False, help="enable video recording" # False for eval
    )
    parser.add_argument("--record_caption", type=str2bool, default=True)
    parser.add_argument(
        "--num_record_samples",
        type=int,
        default=1,
        help="number of trajectories to collect during eval",
    )
    parser.add_argument(
        "--three_hundred_eval_five_seeds", type=str2bool, default=False, help="500 evaluations (100 for each random seed [1234, 200, 500, 2320, 1800])"
    )
    parser.add_argument(
        "--eval_noise", type=bool, default=False, help="whether to add gaussian noise in the expert trajectories or not"
    )

    # misc
    parser.add_argument("--prefix", type=str, default="test", help="prefix for wandb")
    parser.add_argument("--notes", type=str, default="", help="notes")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument(
        "--debug", type=str2bool, default=False, help="enable debugging model"
    )

    return parser

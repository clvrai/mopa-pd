#<!/bin/bash -x
gpu=$1
seed=$2

# Model parameters
prefix="BC-RL-SAC"
algo='aac'
policy="asym"
obs_space="all"
lr_actor='0.00001'
lr_critic='0.00001'
pretrain_step='15000'
pretrain_evaluate_interval='5000'

# machine parameters
bc_checkpoint="out/bc_visual_policy_sawyer_assembly_32px_checkpoint_30out30_screenresfix/epoch_30.pth"
mopa_checkpoint="out/mopa_rl_sawyer_assembly_checkpoint/ckpt_03000000.pt"
save_img_to_disk="False"
save_img_folder="out/bc_visual_policy_sawyer_assembly_32px_checkpoint_30out30_screenresfix-img_folder"
parallel_dataloading="False"
parallel_dataloading_mode="disk"

# data related parameters
env_image_size='32'

env="SawyerAssemblyObstacle-v0"
max_episode_step="250"
debug="False"
reward_type='sparse'
log_root_dir="./logs"
mopa="True"
reuse_data="True"
action_range="0.5"
omega='0.7'
stochastic_eval="True"
invalid_target_handling="True"
vis_replay="True"
plot_type='3d'
ac_space_type="piecewise"
use_smdp_update="True"
success_reward="150.0"
max_reuse_data='15'
reward_scale="1.0"
evaluate_interval="10000"


# for evaluation
date='None'
# log_dir='out/bc-rl-asym-sac-assembly-best-checkpoint/absurd-sun-250' # without pretrained actor/critic
# ckpt_num='3200000' # without pretrained actor/critic
log_dir='out/bc_rl_asym_sac_pretrained_actor_critic_1234_3dassembly' # with pretrained actor/critic
ckpt_num='1750000' # with pretrained actor/critic
is_train='False'
wandb='False'
num_eval='100'
save_rollout='True'
record='True'
three_hundred_eval_five_seeds='True'

python -m rl.main \
    --log_root_dir $log_root_dir \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --reward_type $reward_type \
    --mopa $mopa \
    --reuse_data $reuse_data \
    --action_range $action_range \
    --omega $omega \
    --stochastic_eval $stochastic_eval \
    --invalid_target_handling $invalid_target_handling \
    --vis_replay $vis_replay \
    --plot_type $plot_type \
    --use_smdp_update $use_smdp_update \
    --ac_space_type $ac_space_type \
    --success_reward $success_reward \
    --max_reuse_data $max_reuse_data \
    --reward_scale $reward_scale \
    --policy $policy \
    --obs_space $obs_space \
    --evaluate_interval $evaluate_interval \
    --lr_actor $lr_actor \
    --lr_critic $lr_critic \
    --bc_checkpoint $bc_checkpoint \
    --save_img_to_disk $save_img_to_disk \
    --save_img_folder $save_img_folder \
    --parallel_dataloading $parallel_dataloading \
    --parallel_dataloading_mode $parallel_dataloading_mode \
    --env_image_size $env_image_size \
    --pretrain_step $pretrain_step \
    --pretrain_evaluate_interval $pretrain_evaluate_interval \
    --date $date \
    --is_train $is_train \
    --wandb $wandb \
    --ckpt_num $ckpt_num \
    --num_eval $num_eval \
    --save_rollout $save_rollout \
    --record $record \
    --log_dir $log_dir \
    --mopa_checkpoint $mopa_checkpoint \
    --three_hundred_eval_five_seeds $three_hundred_eval_five_seeds \

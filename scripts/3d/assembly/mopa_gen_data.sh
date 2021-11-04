#<!/bin/bash -x
# Please modify mopa_gen_data.sh first before running this script.

gpu=$1
seed=$2

prefix="MoPA-SAC"
algo='sac'
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

# evaluation parameters
log_dir='out/mopa_rl_sawyer_assembly_1million_checkpoint'
ckpt_num='1000000'
date='01.01'
obs_space='state'
policy='mlp'
is_train='False'
wandb='False'
num_eval='500'
save_rollout='True'
record='False'
eval_noise='True'

# data related parameters
env_image_size='32'
screen_width='32'
screen_height='32'

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
    --date $date \
    --obs_space $obs_space \
    --policy $policy \
    --is_train $is_train \
    --wandb $wandb \
    --ckpt_num $ckpt_num \
    --num_eval $num_eval \
    --save_rollout $save_rollout \
    --record $record \
    --eval_noise $eval_noise \
    --screen_width $screen_width \
    --screen_height $screen_height \
    --env_image_size $env_image_size \
    --log_dir $log_dir \

filename_prefix = 'SawyerLift-ADR'
xlabel = 'Environment steps (1.4M)'
ylabel = "Average Discounted Rewards"
max_step = 1400000
max_y_axis_value = 110
legend = False
data_key = "train_ep/rew_discounted"
bc_y_value = 37.61

plot_labels = {
    "Ours": ['Ours - 3DLift - olive-salad-40_log_alpha_-4.2_seed_1234', 'Ours - 3DLift - leafy-shape-41_seed_200', 'Ours - 3DLift - true-durian-42_seed_500'],
    "Ours (w/o BC smoothing)": ['Ours MoPA - 3DLift - honest-plant-589 - 1234', 'Ours MoPA - 3DLift - young-deluge-590 - 200', 'Ours MoPA - 3DLift - happy-salad-591 - 500'],
    "CoL": ['CoL-MoPA-RL - 3DLift - icy-durian-564 - 1234', 'CoL-MoPA-RL - 3DLift - feasible-moon-44-200', 'CoL-MoPA-RL - 3DLift - ethereal-firebrand-45-500'],
    "CoL(w BC smoothing)": ['CoL-BC - 3DLift - rose-silence-4_seed_1234', 'CoL-BC - 3DLift - eager-rain-5_seed_200', 'CoL-BC - 3DLift - fancy-smoke-6_seed_500'],
    "MoPA Asym. SAC": ['MoPA Asymm. SAC - rerun - rl.SawyerLiftObstacle-v0.06.17.00.34.Asymmetric-MoPA-SAC.0 - 1234', 'MoPA Asymm. SAC - rerun - rl.SawyerLiftObstacle-v0.06.17.00.32.Asymmetric-MoPA-SAC.0 - 200', 'MoPA Asymm. SAC - rerun - rl.SawyerLiftObstacle-v0.06.17.00.33.Asymmetric-MoPA-SAC.0 - 500'],
    "Asym. SAC": ['Asym. SAC - 3DLift - efficient-cosmos-53 - 1234', 'Asym. SAC - 3DLift - ethereal-snowflake-54 - 200', 'Asym. SAC - 3DLift - revived-bush-55 - 500'],
}

line_labels = {
    "BC-Visual": ['Ours - 3DLift - olive-salad-40_log_alpha_-4.2_seed_1234'],
}

line_colors = {
    'Ours': 'C0',
    'Ours (w/o BC smoothing)': 'C1',
    'CoL': 'C2',
    'CoL(w BC smoothing)': 'C3',
    'MoPA Asym. SAC': 'C4',
    'Asym. SAC': 'C5',
    'BC-Visual': 'C6',
}
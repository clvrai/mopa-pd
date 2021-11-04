filename_prefix = 'SawyerPush-ASR'
xlabel = 'Environment steps (1.2M)'
ylabel = "Average Success Rate"
max_step = 1200000
max_y_axis_value = 1.0
legend = True
data_key = "train_ep/episode_success"
bc_y_value = 0.9967

plot_labels = {
    "Ours": ['Ours - 3DPush - lyric-meadow-35-seed-1234', 'Ours - 3DPush - bumbling-shadow-37-seed-200', 'Ours - 3DPush - icy-sea-38-seed-500'],
    "Ours (w/o BC smoothing)": ['Ours MoPA - 3DPush - stellar-vortex-586 - 1234', 'Ours MoPA - 3DPush - northern-smoke-587 - 200', 'Ours MoPA - 3DPush - bumbling-bush-588 - 500'],
    "CoL": ['CoL-MoPA-RL - 3DPush - helpful-terrain-563 - 1234', 'CoL-MoPA-RL - 3DPush - swept-lake-580 - 200', 'CoL-MoPA-RL - 3DPush - graceful-lake-581 - 500'],
    "CoL(w BC smoothing)": ['CoL-BC - 3DPush - icy-hill-1_seed_1234', 'CoL-BC - 3DPush - radiant-glitter-2_seed_200', 'CoL-BC - 3DPush - decent-morning-3_seed_500'],
    "MoPA Asym. SAC": ['Asym. SAC - 3DPush - jumping-flower-50 - 1234', 'Asym. SAC - 3DPush - jumping-flower-50 - 1234', 'Asym. SAC - 3DPush - jumping-flower-50 - 1234'],
    "Asym. SAC": ['Asym. SAC - 3DPush - jumping-flower-50 - 1234', 'Asym. SAC - 3DPush - glad-breeze-51 - 200', 'Asym. SAC - 3DPush - super-cloud-52 - 500'],
}

line_labels = {
    "BC-Visual": ['Ours - 3DPush - lyric-meadow-35-seed-1234'],
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

# plot_labels = {
#     "Ours": [],
#     "Ours (w/o BC smoothing)": [],
#     "CoL": [],
#     "CoL(w BC smoothing)": [],
#     "MoPA Asym. SAC": [],
#     "Asym. SAC": [],
# }

# line_labels = {
#     "BC-Visual": [],
# }
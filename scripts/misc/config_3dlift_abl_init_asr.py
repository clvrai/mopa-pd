filename_prefix = 'SawyerLift-Abl-Init-ASR'
xlabel = 'Environment steps (1.4M)'
ylabel = "Average Success Rate"
max_step = 1400000
max_y_axis_value = 1.0
legend = False
data_key = "train_ep/episode_success"
bc_y_value = 0

plot_labels = {
    "Ours": ['Ours - 3DLift - olive-salad-40_log_alpha_-4.2_seed_1234', 'Ours - 3DLift - leafy-shape-41_seed_200', 'Ours - 3DLift - true-durian-42_seed_500'],
    "Ours (w/o initalization)": ['abl. - 3DLift - wo-init - blooming-glitter-69 - 1234', 'abl. - 3DLift - wo-init - lemon-haze-70 - 200', 'abl. - 3DLift - wo-init - wandering-violet-71 - 500'],
}

line_labels = {
    "BC-Visual": [],
}

line_colors = {
    'Ours': 'C0',
    'Ours (w/o initalization)': 'C1',
}


filename_prefix = 'SawyerPush-Abl-Init-ASR'
xlabel = 'Environment steps (1.2M)'
ylabel = "Average Success Rate"
max_step = 1200000
max_y_axis_value = 1.0
legend = True
data_key = "train_ep/episode_success"
bc_y_value = 0

plot_labels = {
    "Ours": ['Ours - 3DPush - lyric-meadow-35-seed-1234', 'Ours - 3DPush - bumbling-shadow-37-seed-200', 'Ours - 3DPush - icy-sea-38-seed-500'],
    "Ours (w/o initalization)": ['abl. - 3DPush - wo-init - royal-energy-63 - 1234', 'abl. - 3DPush - wo-init - comfy-snowflake-64 - 200', 'abl. - 3DPush - wo-init - spring-haze-65 - 500'],
}

line_labels = {
    "BC-Visual": [],
}

line_colors = {
    'Ours': 'C0',
    'Ours (w/o initalization)': 'C1',
}


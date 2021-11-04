filename_prefix = 'SawyerAssembly-Abl-Init-ASR'
xlabel = 'Environment steps (1.3M)'
ylabel = "Average Success Rate"
max_step = 1300000
max_y_axis_value = 1.0
legend = False
data_key = "train_ep/episode_success"
bc_y_value = 0

plot_labels = {
    "Ours": ['Ours - 3DAssembly - splendid-dragon-582-seed-1234', 'Ours - 3DAssembly - pleasant-thunder-48-seed-200', 'Ours - 3DAssembly - wise-surf-49-seed-500'],
    "Ours (w/o initalization)": ['abl. - 3DAssembly - wo-init - radiant-valley-99 - 1234', 'abl. - 3DAssembly - wo-init - sunny-firefly-98 - 200', 'abl. - 3DAssembly - wo-init - legendary-water-97 - 500'],
}

line_labels = {
    "BC-Visual": [],
}

line_colors = {
    'Ours': 'C0',
    'Ours (w/o initalization)': 'C1',
}


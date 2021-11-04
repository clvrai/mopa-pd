filename_prefix = 'SawyerAssembly-ADR'
xlabel = 'Environment steps (1.4M)'
ylabel = "Average Discounted Rewards"
max_step = 1400000
max_y_axis_value = 110
legend = True
data_key = "train_ep/rew_discounted"
bc_y_value = 48.48

plot_labels = {
    "Ours": ['Ours - 3DAssembly - splendid-dragon-582-seed-1234', 'Ours - 3DAssembly - pleasant-thunder-48-seed-200', 'Ours - 3DAssembly - wise-surf-49-seed-500'],
    "Ours (w/o BC smoothing)": ['Ours MoPA - 3DAssembly - ruby-music-592 - 1234', 'Ours MoPA - 3DAssembly - driven-sun-593 - 200', 'Ours MoPA - 3DAssembly - proud-flower-594 - 500'],
    "CoL": ['CoL-MoPA-RL - 3DAssembly - zany-thunder-27 - 1234', 'CoL-MoPA-RL - 3DAssembly - charmed-dawn-28 - 200', 'CoL-MoPA-RL - 3DAssembly - vibrant-vortex-29 - 500'],
    "CoL(w BC smoothing)": ['CoL-BC - 3DAssembly - flowing-surf-7 - 1234', 'CoL-BC - 3DAssembly - vocal-bush-8 - 200', 'CoL-BC - 3DAssembly - good-durian-9 - 500'],
    "MoPA Asym. SAC": ['CoL-MoPA-RL - 3DAssembly - charmed-dawn-28 - 200', 'CoL-MoPA-RL - 3DAssembly - charmed-dawn-28 - 200', 'CoL-MoPA-RL - 3DAssembly - charmed-dawn-28 - 200'],
    "Asym. SAC": ['Asym. SAC - 3DAssembly -cosmic-mountain-56 - 1234', 'Asym. SAC - 3DAssembly - radiant-oath-57 - 200', 'Asym. SAC - 3DAssembly - kind-feather-58 - 500'],
}

line_labels = {
    "BC-Visual": ['Ours - 3DAssembly - splendid-dragon-582-seed-1234'],
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
filename_prefix = 'SawyerAssembly-Abl-Alpha-ASR-Line3'
xlabel = 'Environment steps (1.4M)'
ylabel = "Average Success Rate"
max_step = 1400000
max_y_axis_value = 1.0
legend = False
data_key = "train_ep/episode_success"
bc_y_value = 0

plot_labels = {
    r'log($\alpha$): 0.5': ['abl._3DAssembly_ours_log_alpha_0.5_gallant-microwave-589'],
    r'log($\alpha$): 0': ['abl._3DAssembly_ours_log_alpha_0_celestial-sun-590'],
    # r'log($\alpha$): -0.5': ['abl._3DAssembly_ours_log_alpha_-0.5_genial-thunder-591'],
    # r'log($\alpha$): -1': ['abl._3DAssembly_ours_log_alpha_-1_polished-galaxy-592'],
    r'log($\alpha$): -3': ['abl._3DAssembly_ours_log_alpha_-3_lunar-valley-593'],
    # r'log($\alpha$): -5': ['abl._3DAssembly_ours_log_alpha_-5_fresh-cosmos-594'],
    # r'log($\alpha$): -10': ['abl._3DAssembly_ours_log_alpha_-10_mild-snow-596'],
    # r'log($\alpha$): -20': ['abl._3DAssembly_ours_log_alpha_-20_glowing-dawn-597'],
    r'log($\alpha$): -40': ['Ours - 3DAssembly - splendid-dragon-582-seed-1234'],
}

line_labels = {}

line_colors = {
    r'log($\alpha$): 0.5': 'C0',
    r'log($\alpha$): 0': 'C1',
    # r'log($\alpha$): -0.5': 'C2',
    # r'log($\alpha$): -1': 'C3',
    r'log($\alpha$): -3': 'C4',
    # r'log($\alpha$): -5': 'C5',
    # r'log($\alpha$): -10': 'C6',
    # r'log($\alpha$): -20': 'C7',
    r'log($\alpha$): -40': 'C8',
}
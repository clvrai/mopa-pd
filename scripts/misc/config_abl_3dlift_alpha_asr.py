filename_prefix = 'SawyerLift-Abl-Alpha-ASR'
xlabel = 'Environment steps (1.4M)'
ylabel = "Average Success Rate"
max_step = 1400000
max_y_axis_value = 1.0
legend = False
data_key = "train_ep/episode_success"
bc_y_value = 0

plot_labels = {
    r'log($\alpha$): 0.5': ['abl._3DLift_ours_log_alpha_0.5_twilight-fire-585'],
    r'log($\alpha$): 0': ['abl._3DLift_ours_log_alpha_0_dulcet-blaze-584'],
    # r'log($\alpha$): -0.5': ['abl._3DLift_ours_log_alpha_-0.5_northern-lake-587'],
    # r'log($\alpha$): -1': ['abl._3DLift_ours_log_alpha_-1_dulcet-water-588'],
    r'log($\alpha$): -3': ['abl._3DLift_ours_log_alpha_-3_noble-plasma-589'],
    # r'log($\alpha$): -5': ['abl._3DLift_ours_log_alpha_-5_celestial-blaze-590'],
    r'log($\alpha$): -10': ['abl._3DLift_ours_log_alpha_-10_avid-dream-591'],
    r'log($\alpha$): -20': ['abl._3DLift_ours_log_alpha_-20_spring-armadillo-586'],
    r'log($\alpha$): -40': ['abl._3DLift_ours_log_alpha_-40_fresh-firefly-579'],
}

line_labels = {}

line_colors = {
    r'log($\alpha$): 0.5': 'C0',
    r'log($\alpha$): 0': 'C1',
    # r'log($\alpha$): -0.5': 'C2',
    # r'log($\alpha$): -1': 'C3',
    r'log($\alpha$): -3': 'C4',
    # r'log($\alpha$): -5': 'C5',
    r'log($\alpha$): -10': 'C6',
    r'log($\alpha$): -20': 'C7',
    r'log($\alpha$): -40': 'C8',
}
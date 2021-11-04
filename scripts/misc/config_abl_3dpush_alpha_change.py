filename_prefix = 'SawyerPush-Abl-Alpha-Change'
xlabel = 'Environment steps (1.2M)'
ylabel = r'$\alpha$'
max_step = 1200000
max_y_axis_value = 2.4
legend = True
data_key = "train_rl/entropy_alpha"
bc_y_value = 0
smoothing = False
smoothing_weight = 0
legend_loc='upper left'


plot_labels = {
    r'log($\alpha$): 0.5': ['abl._3DPush_ours_log_alpha_0.5_upbeat-serenity-599'],
    r'log($\alpha$): 0': ['abl._3DPush_ours_log_alpha_0_rose-bird-600'],
    # r'log($\alpha$): -0.5': ['abl._3DPush_ours_log_alpha_-0.5_vague-cosmos-601'],
    # r'log($\alpha$): -1': ['abl._3DPush_ours_log_alpha_-1_breezy-moon-602'],
    r'log($\alpha$): -3': ['abl._3DPush_ours_log_alpha_-3_devoted-serenity-603'],
    # r'log($\alpha$): -5': ['abl._3DPush_ours_log_alpha_-5_major-resonance-604'],
    r'log($\alpha$): -10': ['abl._3DPush_ours_log_alpha_-10_fluent-glitter-605'],
    r'log($\alpha$): -20': ['abl._3DPush_ours_log_alpha_-20_valiant-deluge-606'],
    r'log($\alpha$): -40': ['abl._3DPush_ours_log_alpha_-40_olive-elevator-607'],
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


# ######## for debugging
# plot_labels = {
#     # r'log($\alpha$): 0.5': ['abl._3DPush_ours_log_alpha_0.5_upbeat-serenity-599'],
#     # r'log($\alpha$): 0': ['abl._3DPush_ours_log_alpha_0_rose-bird-600'],
#     # # r'log($\alpha$): -0.5': ['abl._3DPush_ours_log_alpha_-0.5_vague-cosmos-601'],
#     # # r'log($\alpha$): -1': ['abl._3DPush_ours_log_alpha_-1_breezy-moon-602'],
#     # r'log($\alpha$): -3': ['abl._3DPush_ours_log_alpha_-3_devoted-serenity-603'],
#     # r'log($\alpha$): -5': ['abl._3DPush_ours_log_alpha_-5_major-resonance-604'],
#     r'log($\alpha$): -10': ['abl._3DPush_ours_log_alpha_-10_fluent-glitter-605'],
#     # r'log($\alpha$): -20': ['abl._3DPush_ours_log_alpha_-20_valiant-deluge-606'],
#     # r'log($\alpha$): -40': ['abl._3DPush_ours_log_alpha_-40_olive-elevator-607'],
# }

# line_labels = {}

# line_colors = {
#     # r'log($\alpha$): 0.5': 'C0',
#     # r'log($\alpha$): 0': 'C1',
#     # # r'log($\alpha$): -0.5': 'C2',
#     # # r'log($\alpha$): -1': 'C3',
#     # r'log($\alpha$): -3': 'C4',
#     # r'log($\alpha$): -5': 'C5',
#     r'log($\alpha$): -10': 'C6',
#     # r'log($\alpha$): -20': 'C7',
#     # r'log($\alpha$): -40': 'C8',
# }
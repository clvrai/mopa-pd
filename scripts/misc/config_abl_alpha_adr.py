filename_prefix = 'SawyerLift-Abl-Alpha-ADR'
xlabel = 'Environment steps (1.4M)'
ylabel = "Average Discounted Rewards"
max_step = 1400000
max_y_axis_value = 110
legend = True
data_key = "train_ep/rew_discounted"
bc_y_value = 0

plot_labels = {
    r'log($\alpha$): 0.5': ['abl. - 3DLift - ours - log_alpha 0.5 - twilight-fire-585'],
    r'log($\alpha$): 0': ['abl. - 3DLift - ours - log_alpha 0 - dulcet-blaze-584'],
    r'log($\alpha$): -0.5': ['abl. - 3DLift - ours - log_alpha -0.5 - northern-lake-587'],
    r'log($\alpha$): -1': ['abl. - 3DLift - ours - log_alpha -1 dulcet-water-588'],
    r'log($\alpha$): -3': ['abl. - 3DLift - ours - log_alpha -3 noble-plasma-589'],
    r'log($\alpha$): -20': ['abl. - 3DLift - ours - log_alpha -20 - spring-armadillo-586'],
}

line_labels = {}

line_colors = {
    r'log($\alpha$): 0.5': 'C0',
    r'log($\alpha$): 0': 'C1',
    r'log($\alpha$): -0.5': 'C2',
    r'log($\alpha$): -1': 'C3',
    r'log($\alpha$): -3': 'C4',
    r'log($\alpha$): -20': 'C5',
}
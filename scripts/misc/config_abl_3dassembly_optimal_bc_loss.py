# for this figure, we need to multiple all y values by 1e6 (y_data = y_data * 1e6) and set
# y-tick labels directly plt.yticks([1.3, 1.4, 1.6, 2.0], fontsize=12).

filename_prefix = 'SawyerAssembly-Abl-Optimal-BC-Loss'
xlabel = 'Epoch'
ylabel = 'Mean Square Error (x 1e-6)'
max_step = 40
min_y_axis_value = 1e-6
max_y_axis_value = 2e-6
legend = True
data_key = ["Action Prediction Loss (Train)", "Action Prediction Loss (Test)"]
bc_y_value = 0
smoothing = False
smoothing_weight = 0
legend_loc = 'upper right'
wandb_api_path = 'arthur801031/mopa-rl-bc-visual'
num_points = 40
x_scale = 1
divide_max_step_by_1mill = False
build_log_from_multiple_keys = True
limit_y_max = True
limit_y_max_value = 2e-6

plot_labels = {
    'Train': ['BC Visual Stochastic_3DAssembly_curious-spaceship-136'],
    'Test': ['BC Visual Stochastic_3DAssembly_curious-spaceship-136'],
}

line_labels = {}

line_colors = {
    'Train': 'C0',
    'Test': 'C1',
}
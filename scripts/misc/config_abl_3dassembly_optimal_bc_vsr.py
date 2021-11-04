filename_prefix = 'SawyerAssembly-Abl-Optimal-BC-VSR'
xlabel = 'Epoch'
ylabel = 'Validation Success Rate'
max_step = 40
max_y_axis_value = 1.0
legend = False
data_key = "Total Success"
bc_y_value = 0
smoothing = False
smoothing_weight = 0
legend_loc = 'upper left'
wandb_api_path = 'arthur801031/mopa-rl-bc-visual'
num_points = 40
x_scale = 1
divide_max_step_by_1mill = False

plot_labels = {
    'BC-Visual': ['BC Visual Stochastic_3DAssembly_curious-spaceship-136'],
}

line_labels = {}

line_colors = {
    'BC-Visual': 'C0',
}
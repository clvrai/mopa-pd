filename_prefix = 'Rebuttal-SawyerAssembly-ADR-BC-Smoothing'
xlabel = 'Environment steps (2M)'
ylabel = "Average Discounted Rewards"
max_step = 2000000
max_y_axis_value = 110
legend = False
data_key = "train_ep/rew_discounted"
bc_y_value = 41.31

plot_labels = {
    "Ours": [
        'Rebuttal_Ours_Assembly_1234_classic-cosmos-676',
        'Rebuttal_Ours_Assembly_200_spring-thunder-677',
        'Rebuttal_Ours_Assembly_2320_eager-yogurt-696',
        'Rebuttal_Ours_Assembly_500_firm-dawn-678',
        'Rebuttal_Ours_Assembly_1800_wise-water-677',
    ],
    "Ours (w/o BC smoothing)": [
        'Rebuttal_Ours(wo_bc_smoothing)_Assembly_1234_classic-dust-688',
        'Rebuttal_Ours(wo_bc_smoothing)_Assembly_200_restful-pine-689',
        'Rebuttal_Ours(wo_bc_smoothing)_Assembly_2320_lunar-blaze-695',
        'Rebuttal_Ours(wo_bc_smoothing)_Assembly_500_distinctive-salad-694',
        'Rebuttal_Ours(wo_bc_smoothing)_Assembly_1800_astral-leaf-674',
    ],
}

# choosing Rebuttal_Asym._SAC_Assembly_200_comfy-violet-635 because it has 3m env. steps
line_labels = {
    # "BC-Visual": ['Rebuttal_Asym._SAC_Assembly_200_comfy-violet-635'],
}

line_colors = {
    'Ours': 'C0',
    'Ours (w/o BC smoothing)': 'C1',
}






# plot_labels = {
#     "Ours": [],
#     "Ours (w/o BC smoothing)": [],
#     "CoL": [],
#     "CoL(w BC smoothing)": [],
#     "MoPA Asym. SAC": [],
#     "Asym. SAC": [],
# }

# line_labels = {
#     "BC-Visual": [],
# }
filename_prefix = 'Rebuttal-SawyerLift-ADR-BC-Smoothing'
xlabel = 'Environment steps (2M)'
ylabel = "Average Discounted Rewards"
max_step = 2000000
max_y_axis_value = 110
legend = False
data_key = "train_ep/rew_discounted"
bc_y_value = 34.77

plot_labels = {
    "Ours": [
        'Rebuttal_Ours_Lift_1234_smart-river-671',
        'Rebuttal_Ours_Lift_200_clean-cherry-673',
        'Rebuttal_Ours_Lift_2320_clear-water-695',
        'Rebuttal_Ours_Lift_500_vivid-lake-675',
        'Rebuttal_Ours_Lift_1800_lilac-butterfly-676',
    ],
    "Ours (w/o BC smoothing)": [
        'Rebuttal_Ours(wo_bc_smoothing)_Lift_1234_sparkling-terrain-670',
        'Rebuttal_Ours(wo_bc_smoothing)_Lift_200_toasty-pine-678',
        'Rebuttal_Ours(wo_bc_smoothing)_Lift_2320_ethereal-deluge-692',
        'Rebuttal_Ours(wo_bc_smoothing)_Lift_500_floral-water-679',
        'Rebuttal_Ours(wo_bc_smoothing)_Lift_1800_youthful-microwave-674',
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
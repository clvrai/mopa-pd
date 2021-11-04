filename_prefix = 'Rebuttal-SawyerPush-ADR-SOTA'
xlabel = 'Environment steps (3M)'
ylabel = "Average Discounted Rewards"
mopa_cutoff_step = 1000000
others_cutoff_step = 2000000
max_step = 3000000
max_y_axis_value = 110
legend = False
data_key = "train_ep/rew_discounted"
bc_y_value = 57.39

plot_labels = {
    "Ours": [
        'Rebuttal_Ours_Push_1234_grateful-sea-670',
        'Rebuttal_Ours_Push_200_dark-resonance-672',
        'Rebuttal_Ours_Push_2320_light-shape-694',
        'Rebuttal_Ours_Push_500_splendid-glade-674',
        'Rebuttal_Ours_Push_1800_fiery-butterfly-675',
    ],
    "CoL": [
        'Rebuttal_CoL_MoPA_Push_1234_royal-bee-190',
        'Rebuttal_CoL_MoPA_Push_2320_sweet-sunset-200',
        'Rebuttal_CoL_MoPA_Push_200_visionary-wave-191',
        'Rebuttal_CoL_MoPA_Push_500_giddy-star-192',
        'Rebuttal_CoL_MoPA_Push_1800-dainty-bush-223',
    ],
    "CoL(w BC smoothing)": [
        'Rebuttal_CoL_Push_1234_colorful-wind-680',
        'Rebuttal_CoL_Push_200_generous-snow-682',
        'Rebuttal_CoL_Push_2320_fragrant-sunset-693',
        'Rebuttal_CoL_Push_500_atomic-pond-684',
        'Rebuttal_CoL_Push_1800_zesty-wind-670',
    ],
}

# choosing Rebuttal_Asym._SAC_Assembly_200_comfy-violet-635 because it has 3m env. steps
line_labels = {
    # "BC-Visual": ['Rebuttal_Asym._SAC_Assembly_200_comfy-violet-635'],
}

line_colors = {
    'Ours': 'C0',
    'CoL': 'C2',
    'CoL(w BC smoothing)': 'C3',
}

mopa_curve = {
    "Ours": 'Rebuttal_MoPA_RL_Push_1234_rl.SawyerPushObstacle-v0.08.24.01.35.MoPA-SAC.0',
    "CoL": 'Rebuttal_MoPA_RL_Push_1234_rl.SawyerPushObstacle-v0.08.24.01.35.MoPA-SAC.0',
    "CoL(w BC smoothing)": 'Rebuttal_MoPA_RL_Push_1234_rl.SawyerPushObstacle-v0.08.24.01.35.MoPA-SAC.0',
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
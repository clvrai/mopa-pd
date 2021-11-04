filename_prefix = 'Rebuttal-SawyerPush-ADR-Baselines2'
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
    "MoPA Asym. SAC": [
        'Rebuttal_MoPA-Asym._SAC_Push_1234_rl.SawyerPushObstacle-v0.08.20.09.09.Asymmetric-MoPA-SAC.0',
        'Rebuttal_MoPA-Asym._SAC_Push_1800_rl.SawyerPushObstacle-v0.08.24.14.33.Asymmetric-MoPA-SAC.0',
        'Rebuttal_MoPA-Asym._SAC_Push_200_rl.SawyerPushObstacle-v0.08.20.09.12.Asymmetric-MoPA-SAC.0',
        'Rebuttal_MoPA-Asym._SAC_Push_2320_rl.SawyerPushObstacle-v0.08.24.10.42.Asymmetric-MoPA-SAC.0',
        'Rebuttal_MoPA-Asym._SAC_Push_500_rl.SawyerPushObstacle-v0.08.21.07.42.Asymmetric-MoPA-SAC.0',
    ],
    "Asym. SAC": [
        'Rebuttal_Asym._SAC_Push_1234_bumbling-frog-628',
        'Rebuttal_Asym._SAC_Push_200_laced-puddle-629',
        'Rebuttal_Asym._SAC_Push_500_flowing-surf-630',
        'Rebuttal_Asym._SAC_Push_1800_pretty-wave-683',
        'Rebuttal_Asym._SAC_Push_2320_feasible-sunset-682',
    ],
}

# choosing Rebuttal_Asym._SAC_Assembly_200_comfy-violet-635 because it has 3m env. steps
line_labels = {
    # "BC-Visual": ['Rebuttal_Asym._SAC_Assembly_200_comfy-violet-635'],
}

line_colors = {
    'Ours': 'C0',
    'MoPA Asym. SAC': 'C4',
    'Asym. SAC': 'C5',
}

mopa_curve = {
    "Ours": 'Rebuttal_MoPA_RL_Push_1234_rl.SawyerPushObstacle-v0.08.24.01.35.MoPA-SAC.0',
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
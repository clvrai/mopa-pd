filename_prefix = 'Rebuttal-SawyerPush-ADR-BC-Smoothing'
xlabel = 'Environment steps (2M)'
ylabel = "Average Discounted Rewards"
max_step = 2000000
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
    "Ours (w/o BC smoothing)": [
        'Rebuttal_Ours(wo_bc_smoothing)_Push_1234_lucky-waterfall-663',
        'Rebuttal_Ours(wo_bc_smoothing)_Push_200_fanciful-star-664',
        'Rebuttal_Ours(wo_bc_smoothing)_Push_2320_prime-meadow-693',
        'Rebuttal_Ours(wo_bc_smoothing)_Push_500_light-wave-665',
        'Rebuttal_Ours(wo_bc_smoothing)_Push_1800_desert-eon-673',
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
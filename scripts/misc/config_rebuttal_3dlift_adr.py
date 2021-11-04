filename_prefix = 'Rebuttal-SawyerLift-ADR'
xlabel = 'Environment steps (3M)'
ylabel = "Average Discounted Rewards"
mopa_cutoff_step = 1000000
others_cutoff_step = 2000000
max_step = 3000000
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
    "CoL": [
        'Rebuttal_CoL_MoPA_Lift_1234_stoic-fire-187',
        'Rebuttal_CoL_MoPA_Lift_200_wandering-waterfall-188',
        'Rebuttal_CoL_MoPA_Lift_2320_desert-flower-696',
        'Rebuttal_CoL_MoPA_Lift_500_expert-brook-189',
        'Rebuttal_CoL_MoPA_Lift_1800_usual-cloud-224',
    ],
    "CoL(w BC smoothing)": [
        'Rebuttal_CoL_Lift_1234_scarlet-yogurt-681',
        'Rebuttal_CoL_Lift_200_lunar-sun-683',
        'Rebuttal_CoL_Lift_2320_lyric-frog-689',
        'Rebuttal_CoL_Lift_500_scarlet-eon-685',
        'Rebuttal_CoL_Lift_1800_astral-night-671',
    ],
    "MoPA Asym. SAC": [
        'Rebuttal_MoPA-Asym._SAC_Lift_1234_rl.SawyerLiftObstacle-v0.08.21.07.52.Asymmetric-MoPA-SAC.0',
        'Rebuttal_MoPA-Asym._SAC_Lift_200_rl.SawyerLiftObstacle-v0.08.22.01.30.Asymmetric-MoPA-SAC.0',
        'Rebuttal_MoPA-Asym._SAC_Lift_2320_rl.SawyerLiftObstacle-v0.08.28.15.22.Asymmetric-MoPA-SAC.0',
        'Rebuttal_MoPA-Asym._SAC_Lift_500_rl.SawyerLiftObstacle-v0.08.22.01.35.Asymmetric-MoPA-SAC.0',
        'Rebuttal_MoPA-Asym._SAC_Lift_1800_rl.SawyerLiftObstacle-v0.08.30.07.28.Asymmetric-MoPA-SAC.0',
    ],
    "Asym. SAC": [
        'Rebuttal_Asym._SAC_Lift_1234_fast-bee-631',
        'Rebuttal_Asym._SAC_Lift_200_swift-smoke-632',
        'Rebuttal_Asym._SAC_Lift_500_visionary-dream-633',
        'Rebuttal_Asym._SAC_Lift_2320_MyMachine_rural-valley-691',
        'Rebuttal_Asym._SAC_Lift_1800_zany-wave-684',
    ],
}

# choosing Rebuttal_Asym._SAC_Assembly_200_comfy-violet-635 because it has 3m env. steps
line_labels = {
    "BC-Visual": ['Rebuttal_Asym._SAC_Assembly_200_comfy-violet-635'],
}

line_colors = {
    'Ours': 'C0',
    'Ours (w/o BC smoothing)': 'C1',
    'CoL': 'C2',
    'CoL(w BC smoothing)': 'C3',
    'MoPA Asym. SAC': 'C4',
    'Asym. SAC': 'C5',
    'BC-Visual': 'C6',
}

mopa_curve = {
    "Ours": 'Rebuttal_MoPA_RL_Lift_2000_rl.SawyerLiftObstacle-v0.08.27.MoPA-SAC.2000',
    "Ours (w/o BC smoothing)": 'Rebuttal_MoPA_RL_Lift_2000_rl.SawyerLiftObstacle-v0.08.27.MoPA-SAC.2000',
    "CoL": 'Rebuttal_MoPA_RL_Lift_2000_rl.SawyerLiftObstacle-v0.08.27.MoPA-SAC.2000',
    "CoL(w BC smoothing)": 'Rebuttal_MoPA_RL_Lift_2000_rl.SawyerLiftObstacle-v0.08.27.MoPA-SAC.2000',
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
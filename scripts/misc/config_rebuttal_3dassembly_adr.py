filename_prefix = 'Rebuttal-SawyerAssembly-ADR'
xlabel = 'Environment steps (3M)'
ylabel = "Average Discounted Rewards"
mopa_cutoff_step = 1000000
others_cutoff_step = 2000000
max_step = 3000000
max_y_axis_value = 110
legend = True
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
    "CoL": [
        'Rebuttal_CoL_MoPA_Assembly_1234_upbeat-blaze-184',
        'Rebuttal_CoL_MoPA_Assembly_200_glorious-vortex-185',
        'Rebuttal_CoL_MoPA_Assembly_2320_rosy-yogurt-692',
        'Rebuttal_CoL_MoPA_Assembly_500_effortless-armadillo-186',
        'Rebuttal_CoL_MoPA_Assembly_1800_stilted-armadillo-225',
    ],
    "CoL(w BC smoothing)": [
        'Rebuttal_CoL_Assembly_1234_true-moon-687',
        'Rebuttal_CoL_Assembly_200_floral-dust-686',
        'Rebuttal_CoL_Assembly_2320_radiant-terrain-690',
        'Rebuttal_CoL_Assembly_500_glowing-meadow-678',
        'Rebuttal_CoL_Assembly_1800_toasty-frost-672',
    ],
    "MoPA Asym. SAC": [
        'Rebuttal_MoPA-Asym._SAC_Assembly_1234_rl.SawyerAssemblyObstacle-v0.08.22.04.49.Asymmetric-MoPA-SAC.0',
        'Rebuttal_MoPA-Asym._SAC_Assembly_200_rl.SawyerAssemblyObstacle-v0.08.22.08.22.Asymmetric-MoPA-SAC.0',
        'Rebuttal_MoPA-Asym._SAC_Assembly_2320_rl.SawyerAssemblyObstacle-v0.08.25.07.34.Asymmetric-MoPA-SAC.0',
        'Rebuttal_MoPA-Asym._SAC_Assembly_500_rl.SawyerAssemblyObstacle-v0.08.22.08.24.Asymmetric-MoPA-SAC.0',
        'Rebuttal_MoPA-Asym._SAC_Assembly_1800_rl.SawyerAssemblyObstacle-v0.08.30.07.30',
    ],
    "Asym. SAC": [
        'Rebuttal_Asym._SAC_Assembly_1234_brisk-fire-634',
        'Rebuttal_Asym._SAC_Assembly_200_comfy-violet-635',
        'Rebuttal_Asym._SAC_Assembly_500_winter-gorge-636',
        'Rebuttal_Asym._SAC_Assembly_2320_MyMachine_mild-dew-691',
        'Rebuttal_Asym._SAC_Assembly_1800_copper-voice-687',
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
    "Ours": 'Rebuttal_MoPA_RL_Assembly_200_rl.SawyerAssemblyObstacle-v0.08.26.MoPA-SAC.200',
    "Ours (w/o BC smoothing)": 'Rebuttal_MoPA_RL_Assembly_200_rl.SawyerAssemblyObstacle-v0.08.26.MoPA-SAC.200',
    "CoL": 'Rebuttal_MoPA_RL_Assembly_200_rl.SawyerAssemblyObstacle-v0.08.26.MoPA-SAC.200',
    "CoL(w BC smoothing)": 'Rebuttal_MoPA_RL_Assembly_200_rl.SawyerAssemblyObstacle-v0.08.26.MoPA-SAC.200',
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
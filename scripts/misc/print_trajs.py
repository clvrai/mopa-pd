import numpy as np
import matplotlib.pyplot as plt

what_to_plot = 'push' # push, lift, assembly

if what_to_plot == 'push':
    with open('bc_sawyer_push_eef_positions.npy', 'rb') as f:
        bc_trajs = np.load(f, allow_pickle=True)
    with open('mopa_rl_sawyer_push_eef_positions.npy', 'rb') as f:
        mopa_trajs = np.load(f, allow_pickle=True)
elif what_to_plot == 'lift':
    with open('bc_sawyer_lift_eef_positions.npy', 'rb') as f:
        bc_trajs = np.load(f, allow_pickle=True)
    with open('mopa_rl_sawyer_lift_eef_positions.npy', 'rb') as f:
        mopa_trajs = np.load(f, allow_pickle=True)
elif what_to_plot == 'assembly':
    with open('bc_sawyer_assembly_eef_positions.npy', 'rb') as f:
        bc_trajs = np.load(f, allow_pickle=True)
    with open('mopa_rl_sawyer_assembly_eef_positions.npy', 'rb') as f:
        mopa_trajs = np.load(f, allow_pickle=True)
else:
    raise NotImplementedError


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# scatter plot
# ax.scatter(
#     bc_trajs[:, 0],
#     bc_trajs[:, 1],
#     bc_trajs[:, 2],
#     s=5,
#     c='tab:blue',
#     label='BC trajectory',
#     alpha=0.5,
# )
# ax.scatter(
#     mopa_trajs[:, 0],
#     mopa_trajs[:, 1],
#     mopa_trajs[:, 2],
#     s=5,
#     c='tab:green',
#     label='MoPA-RL trajectory',
#     alpha=0.2,
# )



# Sawyer Push start eef
# traj_1: array([1.00103276, 0.19048722, 1.39140842])
# traj_2: array([1.00597903, 0.16782076, 1.39077113])
# traj_3: array([1.0076476 , 0.16110221, 1.38615579])

# Sawyer Lift start eef
# traj_1: array([1.05998562, 0.0033063 , 1.3634719 ])
# traj_2: array([ 1.05770587, -0.02708154,  1.36144588])
# traj_3: array([ 1.04828012, -0.04477079,  1.31336254])

# Sawyer Assembly start eef
# traj_1: array([0.81968783, 0.58831886, 1.07969404])
# traj_2: array([0.83076404, 0.56423978, 1.06143582])
# traj_3: array([0.82917197, 0.55269846, 1.03260606])

# Manually inserting start position because it wasn't included when trajs were generated
if what_to_plot == 'push':
    bc_trajs[0] = np.insert(bc_trajs[0], 0, [1.00103276, 0.19048722, 1.39140842], axis=0)
    bc_trajs[1] = np.insert(bc_trajs[1], 0, [1.00597903, 0.16782076, 1.39077113], axis=0)
    bc_trajs[2] = np.insert(bc_trajs[2], 0, [1.0076476 , 0.16110221, 1.38615579], axis=0)
    mopa_trajs[0] = np.insert(mopa_trajs[0], 0, [1.00103276, 0.19048722, 1.39140842], axis=0)
    mopa_trajs[1] = np.insert(mopa_trajs[1], 0, [1.00597903, 0.16782076, 1.39077113], axis=0)
    mopa_trajs[2] = np.insert(mopa_trajs[2], 0, [1.0076476 , 0.16110221, 1.38615579], axis=0)
elif what_to_plot == 'lift':
    bc_trajs[0] = np.insert(bc_trajs[0], 0, [1.05998562, 0.0033063 , 1.3634719], axis=0)
    bc_trajs[1] = np.insert(bc_trajs[1], 0, [1.05770587, -0.02708154,  1.36144588], axis=0)
    bc_trajs[2] = np.insert(bc_trajs[2], 0, [1.04828012, -0.04477079,  1.31336254], axis=0)
    mopa_trajs[0] = np.insert(mopa_trajs[0], 0, [1.05998562, 0.0033063 , 1.3634719], axis=0)
    mopa_trajs[1] = np.insert(mopa_trajs[1], 0, [1.05770587, -0.02708154,  1.36144588], axis=0)
    mopa_trajs[2] = np.insert(mopa_trajs[2], 0, [1.04828012, -0.04477079,  1.31336254], axis=0)
elif what_to_plot == 'assembly':
    bc_trajs[0] = np.insert(bc_trajs[0], 0, [0.81968783, 0.58831886, 1.07969404], axis=0)
    bc_trajs[1] = np.insert(bc_trajs[1], 0, [0.83076404, 0.56423978, 1.06143582], axis=0)
    bc_trajs[2] = np.insert(bc_trajs[2], 0, [0.82917197, 0.55269846, 1.03260606], axis=0)
    mopa_trajs[0] = np.insert(mopa_trajs[0], 0, [0.81968783, 0.58831886, 1.07969404], axis=0)
    mopa_trajs[1] = np.insert(mopa_trajs[1], 0, [0.83076404, 0.56423978, 1.06143582], axis=0)
    mopa_trajs[2] = np.insert(mopa_trajs[2], 0, [0.82917197, 0.55269846, 1.03260606], axis=0)

# line plot
i = 0
plot_once = True
for bc_trajs_per_eps, mopa_trajs_per_eps in zip(bc_trajs, mopa_trajs):
    if i != 1:
        i += 1
        continue

    if plot_once:
        ax.plot(
            bc_trajs_per_eps[:, 0],
            bc_trajs_per_eps[:, 1],
            bc_trajs_per_eps[:, 2],
            c='tab:blue',
            label='BC trajectory',
            alpha=0.8,
        )
        ax.plot(
            mopa_trajs_per_eps[:, 0],
            mopa_trajs_per_eps[:, 1],
            mopa_trajs_per_eps[:, 2],
            c='tab:orange',
            label='MoPA-RL trajectory',
            alpha=1,
        )
        ax.scatter(bc_trajs_per_eps[0, 0], bc_trajs_per_eps[0, 1], bc_trajs_per_eps[0, 2], marker='o', color='tab:green', zorder=1)
        ax.scatter(bc_trajs_per_eps[-1, 0], bc_trajs_per_eps[-1, 1], bc_trajs_per_eps[-1, 2], marker='x', color='tab:red', zorder=1)
        ax.scatter(mopa_trajs_per_eps[0, 0], mopa_trajs_per_eps[0, 1], mopa_trajs_per_eps[0, 2], marker='o', color='tab:green', zorder=1)
        ax.scatter(mopa_trajs_per_eps[-1, 0], mopa_trajs_per_eps[-1, 1], mopa_trajs_per_eps[-1, 2], marker='x', color='tab:red', zorder=1)
    else:
        ax.plot(
            bc_trajs_per_eps[:, 0],
            bc_trajs_per_eps[:, 1],
            bc_trajs_per_eps[:, 2],
            c='tab:blue',
            alpha=0.8,
        )
        ax.plot(
            mopa_trajs_per_eps[:, 0],
            mopa_trajs_per_eps[:, 1],
            mopa_trajs_per_eps[:, 2],
            c='tab:orange',
            alpha=1,
        )
        ax.scatter(bc_trajs_per_eps[0, 0], bc_trajs_per_eps[0, 1], bc_trajs_per_eps[0, 2], marker='o', color='tab:green', zorder=1)
        ax.scatter(bc_trajs_per_eps[-1, 0], bc_trajs_per_eps[-1, 1], bc_trajs_per_eps[-1, 2], marker='x', color='tab:red', zorder=1)
        ax.scatter(mopa_trajs_per_eps[0, 0], mopa_trajs_per_eps[0, 1], mopa_trajs_per_eps[0, 2], marker='o', color='tab:green', zorder=1)
        ax.scatter(mopa_trajs_per_eps[-1, 0], mopa_trajs_per_eps[-1, 1], mopa_trajs_per_eps[-1, 2], marker='x', color='tab:red', zorder=1)
    plot_once = False
    i += 1

ax.legend(fontsize=14)

# def set_axes_equal(ax):
#     x_limits = ax.get_xlim3d()
#     y_limits = ax.get_ylim3d()
#     z_limits = ax.get_zlim3d()

#     x_range = abs(x_limits[1] - x_limits[0])
#     x_middle = np.mean(x_limits)
#     y_range = abs(y_limits[1] - y_limits[0])
#     y_middle = np.mean(y_limits)
#     z_range = abs(z_limits[1] - z_limits[0])
#     z_middle = np.mean(z_limits)

#     # The plot bounding box is a sphere in the sense of the infinity
#     # norm, hence I call half the max range the plot radius.
#     plot_radius = 0.5 * max([x_range, y_range, z_range])

#     ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
#     ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
#     ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# set_axes_equal(ax)

ax.dist = 9


if what_to_plot == 'push':
    ax.set_xlim3d([0.38, 1.02])
    ax.set_ylim3d([-0.35, 0.25])
    ax.set_zlim3d([0.7, 1.42])
    plt.savefig('bc_mopa_trajs_sawyer_push.png')
elif what_to_plot == 'lift':
    ax.set_xlim3d([0.4, 1.2])
    ax.set_ylim3d([-0.45, 0])
    ax.set_zlim3d([0.9, 1.4])
    plt.savefig('bc_mopa_trajs_sawyer_lift.png')
elif what_to_plot == 'assembly':
    ax.set_xlim3d([0.3, 0.9])
    ax.set_ylim3d([-0.05, 0.6])
    ax.set_zlim3d([0, 1.8])
    plt.savefig('bc_mopa_trajs_sawyer_assembly.png')
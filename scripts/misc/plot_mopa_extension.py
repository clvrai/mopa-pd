from collections import namedtuple, defaultdict
import subprocess

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
import wandb
import pandas as pd


matplotlib.rcParams["pdf.fonttype"] = 42  # Important!!! Remove Type 3 fonts


def save_fig(file_name, file_format="pdf", tight=True, **kwargs):
    if tight:
        plt.tight_layout()
    file_name = "{}.{}".format(file_name, file_format).replace(" ", "-")
    plt.savefig(file_name, format=file_format, dpi=1000, **kwargs)


def draw_line(
    log,
    method,
    avg_step=3,
    mean_std=False,
    max_step=None,
    max_y=None,
    x_scale=1.0,
    ax=None,
    color="C0",
    smooth_steps=10,
    num_points=50,
    line_style="-",
    marker=None,
    no_fill=False,
    smoothing_weight=0.0,
    smoothing=True,
    limit_y_max=False,
    limit_y_max_value=None,
    mopa_curve=None,
    mopa_cutoff_step_scaled=0,
    plot_first=True,
):
    steps = {}
    values = {}
    max_step = max_step * x_scale
    seeds = log.keys()
    is_line = True

    for seed in seeds:
        step = np.array(log[seed].steps)
        value = np.array(log[seed].values)

        if not np.isscalar(log[seed].values):
            is_line = False

            # filter NaNs
            for i in range(len(value)):
                if np.isnan(value[i]):
                    value[i] = 0 if i == 0 else value[i - 1]

        if max_step:
            max_step = min(max_step, step[-1])
        else:
            max_step = step[-1]

        steps[seed] = step
        values[seed] = value

    if is_line:
        y_data = [values[seed] for seed in seeds]
        std_y = np.std(y_data)
        avg_y = np.mean(y_data)
        min_y = np.min(y_data)
        max_y = np.max(y_data)

        l = ax.axhline(
            y=avg_y, label=method, color=color, linestyle=line_style, marker=marker
        )
        ax.axhspan(
            avg_y - std_y,  # max(avg_y - std_y, min_y),
            avg_y + std_y,  # min(avg_y + std_y, max_y),
            color=color,
            alpha=0.1,
        )
        return l, min_y, max_y

    # exponential moving average smoothing
    for seed in seeds:
        last = values[seed][:10].mean()  # First value in the plot (first timestep)
        smoothed = list()
        for point in values[seed]:
            smoothed_val = (
                last * smoothing_weight + (1 - smoothing_weight) * point
            )  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value
        values[seed] = smoothed

    # cap all sequences to max number of steps
    data = []
    for seed in seeds:
        for i in range(len(steps[seed])):
            if steps[seed][i] <= max_step:
                data.append((steps[seed][i], values[seed][i]))
    data.sort()
    x_data = []
    y_data = []
    for step, value in data:
        x_data.append(step)
        y_data.append(value)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    if limit_y_max:
        y_data[y_data > limit_y_max_value] = limit_y_max_value

    # may need to enable/disable depending on what you want to graph: temporarily disable because not needed for ADR figures
    # y_data = y_data * 1e6

    min_y = np.min(y_data)
    max_y = np.max(y_data)
    # l = sns.lineplot(x=x_data, y=y_data)
    # return l, min_y, max_y

    plot_first = False
    if not smoothing:
        n = len(x_data)
        avg_step = int(n // num_points)

        x_data = x_data[: n // avg_step * avg_step].reshape(-1, avg_step)
        y_data = y_data[: n // avg_step * avg_step].reshape(-1, avg_step)
        avg_x, avg_y = np.max(x_data, axis=1), np.max(y_data, axis=1)
        l = ax.plot(avg_x, avg_y, label=method)
        plt.setp(l, linewidth=2, color=color, linestyle=line_style, marker=marker)
        return l, min_y, max_y, plot_first

    # filling
    if not no_fill:
        n = len(x_data)
        avg_step = int(n // num_points)

        x_data = x_data[: n // avg_step * avg_step].reshape(-1, avg_step)
        y_data = y_data[: n // avg_step * avg_step].reshape(-1, avg_step)

        std_y = np.std(y_data, axis=1)

        avg_x, avg_y = np.mean(x_data, axis=1), np.mean(y_data, axis=1)
    else:
        avg_x, avg_y = x_data, y_data

    # manually setting value right after 1.0 to be 1.01
    avg_x[np.where(avg_x > mopa_cutoff_step_scaled)[0][0]] = 1.01
    # manually setting the last value
    avg_x[-1] = 3.0
    # avg_x[-1] = 1.2
    
    # manually setting the first value to start at x = 0
    avg_x[0] = 0.0

    if mopa_curve is not None and method in mopa_curve:
        # only subsampling smoothing on the second half of curve
        other_avg_y = avg_y[avg_x > mopa_cutoff_step_scaled]
        other_avg_x = avg_x[avg_x > mopa_cutoff_step_scaled]
        n = len(other_avg_x)
        ns = smooth_steps
        other_avg_x = other_avg_x[: n // ns * ns].reshape(-1, ns).mean(axis=1)
        other_avg_y = other_avg_y[: n // ns * ns].reshape(-1, ns).mean(axis=1)
        if not no_fill:
            std_y = std_y[: len(avg_x) // ns * ns].reshape(-1, ns).mean(axis=1)
            std_y = std_y[avg_x > mopa_cutoff_step_scaled]
        if not no_fill:
            ax.fill_between(
                other_avg_x,
                other_avg_y - std_y,  # np.clip(avg_y - std_y, 0, max_y),
                other_avg_y + std_y,  # np.clip(avg_y + std_y, 0, max_y),
                alpha=0.1,
                color=color,
            )
        # subsampling smoothing on entire curve
        n = len(avg_x)
        ns = smooth_steps
        avg_x = avg_x[: n // ns * ns].reshape(-1, ns).mean(axis=1)
        avg_y = avg_y[: n // ns * ns].reshape(-1, ns).mean(axis=1)
    else:
        # subsampling smoothing
        n = len(avg_x)
        ns = smooth_steps
        avg_x = avg_x[: n // ns * ns].reshape(-1, ns).mean(axis=1)
        avg_y = avg_y[: n // ns * ns].reshape(-1, ns).mean(axis=1)
        if not no_fill:
            std_y = std_y[: n // ns * ns].reshape(-1, ns).mean(axis=1)

        if not no_fill:
            ax.fill_between(
                avg_x,
                avg_y - std_y,  # np.clip(avg_y - std_y, 0, max_y),
                avg_y + std_y,  # np.clip(avg_y + std_y, 0, max_y),
                alpha=0.1,
                color=color,
            )

    # horizontal line
    # if "SAC" in method:
    #     l = ax.axhline(
    #         y=avg_y[-1], xmin=0.1, xmax=1.0, color=color, linestyle="--", marker=marker
    #     )
    #     plt.setp(l, linewidth=2, color=color, linestyle="--", marker=marker)

    if mopa_curve is not None and plot_first and method in mopa_curve:
        mopa_avg_y = avg_y[avg_x <= mopa_cutoff_step_scaled]
        mopa_avg_x = avg_x[avg_x <= mopa_cutoff_step_scaled]
        mopa_l = ax.plot(mopa_avg_x, mopa_avg_y, label='MoPA-RL')
        plt.setp(mopa_l, linewidth=2, color='C7', linestyle='dotted', marker=marker)
        avg_y = avg_y[avg_x > mopa_cutoff_step_scaled]
        avg_x = avg_x[avg_x > mopa_cutoff_step_scaled]
        plt.axvline(x=1.0, color='C8', linestyle='--')
    elif mopa_curve is not None and not plot_first and method in mopa_curve:
        # already plot the first Mo-RAL line, so only plot the other curve's line
        avg_y = avg_y[avg_x > mopa_cutoff_step_scaled]
        avg_x = avg_x[avg_x > mopa_cutoff_step_scaled]

    l = ax.plot(avg_x, avg_y, label=method)
    plt.setp(l, linewidth=2, color=color, linestyle=line_style, marker=marker)

    return l, min_y, max_y, plot_first


def draw_graph(
    plot_logs,
    line_logs,
    method_names=None,
    title=None,
    xlabel="Step",
    ylabel="Success",
    legend=False,
    mean_std=False,
    min_step=0,
    max_step=None,
    min_y=None,
    max_y=None,
    num_y_tick=5,
    smooth_steps=10,
    num_points=50,
    no_fill=False,
    num_x_tick=5,
    legend_loc=2,
    markers=None,
    smoothing_weight=0.0,
    file_name=None,
    line_styles=None,
    line_colors=None,
    smoothing=True,
    limit_y_max=False,
    limit_y_max_value=None,
    mopa_curve=None,
    mopa_cutoff_step_scaled=0,
):
    # if legend:
    #     fig, ax = plt.subplots(figsize=(15, 5))
    # else:
    #     fig, ax = plt.subplots(figsize=(5, 4))
    fig, ax = plt.subplots(figsize=(5, 4))

    max_value = -np.inf
    min_value = np.inf

    if method_names is None:
        method_names = list(plot_logs.keys()) + list(line_logs.keys())

    lines = []
    num_colors = len(method_names)
    two_lines_per_method = False
    if "Pick" in method_names[0] or "Attach" in method_names[0]:
        two_lines_per_method = True
        num_colors = len(method_names) / 2

    plot_first = True
    for idx, method_name in enumerate(method_names):
        if method_name in plot_logs.keys():
            log = plot_logs[method_name]
        else:
            log = line_logs[method_name]

        seeds = log.keys()
        if len(seeds) == 0:
            continue

        color = (
            line_colors[method_name] if line_colors else "C%d" % (num_colors - idx - 1)
        )
        line_style = line_styles[method_name] if line_styles else "-"

        l_, min_, max_, plot_first = draw_line(
            log,
            method_name,
            mean_std=mean_std,
            max_step=max_step,
            max_y=max_y,
            x_scale=1.0,
            ax=ax,
            color=color,
            smooth_steps=smooth_steps,
            num_points=num_points,
            line_style=line_style,
            no_fill=no_fill,
            smoothing_weight=smoothing_weight[idx]
            if isinstance(smoothing_weight, list)
            else smoothing_weight,
            marker=markers[idx] if isinstance(markers, list) else markers,
            smoothing=smoothing,
            limit_y_max=limit_y_max,
            limit_y_max_value=limit_y_max_value,
            mopa_curve=mopa_curve,
            mopa_cutoff_step_scaled=mopa_cutoff_step_scaled,
            plot_first=plot_first
        )
        # lines += l_
        max_value = max(max_value, max_)
        min_value = min(min_value, min_)

    if min_y == None:
        min_y = int(min_value - 1)
    if max_y == None:
        max_y = max_value
        # max_y = int(max_value + 1)

    # may need to enable/disable depending on what you want to graph: manually printing y-axis ticks
    # plt.yticks([1.3, 1.4, 1.6, 2.0], fontsize=12)

    # y-axis tick (belows are commonly used settings)
    if max_y == 1:
        plt.yticks(np.arange(min_y, max_y + 0.1, 0.2), fontsize=12)
    else:
        if max_y > 1:
            plt.yticks(
                np.arange(min_y, max_y + 0.01, (max_y - min_y) / num_y_tick),
                fontsize=12,
            )  # make this 4 for kitchen
        elif max_y > 0.8:
            plt.yticks(np.arange(0, 1.0, 0.2), fontsize=12)
        elif max_y > 0.5:
            plt.yticks(np.arange(0, 0.8, 0.2), fontsize=12)
        elif max_y > 0.3:
            plt.yticks(np.arange(0, 0.5, 0.1), fontsize=12)
        elif max_y > 0.2:
            plt.yticks(np.arange(0, 0.4, 0.1), fontsize=12)
        else:
            y_ticks = np.arange(min_y, max_y, (max_y+min_y)/5)
            y_ticks = np.append(y_ticks, max_y)
            plt.yticks(y_ticks, fontsize=12)

    # x-axis tick
    plt.xticks(
        np.round(
            np.arange(min_step, max_step + 0.1, (max_step - min_step) / num_x_tick), 2
        ),
        fontsize=12,
    )

    # background grid
    ax.grid(b=True, which="major", color="lightgray", linestyle="--")

    # axis titles
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    # set axis range
    ax.set_xlim(min_step, max_step)
    if max_y > 0.2:
        ax.set_ylim(bottom=-0.01, top=max_y + 0.01)  # use -0.01 to print ytick 0

    # print legend
    if legend:
        if isinstance(legend_loc, tuple):
            print("print legend outside of frame")
            leg = plt.legend(fontsize=15, bbox_to_anchor=legend_loc, ncol=6)
        else:
            # leg = plt.legend(fontsize=11, loc=legend_loc) # original
            leg = plt.legend(fontsize=9, loc=legend_loc)

    #         for line in leg.get_lines():
    #             line.set_linewidth(2)
    # labs = [l.get_label() for l in lines]
    # plt.legend(lines, labs, fontsize='small', loc=2)

    # print title
    if title:
        plt.title(title, y=1.00, fontsize=16)

    # save plot to file
    if file_name:
        save_fig(file_name)


def build_logs(
    methods_label,
    runs,
    data_key="train_ep/episode_success",
    x_scale=1000000,
    op=None,
    max_step=1000000,
    y_value=None,
    build_log_from_multiple_keys=False,
    mopa_cutoff_step=0,
    mopa_curve=None,
    others_cutoff_step=0,
):
    Log = namedtuple("Log", ["values", "steps"])
    logs = defaultdict(dict)
    method_index = 0
    for run_name in methods_label.keys():
        for i, seed_path in enumerate(methods_label[run_name]):
            found_path = False
            for run in runs:
                if run.name == seed_path:
                    data = run.history(samples=200000)
                    if build_log_from_multiple_keys:
                        cur_data_key = data_key[method_index]
                    else:
                        cur_data_key = data_key

                    values = data[cur_data_key]
                    values = values.fillna(0) # make sure there is no NaN
                    if y_value is not None:
                        # BC policies
                        values[:] = y_value
                    if cur_data_key == 'Total Success':
                        # display validation success rate for BC-Visual
                        values = values / 30.0

                    # use in rebuttal to present first 1m env. steps from MoPA-RL and the rest from other methods
                    if mopa_curve is not None and run_name in mopa_curve:
                        steps =  data["_step"][data["_step"] <= others_cutoff_step]
                        second_half_values = values[steps.index]
                        has_mopa_run_found = False
                        for mopa_run in runs:
                            if mopa_run.name == mopa_curve[run_name]:
                                # get first half values
                                mopa_data = mopa_run.history(samples=200000)
                                mopa_values = mopa_data[cur_data_key].fillna(0)
                                mopa_steps = mopa_data["_step"][mopa_data["_step"] < mopa_cutoff_step]
                                first_half_values = mopa_values[mopa_steps.index]

                                # correct second_half indicies
                                starting_index_for_second_half = first_half_values.index[-1] + 1
                                second_half_values.index = second_half_values.index + starting_index_for_second_half
                                values = pd.concat([first_half_values, second_half_values])

                                # combine first_half and second_half steps
                                starting_index_for_second_half_steps = mopa_steps.index[-1] + 1
                                steps.index = steps.index + starting_index_for_second_half_steps
                                steps = steps + mopa_steps.iloc[-1]
                                steps = pd.concat([mopa_steps, steps])
                                steps = steps / x_scale
                                has_mopa_run_found = True
                        if not has_mopa_run_found:
                            print("Could not find MoPA-RL run at ", mopa_curve[run_name])
                    else:
                        # if a method does not require MoPA-RL (i.e. train from scratch)
                        # cap out at specified env step
                        steps =  (data["_step"][data["_step"] <= max_step]) / x_scale
                        values = values[data["_step"] <= max_step]

                    logs[run_name][i] = Log(values, steps)
                    print(run_name, i, run, len(steps))
                    found_path = True
            if not found_path:
                # raise ValueError("Could not find run: {}".format(seed_path))
                print("Could not find run: {}".format(seed_path))
        method_index += 1
    return logs


def build_logs_pick_attach(
    methods_label, runs, x_scale=1000000, data_key=None, op=None, exclude_runs=[],
):
    Log = namedtuple("Log", ["values", "steps"])
    logs = defaultdict(dict)
    for method_name, method_runs in methods_label.items():
        for i, seed_path in enumerate(method_runs):
            if seed_path in exclude_runs:
                print("Exclude run: {}".format(seed_path))
                continue

            found_path = False
            for run in runs:
                if run.name == seed_path:
                    data = run.history(samples=10000)
                    pick_values = (data[data_key + "phase"] >= 4) + (
                        data[data_key + "phase"] >= 12
                    )
                    attach_values = data[data_key + "success_reward"] / 100
                    steps = data["_step"] / x_scale
                    if op == "max":
                        pick_values = max(pick_values)
                        attach_values = max(attach_values)
                    logs[method_name + "-Pick"][i] = Log(pick_values, steps)
                    logs[method_name + "-Attach"][i] = Log(attach_values, steps)
                    print(method_name, i, run, len(steps))
                    found_path = True

            if not found_path:
                # raise ValueError("Could not find run: {}".format(seed_path))
                print("Could not find run: {}".format(seed_path))
    return logs

def plot_methods():
    print("** start plot")
    print()

    ghost_script = True

    # default values
    wandb_api_path = 'corl_experiments/mopa-rl-image'
    x_scale = 1000000
    min_y_axis_value = 0
    divide_max_step_by_1mill = True
    smoothing = True
    smoothing_weight = 0.99
    num_points = 100
    legend_loc = 'upper left'
    build_log_from_multiple_keys = False
    limit_y_max = False
    limit_y_max_value = None
    mopa_curve = None
    mopa_cutoff_step = 0
    mopa_cutoff_step_scaled = 0
    others_cutoff_step = 0

    ##### Average Discounted Rewards configs
    # from config_3dpush_adr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_3dlift_adr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_3dassembly_adr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_rebuttal_3dpush_adr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step
    # from config_rebuttal_3dlift_adr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step
    # from config_rebuttal_3dassembly_adr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step
    
    # from config_rebuttal_3dpush_adr_bc_smoothing import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_rebuttal_3dlift_adr_bc_smoothing import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_rebuttal_3dassembly_adr_bc_smoothing import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors

    # from config_rebuttal_3dpush_adr_baselines import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step
    # from config_rebuttal_3dlift_adr_baselines import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step
    # from config_rebuttal_3dassembly_adr_baselines import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step

    # from config_rebuttal_3dpush_adr_baselines1 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step
    # from config_rebuttal_3dpush_adr_baselines2 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step

    # from config_rebuttal_3dlift_adr_baselines1 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step
    # from config_rebuttal_3dlift_adr_baselines2 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step

    # from config_rebuttal_3dassembly_adr_baselines1 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step
    # from config_rebuttal_3dassembly_adr_baselines2 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step

    # from config_rebuttal_3dpush_adr_sota import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step
    # from config_rebuttal_3dlift_adr_sota import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step
    # from config_rebuttal_3dassembly_adr_sota import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, mopa_curve, mopa_cutoff_step, others_cutoff_step


    ##### Average Success Rate configs
    # from config_3dpush_asr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_3dlift_asr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_3dassembly_asr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors

    ##### Ablations configs
    # from config_abl_3dassembly_alpha_asr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_abl_3dlift_alpha_asr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_abl_3dpush_alpha_asr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors

    # from config_abl_3dpush_alpha_asr_line1 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_abl_3dpush_alpha_asr_line2 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_abl_3dpush_alpha_asr_line3 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors

    # from config_abl_3dlift_alpha_asr_line1 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_abl_3dlift_alpha_asr_line2 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_abl_3dlift_alpha_asr_line3 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors

    # from config_abl_3dassembly_alpha_asr_line1 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_abl_3dassembly_alpha_asr_line2 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_abl_3dassembly_alpha_asr_line3 import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors

    # from config_abl_3dpush_alpha_change import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, smoothing, smoothing_weight, legend_loc
    # from config_abl_3dlift_alpha_change import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, smoothing, smoothing_weight, legend_loc
    # from config_abl_3dassembly_alpha_change import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, smoothing, smoothing_weight, legend_loc

    # from config_3dpush_abl_init_asr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_3dlift_abl_init_asr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors
    # from config_3dassembly_abl_init_asr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors

    # from config_abl_3dassembly_optimal_bc_vsr import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, smoothing, smoothing_weight, legend_loc, wandb_api_path, num_points, x_scale, divide_max_step_by_1mill
    # from config_abl_3dassembly_optimal_bc_loss import filename_prefix, xlabel, ylabel, max_step, max_y_axis_value, legend, data_key, bc_y_value, plot_labels, line_labels, line_colors, smoothing, smoothing_weight, legend_loc, wandb_api_path, num_points, x_scale, divide_max_step_by_1mill, build_log_from_multiple_keys, limit_y_max, limit_y_max_value, min_y_axis_value

    if mopa_cutoff_step != 0 and others_cutoff_step != 0:
        # input check if using mopa-rl for first half of curve
        assert (others_cutoff_step + mopa_cutoff_step) == max_step
        mopa_cutoff_step_scaled = mopa_cutoff_step / 1000000

    if divide_max_step_by_1mill:
        max_step_divided = max_step / 1000000
    else:
        max_step_divided = max_step

    print("** load runs from wandb")
    api = wandb.Api()
    runs = api.runs(path=wandb_api_path)

    print("** load data from wandb")
    plot_logs = build_logs(
        plot_labels, runs, data_key=data_key, max_step=max_step, x_scale=x_scale, build_log_from_multiple_keys=build_log_from_multiple_keys, mopa_cutoff_step=mopa_cutoff_step, mopa_curve=mopa_curve, others_cutoff_step=others_cutoff_step
    )
    line_logs = build_logs(
        line_labels,
        runs,
        data_key=data_key,
        max_step=max_step,
        y_value=bc_y_value,
    )

    print("** draw graph")
    draw_graph(
        plot_logs,  # curved lines
        line_logs,  # straight line
        method_names=None,  # method names to plot with order
        title=None,  # figure title on top
        xlabel=xlabel,  # x-axis title
        ylabel=ylabel,  # y-axis title
        legend=legend,  # True if furniture == "three_blocks_peg" else False,
        legend_loc=legend_loc,  # (1.03, 0.73),
        max_step=max_step_divided,
        min_y=min_y_axis_value,
        max_y=max_y_axis_value,
        num_y_tick=5,
        smooth_steps=1,
        num_points=num_points,
        num_x_tick=4,  # 5,
        smoothing_weight=smoothing_weight,
        file_name=filename_prefix,
        line_colors=line_colors,
        smoothing=smoothing,
        limit_y_max=limit_y_max,
        limit_y_max_value=limit_y_max_value,
        mopa_curve=mopa_curve,
        mopa_cutoff_step_scaled=mopa_cutoff_step_scaled,
    )

    def gs_opt(filename):
        filename_reduced = filename.split(".")[-2] + "_reduced.pdf"
        gs = [
            "gs",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/default",  # Image resolution
            "-dNOPAUSE",  # No pause after each image
            "-dQUIET",  # Suppress output
            "-dBATCH",  # Automatically exit
            "-dDetectDuplicateImages=true",  # Embeds images used multiple times only once
            "-dCompressFonts=true",  # Compress fonts in the output (default)
            "-r150",
            # '-dEmbedAllFonts=false',
            # '-dSubsetFonts=true',           # Create font subsets (default)
            "-sOutputFile=" + filename_reduced,  # Save to temporary output
            filename,  # Input file
        ]

        subprocess.run(gs)  # Create temporary file
        # subprocess.run(['del', filename],shell=True)            # Delete input file
        # subprocess.run(['ren', filenameTmp,filename],shell=True) # Rename temporary to input file

    if ghost_script:
        gs_opt(filename_prefix + ".pdf")

if __name__ == "__main__":
    plot_methods()
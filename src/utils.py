import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np


def plot_line_search(func, func_name, line_search_results):
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.title("Convergence of function: " + func_name)
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(line_search_results)))
    # resort such that shorter path will appear above longer path
    line_search_results = sorted(line_search_results, key=lambda ls_res: len(ls_res), reverse=True)
    for ind, ls in enumerate(line_search_results):
        objective_values = ls.objective_values
        num_vals = len(objective_values)
        color = colors[ind]
        label = get_plot_label(ls)
        if num_vals <= 2:
            plt.scatter(np.arange(1, num_vals + 1), objective_values, label=label, linewidth=2, color=color)
        else:
            plt.plot(np.arange(1, num_vals + 1), objective_values, label=label, color=color)
    min_y = min([min(ls.objective_values) for ls in line_search_results])
    max_y = max([max(ls.objective_values) for ls in line_search_results])
    if min_y < 0:
        ax.set_ylim(top=max_y + 0.2 * (max_y - min_y))
    ax.set_ylabel("Objective function value")
    ax.set_xlabel("Number of iterations")
    max_iter = max([len(ls) for ls in line_search_results])
    min_iter = min([len(ls) for ls in line_search_results])
    ax.set_xlim(left=0.9, right=max_iter + 0.1)
    if max_iter > 10 * min_iter:
        ax.set_xscale('log')
        formatter = matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y))
        ax.xaxis.set_major_formatter(formatter)
    plt.legend(loc="upper right", prop={'size': 6})
    plt.show()
    # 3D plot
    regions = np.array([ls.path_xy_min_xy_max() for ls in line_search_results])
    region_x_min = min(regions[:, 0].min(), 0)
    region_y_min = min(regions[:, 1].min(), 0)
    region_x_max = max(regions[:, 2].max(), 0)
    region_y_max = max(regions[:, 3].max(), 0)
    region_dx = 0.2 * (region_x_max - region_x_min)
    region_dy = 0.2 * (region_y_max - region_y_min)
    x_contour = np.linspace(region_x_min - region_dx, region_x_max + region_dx, 30)
    y_contour = np.linspace(region_y_min - region_dy, region_y_max + region_dy, 30)
    x_matrix, y_matrix = np.meshgrid(x_contour, y_contour)
    points = np.vstack([x_matrix.reshape(-1), y_matrix.reshape(-1)])
    z_matrix = func(x=points, calc_gradient=False).reshape(x_matrix.shape)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x_matrix, y_matrix, z_matrix, cmap=matplotlib.cm.gist_heat_r, linewidth=0,
                           antialiased=False, alpha=0.5)
    min_z, max_z = min(z_matrix.min(), 0), max(z_matrix.max(), 0)
    eps_z = 0.05 * (max_z - min_z)
    ax.set_zlim(min_z, max_z)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.title("3D plot of " + func_name)
    for ind, ls in enumerate(line_search_results):
        objective_values = ls.objective_values + eps_z
        color = colors[ind]
        xs, ys = ls.location_values[0], ls.location_values[1]
        label = get_plot_label(ls)
        if len(objective_values) <= 2:
            ax.scatter(xs, ys, objective_values, label=label, linewidth=3, color=color)
        else:
            ax.plot(xs, ys, objective_values, label=label, linewidth=2, color=color)
    ax.set_xlabel("X values")
    ax.set_ylabel("Y values")
    ax.set_zlabel("Objective function value")
    plt.legend()
    plt.show()
    # plot contours
    plt.figure(figsize=(12, 7))
    plt.contour(x_matrix, y_matrix, z_matrix, 200)
    plt.title("2D contours plot of " + func_name)
    for ind, ls in enumerate(line_search_results):
        objective_values = ls.objective_values
        color = colors[ind]
        xs, ys = ls.location_values[0], ls.location_values[1]
        label = get_plot_label(ls)
        if len(objective_values) <= 2:
            plt.scatter(xs, ys, label=label, linewidth=3, color=color)
        else:
            plt.plot(xs, ys, label=label, linewidth=2, color=color)
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()
    plt.show()

def get_plot_label(ls):
    run_param = ls.run_param
    ret = "Hessian" if run_param['hessian'] else "Gradient Descent (step_len={})".format(run_param['step_len'])
    if run_param['wolfe_c1'] > 0:
        ret += " Wolfe:[{},{}]".format(run_param['wolfe_c1'], run_param['wolfe_backtracking'])
    return ret

import numpy as np


class LineSearch:
    eps = np.finfo(float).eps
    max_cond = 1 / np.finfo(float).eps

    def __init__(self, step_len=0.1, obj_tol=1e-12, param_tol=1e-8, max_iter=100, hessian=False, wolfe_c1=0,
                 wolfe_backtracking=0.5, verbose=True, print_every=5):
        self.step_len = step_len
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        self.hessian = hessian
        self.verbose = verbose
        self.print_every = print_every
        self.wolfe_c1 = wolfe_c1
        self.wolfe_backtracking = wolfe_backtracking
        self.run_mode_label = ""
        self.location_values = None
        self.objective_values = []
        self.run_param = {}

    def __len__(self):
        return len(self.objective_values)

    def path_xy_min_xy_max(self):
        return self.location_values[0].min(), self.location_values[1].min(), \
               self.location_values[0].max(), self.location_values[1].max()

    @staticmethod
    def apply_func_with_wolfe(f, x, hessian, step_len, wolfe_c1, wolfe_backtracking):
        f_res = f(x, hessian)
        g_val = f_res[1]
        if hessian:
            try:
                h_val = f_res[2]
                cond = np.linalg.cond(h_val)
                if not np.isfinite(cond) or cond > LineSearch.max_cond:
                    raise ValueError('Invalid condition number for Hessian')
                p_step = np.linalg.solve(h_val, -g_val)  # might throw exceptions
            except (ValueError, np.linalg.LinAlgError):
                # Hessian is not invertible --> could not converge
                p_step = None
        else:
            p_step = -step_len * g_val

        if (p_step is not None) and (wolfe_c1 > 0):
            a = 1
            wolfe_number = wolfe_c1 * g_val.T.dot(p_step)
            if wolfe_number <= -LineSearch.eps:
                while a >= LineSearch.eps:
                    next_f = f(x=x + a * p_step, calc_gradient=False)
                    if next_f > f_res[0] + a * wolfe_number:
                        # still not satisfying Wolfe condition...
                        a *= wolfe_backtracking
                        continue
                    # found step size that satisfied Wolfe condition
                    p_step *= a
                    break
        return f_res + (p_step,)

    def minimize(self, f, x0, step_len=None, obj_tol=None, param_tol=None, max_iter=None, hessian=None,
                 wolfe_c1=None, wolfe_backtracking=None, verbose=None, print_every=None):
        if step_len is None:
            step_len = self.step_len
        if obj_tol is None:
            obj_tol = self.obj_tol
        if param_tol is None:
            param_tol = self.param_tol
        if max_iter is None:
            max_iter = self.max_iter
        if hessian is None:
            hessian = self.hessian
        if wolfe_c1 is None:
            wolfe_c1 = self.wolfe_c1
        if wolfe_backtracking is None:
            wolfe_backtracking = self.wolfe_backtracking
        if verbose is None:
            verbose = self.verbose
        if print_every is None:
            print_every = self.print_every
        self.run_param = {'hessian': hessian, 'step_len': step_len, 'wolfe_c1': wolfe_c1,
                          'wolfe_backtracking': wolfe_backtracking}
        param_tol2 = param_tol * param_tol
        objective_prev = np.inf
        x_curr = np.array(x0, dtype=float).reshape(-1, 1)
        objective_values = []
        location_values = []
        success = False
        for itr in range(max_iter):
            f_res = LineSearch.apply_func_with_wolfe(f, x_curr, hessian, step_len, wolfe_c1, wolfe_backtracking)
            if f_res is None:
                break
            f_val = f_res[0]
            if f_val < objective_prev:
                objective_values.append(f_val)
                location_values.append(x_curr)
                if verbose and (((len(objective_values) - 1) % print_every == 0) or (itr == max_iter - 1)):
                    print("Iter[{}]: objective={} location: {}".format(itr + 1, objective_values[-1],
                                                                       location_values[-1].reshape(-1)))
            if objective_prev - f_val < obj_tol:
                success = True
                if verbose:
                    print("Optimization reached target of objective function.")
                break
            p_step = f_res[3]
            if p_step is None:
                # failure
                if verbose:
                    print("Optimization failed due to invalid p_step.")
                break
            if p_step.T.dot(p_step).item() < param_tol2:
                success = True
                if verbose:
                    print("Optimization reached target of location coordinates.")
                break
            x_curr = x_curr + p_step
            objective_prev = f_val
        self.objective_values = np.array(objective_values)
        self.location_values = np.hstack(location_values)
        return {'location': location_values[-1].reshape(-1), 'objective': objective_values[-1], 'success': success,
                'num_iter': len(objective_values)}

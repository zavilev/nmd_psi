import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from scipy.optimize import curve_fit, minimize
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.special import logsumexp

import os
import sys
from contextlib import contextmanager

def psi2r(psi,gamma):
    return gamma*psi/(1-psi+gamma*psi)
def r2psi(r,gamma):
    return r/(r+gamma-r*gamma)

# Define the hyperbolic curve
def hyperbolic_curve(x, a):
    return x / (x + a * (1 - x))

# Heteroskedastic noise model: parabola

def heteroskedastic_sigma(x,a,b,c):
    return a*x**2+b*x+c


def residuals_control(a,x_control,y_control):
    y_pred = hyperbolic_curve(x_control, a)
    return np.sum((y_control - y_pred) ** 2)


def gaussian_logpdf(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-3) #1e-10 before

    z = (y - mu) / sigma

    return (
        -0.5 * z**2
        - np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
    )


def negative_log_likelihood(params,a0_hat,
                            x_exp,y_exp,x_control,y_control,
                            tau,expected_pi,sigma_penalty,
                            lambda_pi,lambda_sigma):

    a1, pi, a_r, b_r, c_r, a_nr, b_nr, c_nr = params

    pi = np.clip(pi, 1e-6, 1 - 1e-6)

    # Predictions
    y_r = hyperbolic_curve(x_exp, a1)
    y_nr = hyperbolic_curve(x_exp, a0_hat)

    # Heteroskedastic sigma
    sigma_x_r = heteroskedastic_sigma(x_exp, a_r, b_r, c_r)
    sigma_x_nr = heteroskedastic_sigma(x_exp, a_nr, b_nr, c_nr)
    sigma_x_r = np.maximum(sigma_x_r, 1e-3)
    sigma_x_nr = np.maximum(sigma_x_nr, 1e-3)

    # Component log-likelihoods
    log_p_r = gaussian_logpdf(y_exp, y_r, sigma_x_r)
    log_p_nr = gaussian_logpdf(y_exp, y_nr, sigma_x_nr*tau)

    # Mixture log-likelihood:
    #
    # log( pi*p_r + (1-pi)*p_nr )
    #
    log_mix = logsumexp(
        np.vstack([
            np.log(pi) + log_p_r,
            np.log(1 - pi) + log_p_nr
        ]),
        axis=0
    )

    total_loglik = np.sum(log_mix)

    N = log_mix.shape[0]
    pi_penalty = lambda_pi*expected_pi*(pi-expected_pi)**2*N

    #sigma penalty
    sigma_curve = heteroskedastic_sigma(np.linspace(0,1,10), a_nr, b_nr, c_nr)

    sigma_penalty_val = np.mean(sigma_penalty-sigma_curve)/np.std(sigma_penalty)*N
    #penalty is higher for small variance
    multiplier = 1 if sigma_penalty_val>0 else -0.1
    sigma_penalty_val = lambda_sigma*sigma_penalty_val*multiplier
    
    #sigma_penalty_val = lambda_sigma*max(np.mean(sigma_penalty-sigma_curve),0)/np.std(sigma_penalty)*N

    obj_fun = -total_loglik + pi_penalty + sigma_penalty_val

    if not np.isfinite(obj_fun):
        return 1e100

    return obj_fun


def posterior_probabilities(
    x, y,
    a0, a1, pi,
    a_r, b_r, c_r,
    a_nr, b_nr, c_nr,
    tau, eps=1e-14
):

    y_r = hyperbolic_curve(x, a1)
    y_nr = hyperbolic_curve(x, a0)

    sigma_x_r = heteroskedastic_sigma(x, a_r, b_r, c_r)
    sigma_x_nr = heteroskedastic_sigma(x, a_nr, b_nr, c_nr)
    sigma_x_r = np.maximum(sigma_x_r, 1e-3)
    sigma_x_nr = np.maximum(sigma_x_nr, 1e-3)

    # ordinary densities (NOT logs)
    p_r = np.exp(gaussian_logpdf(y, y_r, sigma_x_r))
    p_nr = np.exp(gaussian_logpdf(y, y_nr, sigma_x_nr * tau))

    num = pi * p_r + eps
    denom = pi * p_r + (1 - pi) * p_nr + 2 * eps

    return num / denom


def get_params_mixture(exper, ctrl=None, tau=1, lambda_pi=0, lambda_sigma=0, n_init=10):

    if ctrl is None:  # take mirrored negative dpsi as ctrl
        ctrl = exper[exper.exp_psi < exper.ctrl_psi].rename(
            columns={'exp_psi': 'ctrl_psi', 'ctrl_psi': 'exp_psi'}
        )
        exper = exper[exper.exp_psi > exper.ctrl_psi]

    x_control = ctrl.exp_psi.to_numpy()
    y_control = ctrl.ctrl_psi.to_numpy()

    x_exp = exper.exp_psi.to_numpy()
    y_exp = exper.ctrl_psi.to_numpy()

    
    #expected pi
    cond1 = y_exp!=x_exp
    cond2 = y_control!=x_control
    pmix = np.mean(y_exp[cond1] < x_exp[cond1])
    p0 = np.mean(y_control[cond2] < x_control[cond2])
    expected_pi = max(0,(pmix-p0)/(1-p0))
    
    #print(expected_pi)

    # ---------------------------------------------------------
    # Fit null hyperbolic curve on control data
    # ---------------------------------------------------------

    a0_hat = minimize(
        residuals_control,
        x0=[1],
        args = (x_control,y_control),
        bounds=[(0.01, 500)]
    ).x[0]

    # ---------------------------------------------------------
    # Obtain empirical sigma
    # ---------------------------------------------------------


    y_pred = hyperbolic_curve(x_control,a0_hat)
    res2 = (y_control-y_pred)**2
    
    xy = lowess(
        res2,
        x_control,
        frac=0.4,   # smoothing strength
        return_sorted=True
    )
    x_sorted,sigma_empirical = xy[:,0],xy[:,1]
    sigma_empirical = np.sqrt(np.maximum(sigma_empirical, 1e-10))
    sigma_penalty = np.interp(np.linspace(0,1,10),x_sorted,sigma_empirical)
    #print(sigma_penalty)

    # ---------------------------------------------------------
    # Mixture model
    # ---------------------------------------------------------

    #init_params = [[5, 0.5, 1, 0.1],[20, 0.3, 1, 0.1],[2, 0.8, 0.5, 0.5]]

    bounds = [
        (a0_hat, 80),   # a1
        (0.01, 0.99),  # pi
        (-2,2),       # a_r
        (-2,2),       # b_r
        (-1,1),       # c_r
        (-2,2),       # a_nr
        (-2,2),       # b_nr
        (-1,1),       # c_nr
    ]

    #code to start optimizer from init_params
    def get_res(init_param):
        result = minimize(
            negative_log_likelihood,
            init_param,
            bounds=bounds,
            args = (a0_hat,
                    x_exp,y_exp,x_control,y_control,
                    tau,expected_pi,sigma_penalty,
                    lambda_pi,lambda_sigma),
            options={'ftol': 1e-12, 'maxiter': 5000}
        )
        if result.success:
            results.append([result,init_param])

    results = []
    if n_init == 1:
        init_param = [20, 0.5, -1, 1, 0, -1, 1, 0]
        get_res(init_param)
    else:
        for _ in range(n_init):
            init_param = [np.random.uniform(b[0],b[1]) for b in bounds]
            get_res(init_param)
    
    results.sort(key=lambda r: r[0].fun)
    #print([r[0].x[0] for r in results])
    failed = n_init-len(results)
    if failed==n_init:
        print('Unsuccessful optimization')
    elif failed>0:
        print(f'Warning: optimizer failed with {failed} out of {n_init} init params')
    result = results[0][0]

    a1_hat, pi_hat, a_r_hat, b_r_hat, c_r_hat, a_nr_hat, b_nr_hat, c_nr_hat = result.x

    prob = posterior_probabilities(x_exp, y_exp, a0_hat, a1_hat, pi_hat,
                                   a_r_hat, b_r_hat, c_r_hat, a_nr_hat, b_nr_hat, c_nr_hat,
                                   tau)
    res = (a0_hat, a1_hat, pi_hat, a_r_hat, b_r_hat, c_r_hat, a_nr_hat, b_nr_hat, c_nr_hat)

    other = (expected_pi,sigma_penalty)

    return result.success, res, prob, other, results


def plot_model(x,params,prob,ax=None):
    a0_hat, a1_hat, pi_hat, a_r_hat, b_r_hat, c_r_hat, a_nr_hat, b_nr_hat, c_nr_ha = params
    df = x.copy()
    df['prob'] = prob
    if ax is None:
        ax=plt.subplot()
    
    im = ax.scatter(df.ctrl_psi,df.exp_psi,c=df.prob,s=5)
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    cbar = plt.colorbar(im,ax=ax)
    cbar.ax.set_title("$P_R$")
    x = np.linspace(0,1,200)
    y0 =  psi2r(x,a0_hat)
    y1 =  psi2r(x,a1_hat)
    gamma_r = r'$\gamma_{R}$'
    gamma_nr = r'$\gamma_{NR}$'
    ax.plot(x,y1,label=f'{gamma_r}: {a1_hat:.2f}',ls='-',
            path_effects=[pe.Stroke(linewidth=3, foreground='white',alpha=0.8), pe.Normal()])
    ax.plot(x,y0,label=f'{gamma_nr}: {a0_hat:.2f}',ls='--',
            path_effects=[pe.Stroke(linewidth=3, foreground='white',alpha=0.8), pe.Normal()])
#    ysigma = df.exp_psi.sort_values()
#    sigma = heteroskedastic_sigma(ysigma,a_hat,b_hat,s_hat)
#    ax.plot(sigma,ysigma,label=r'$\sigma$',c='black',alpha=0.5)
    
    pinmd = "$\pi = $"
    #sigma = fr"$\sigma$ = -{a_hat:.2f}x^2 + {a_hat:.2f}x +{c_hat:.3f}"
    #ax.set_title(f'{pinmd} {pi_hat:.2f}\n{sigma}')
    ax.set_title(f'{pinmd} {pi_hat:.2f}')
    ax.legend()
    ax.set_xlabel(r'$\Psi$')
    ax.set_ylabel(r'$r$')
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    return ax



import os
from contextlib import redirect_stdout


def bootstrap(exper,ctrl=None,data_frac=0.5,n_samples=50,replace=True,**kwargs):
    res = []
    with open(os.devnull, 'w') as devnull:
        for i in range(n_samples):
            df = exper.sample(frac=data_frac, replace=replace)
            with redirect_stdout(devnull):
                success, params, *_ = get_params_mixture(df, ctrl=ctrl, **kwargs)
            if success:
                res.append(params)
    cols = 'a0_hat, a1_hat, pi_hat, a_r_hat, b_r_hat, c_r_hat, a_nr_hat, b_nr_hat, c_nr_hat'.split(', ')
    failures = n_samples-len(res)
    if failures>0:
        print(f'Warning: {failures} out of {n_samples} bootstrap samples failed')
    return pd.DataFrame(res,columns=cols)



from concurrent.futures import ProcessPoolExecutor, as_completed


# Worker function MUST be top-level for Windows multiprocessing
def run_bootstrap(args):
    name, exper, kwargs = args

    result = bootstrap(
        exper,
        **kwargs
    )

    return name, result


def parallel_bootstrap(
    datasets,
    n_workers=10,
    bootstrap_kwargs=None
):
    """
    Parameters
    ----------
    datasets : dict
        Example:
        {
            'ds1': df1,
            'ds2': df2,
            ...
        }

    bootstrap_kwargs : dict
        kwargs passed to bootstrap()

    Returns
    -------
    dict of bootstrap results
    """

    if bootstrap_kwargs is None:
        bootstrap_kwargs = dict()
    elif isinstance(bootstrap_kwargs,dict):
        bootstrap_kwargs = [bootstrap_kwargs]*len(datasets)

    jobs = [
        (name, df, kwargs)
        for (name, df),kwargs in zip(datasets.items(),bootstrap_kwargs)
    ]

    results = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        futures = [
            executor.submit(run_bootstrap, job)
            for job in jobs
        ]

        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
            print(f"Finished: {name}")

    return results




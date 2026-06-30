import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from itertools import combinations
import sys
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
import os
from statsmodels.formula.api import logit,ols
from scipy.stats import gaussian_kde

import scipy
from scipy.odr import Model, Data, ODR
import matplotlib.patheffects as pe
from scipy.optimize import curve_fit, minimize
from scipy.stats import gaussian_kde

def tls(y,x,intercept=None,slope=None):
    def slope_inter(B, x):
        return B[0]*x + B[1] 
    if (intercept is None) and (slope is None):
        f = slope_inter
        beta0 = [1,1]
    elif slope is None:
        f = lambda x,B: x*B + intercept
        beta0 = [1]
    elif intercept is None:
        f = lambda x,B: x*slope + B
        beta0 = [1]
    else:
        raise Exception('no parameters to estimate')
    linear = Model(f)
    mydata = Data(x, y)
    myodr = ODR(mydata, linear, beta0=beta0)
    res = myodr.run()
    return res

def get_ols_res(formula,df,log=False):
    if log:
        results = logit(formula,df).fit()
    else:
        results = ols(formula,df).fit()
    df = pd.DataFrame({
        "coef": results.params,
        "std_err": results.bse,
        "t": results.tvalues,
        "p_value": results.pvalues,
        "ci_lower": results.conf_int()[0],
        "ci_upper": results.conf_int()[1],
    })
    if not log:
        df['adj_R'] = results.rsquared_adj
        df['F_pval'] = results.f_pvalue
    return df

def p2star(p):
    thrs = [0.0001,0.001,0.01,0.05,0.1]
    star = ['****','***','**','*','·']
    for t,s in zip(thrs,star):
        if p<t:
            return s
    return ""

def add_N(col,sep='\n',brac=True):
    d = col.value_counts()
    if brac:
        d = {k:f'{k}{sep}$(n={v})$'  for k,v in d.items()}
    else:
        d = {k:f'{k}{sep}$n={v}$'  for k,v in d.items()}
    return col.map(d)


def plot2d(x,y):
    x = np.array(x)
    y = np.array(y)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    return z,idx



def psi2r(psi,gamma):
    return gamma*psi/(1-psi+gamma*psi)
def r2psi(r,gamma):
    return r/(r+gamma-r*gamma)
def psir2gamma(psi,r):
    return r*(1-psi)/(psi*(1-r))
    
def ntls(gamma,psi,r,relative=False):
    epsi = r2psi(r,gamma)
    er = psi2r(psi,gamma)
    if relative:
        #fun = sum(abs(psi-epsi)/psi)+sum(abs(r-er)/r)
        fun = sum(((psi-epsi)/psi)**2)+sum(((r-er)/r)**2)
    else:
        #fun = sum(abs(psi-epsi))+sum(abs(r-er))
        fun = sum((psi-epsi)**2)+sum((r-er)**2)
    return fun


def linear(df,thr=0.01):
    y = 1/(df[(df.ctrl_psi>thr)&(df.exp_psi>thr)].ctrl_psi) - 1
    x = 1/(df[(df.ctrl_psi>thr)&(df.exp_psi>thr)].exp_psi) - 1
    slope = tls(y,x,intercept=0).beta[0]
    return slope

def loglinear(df):
    y = np.log10(1/df.ctrl_psi - 1)
    x = np.log10(1/df.exp_psi - 1)
    inter = 10**tls(y,x,slope=1).beta[0]
    return inter

def nonlinear(df):
    relative = False
    slope = minimize(ntls,1,args=(df.ctrl_psi,df.exp_psi,relative)).x[0]
    return slope

def nonlinear_rel(df):
    relative = True
    slope = minimize(ntls,1,args=(df.ctrl_psi,df.exp_psi,relative)).x[0]
    return slope

def deriv0(df,thr=0.05):
    y = df[df.ctrl_psi<thr].ctrl_psi
    x = df[df.ctrl_psi<thr].exp_psi
    slope = tls(x,y,intercept=0).beta[0]
    return slope
def get_gamma(df,ax=None,plot=True,
              funcs=['linear','loglinear','nonlinear','nonlinear_rel','deriv0'],
              deriv0_thr = 0.05, linear_thr=0.01):
    d = {'linear':linear,'loglinear':loglinear,
         'nonlinear':nonlinear,'nonlinear_rel':nonlinear_rel,'deriv0':deriv0}
    ls = ['-','--',':','-.']
    resd = dict()
    if (ax is None) and plot:
        ax = plt.subplot()
    xmin,xmax = min(df.ctrl_psi),max(df.ctrl_psi)
    x = np.linspace(xmin,xmax,200)
    for i,funcname in enumerate(funcs):
        if funcname=='linear':
            resd[funcname] = d[funcname](df,thr=linear_thr)
        if (funcname=='deriv0') or (funcname=="f'(0)"):
            resd[funcname] = d['deriv0'](df,thr=deriv0_thr)
        else:
            resd[funcname] = d[funcname](df)
    if plot:
        for i,(funcname,gamma) in enumerate(resd.items()):
            y =  psi2r(x,gamma)
            style = ls[i%len(ls)]
#            ax.plot(x,y,label=f'{funcname}: {gamma:.1f}',ls=style,
#                    path_effects=[pe.Stroke(linewidth=3, foreground='white',alpha=0.8), pe.Normal()])
            ax.plot(x,y,label=f'{funcname}: {gamma:.1f}', lw=2)
        ax.legend()
        return resd,ax
    return resd

def compute_weights(x_source, x_target):
    kde_source = gaussian_kde(x_source)
    kde_target = gaussian_kde(x_target)
    
    weights = kde_target(x_source) / kde_source(x_source)
    return weights
##################################
# Define command-line arguments
##################################

import argparse
parser = argparse.ArgumentParser(description='Given a catalog of AS events, split-read counts and experiment design metadata \
                                 this example script outputs PSI values for AS events in each sample, \
                                 then calculates individual gammas and mixture model-based classification \
                                 on NMD sensitive and evading AS events along with corresponding class-level gammas')
parser.add_argument('-N','--NMD_catalog', metavar='NMD_AS_catalog.tsv', type=str, required=True,
                    help='Catalog of AS events of interest')
parser.add_argument('-n','--nonNMD_catalog', metavar='nonNMD_AS_catalog.tsv', type=str, required=True,
                    help='Catalog of AS events, that do not trigger NMD')
parser.add_argument('-m','--meta', metavar='metadata.tsv', type=str, required=True,
                    help='Metadata of NMD inactivation experiment')
parser.add_argument('-j','--junctions', metavar='path/to/dir', type=str, required=True,
                    help='directory with junction split-reads')
parser.add_argument('-s','--sites', metavar='path/to/dir', type=str, required=False,
                    help='directory with reads covering splice sites (needed for intron retention)')
parser.add_argument('-o','--output', metavar='path/to/dir/', type=str, required=True,
                    help='output directory')

#cmd = '-N example/input_data/catalogs/NMD_AS_catalog.tsv -n example/input_data/catalogs/nonNMD_AS_catalog.tsv -m example/input_data/metadata/HeLa_SMG6_SMG7_KD.tsv -j example/input_data/read_counts/junctions/ -s example/input_data/read_counts/sites/ -o example/output/'.split()

args = parser.parse_args()

##################################
# import modules
##################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os, subprocess, glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import mm
import helper_functions as hf

########################################################
# Convert junction split-read counts into PSI values
########################################################

j_files = os.listdir(args.junctions)
psi_out1 = args.output+'NMD_psi.tsv'
psi_out2 = args.output+'nonNMD_psi.tsv'

getpsi = str(Path(__file__).resolve().parent / "get_psi.py")
print(getpsi)

cmd1 = f'python {getpsi} -c {args.NMD_catalog} -j {args.junctions}/* -o {psi_out1} -t 5 --event src event_id --groupcol transcript_biotype --numerator nonsense_mediated_decay'

cmd2 = f'python {getpsi} -c {args.nonNMD_catalog} -j {args.junctions}/* -o {psi_out2} -t 5 --event src event_id --groupcol transcript_type --numerator inclusion'

if args.sites is not None:
    s_files = os.listdir(args.sites)
    cmd1+=f' -s {args.sites}/*'
    cmd2+=f' -s {args.sites}/*'

outdir = args.output.split('/')
outdir = '/'.join(outdir[:-1])
if outdir!='':
    os.makedirs(outdir, exist_ok=True)

result1 = subprocess.run(
    cmd1,
    shell=True,
    check=True,
    capture_output=True,
    text=True
)

result2 = subprocess.run(
    cmd2,
    shell=True,
    check=True,
    capture_output=True,
    text=True
)

######################################################################################################
# Aggregate PSI values from multiple samples and filter to obtain a pair of (PSI, r) for each AS event
######################################################################################################

dtype = {'src':'category','event_id':"category", 'sample_id':"category", 'num':float, 'denom':float}
nmd = pd.read_csv(psi_out1,index_col=[0],comment='#',dtype=dtype)
nonnmd = pd.read_csv(psi_out2,index_col=[0],comment='#',dtype=dtype)

meta = pd.read_csv(args.meta,sep='\t')

def aggregate(nmd,meta):
    nmd = meta.merge(nmd)
    nmd = nmd.groupby(['group','experiment','src','event_id'],as_index=False,observed=True,sort=False)[['denom','num']].agg('mean')
    nmd['psi'] = nmd.num/nmd.denom
    nmd = nmd.pivot(index=['experiment','src','event_id'],columns = 'group', values=['psi','denom'])
    nmd = nmd[(nmd.denom>15).all(1)].copy()
    nmd['dpsi'] = nmd.psi.exp-nmd.psi.ctrl
    nmd = nmd[nmd.dpsi!=0].copy()

    d = {('psi', 'ctrl'):'ctrl_psi',
         ('psi', 'exp'):'exp_psi',
         ('denom', 'ctrl'):'ctrl_denom',
         ('denom', 'exp'):'exp_denom'}
    nmd.columns = nmd.columns.map(d)
    nmd = nmd[d.values()].reset_index()
    nmd = nmd[(nmd.ctrl_psi!=0)&(nmd.exp_psi!=0)]
    return nmd
    

nmd = aggregate(nmd,meta)
nonnmd = aggregate(nonnmd,meta)

##################################
# Run mixture model
##################################

res,params,prob,other,results = mm.get_params_mixture(nmd,nonnmd,n_init=1)
if not res:
    raise Exception('model failed to converge')

print('####\tMixture model parameters:')
print(f'#\tgamma sensitive = {params[1]:.2f}')
print(f'#\tgamma evading = {params[0]:.2f}')
print(f'#\tfrac sensitive = {params[2]:.2f}')

######################################################################################################
# Add info on mixture model classification and individual gammas to quantification table
######################################################################################################

nmd['p_sensitive'] = prob.round(2)

condition = (nmd.p_sensitive>0.5)&(nmd.exp_psi>0.05)
nmd['event_class'] = np.where(condition,'NMD-sensitive','NMD-evading')
nmd['gamma_mixture'] = np.where(condition,params[1],params[0]).round(2)
nmd['gamma_i'] = hf.psir2gamma(nmd.ctrl_psi,nmd.exp_psi).round(2)

nmd.to_csv(args.output+'results.tsv',sep='\t',index=False)

##################################
# Plot model
##################################

f,ax = plt.subplots(figsize=(4.5,4))
mm.plot_model(nmd,params,prob,ax=ax)

a0_hat, a1_hat, pi_hat, a_r_hat, b_r_hat, c_r_hat, a_nr_hat, b_nr_hat, c_nr_hat = params
y = np.linspace(0,1,100)

sigma_r = mm.heteroskedastic_sigma(y, a_r_hat, b_r_hat, c_r_hat)
sigma_nr = mm.heteroskedastic_sigma(y, a_nr_hat, b_nr_hat, c_nr_hat)
sigma_r = np.maximum(sigma_r, 1e-3)
sigma_nr = np.maximum(sigma_nr, 1e-3)
curve_nr = mm.r2psi(y,a0_hat)
curve_r = mm.r2psi(y,a1_hat)

upper,lower = curve_r+sigma_r,curve_r-sigma_r
upper,lower = np.clip(upper,0,1),np.clip(lower,0,1)
ax.fill_betweenx(y,lower,upper,color='lightgrey',alpha=0.5)
ax.plot(upper,y,c='grey',ls=':')
ax.plot(lower,y,c='grey',ls=':')

upper,lower = curve_nr+sigma_nr,curve_nr-sigma_nr
upper,lower = np.clip(upper,0,1),np.clip(lower,0,1)
ax.fill_betweenx(y,lower,upper,color='lightgrey',alpha=0.5)
ax.plot(upper,y,c='grey',ls=':')
ax.plot(lower,y,c='grey',ls=':')

f.savefig(args.output+'mixture_model.pdf')
#!/opt/python/bin/python3.8
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, required=True, help='path to catalog of junctions')
parser.add_argument('-j', type=str, required=True, nargs='+', help='pyIPSA J6 files')
parser.add_argument('-s', type=str, required=False, nargs='+', help='pyIPSA S6 files',default=[])
parser.add_argument('-o', type=str, required=True, help='output file')
parser.add_argument('-t', type=int, required=False, default=1, help='number of threads')
parser.add_argument('--step', type=int, required=False, default=1000, help='step to print info')
parser.add_argument('--event', type=str, required=False, nargs='+', default=['event_id'], help='colname(s) of event id')
parser.add_argument('--groupcol', type=str, required=False, default='transcript_biotype', help='colname defining  2 groups for PSI calculation')
parser.add_argument('--numerator', type=str, required=False, default='nonsense_mediated_decay', help='a group name in column "--groupcol" which will be presented in numerator of PSI')

args = parser.parse_args()

import pandas as pd
import sys,os
import warnings
warnings.filterwarnings("ignore")

sys.path.append("C:/Users/zavil/arkuda_backup/unproductive_splicing/ipsa_related/")

import retrieve_psi_from_db as ipr

if __name__=='__main__':
    catalog = pd.read_csv(args.c,sep='\t')

    junctions = catalog[catalog.jtype.str.startswith('j')][['long_id']].drop_duplicates()
    junctions.columns = ['SE']

    sites = catalog[~catalog.jtype.str.startswith('j')][['long_id']].drop_duplicates()
    sites.columns = ['SE']

    jfiles = args.j
    sfiles = args.s
    if len(jfiles)==1:
        jfiles=jfiles[0]
    if len(sfiles)==1:
        sfiles=sfiles[0]

    print('Collecting IPSA J6 counts...')
    jnc = ipr.collect_counts(jfiles,ncores=args.t,se=junctions,step=args.step)
    print('Collecting IPSA S6 counts...')
    if sfiles:
        sts = ipr.collect_counts(sfiles,ncores=args.t,se=sites,step=args.step)
    else:
        print('S6 files not specified, skipping...')


    print('Joining J6 and S6 results...')
    #jcol = [i.rpartition('/')[-1].partition('.J6')[0] for i in jnc.columns]
    jcol = [os.path.basename(i).partition('.J6')[0] for i in jnc.columns]
    jnc.columns=jcol
    if sfiles:
        #scol = [i.rpartition('/')[-1].partition('.S6')[0] for i in sts.columns]
        scol = [os.path.basename(i).partition('.S6')[0] for i in sts.columns]
        sts.columns=scol
        cols = list(set(jcol)&set(scol))
        res = pd.concat([jnc[cols],sts[cols]])
    else:
        res = jnc

    del jnc
    del sts

    print('Calculating PSI...')
    psi = ipr.get_psi(catalog,res,event_col=args.event,
                      nmd_col=args.groupcol,nmd_tag=args.numerator)

    print('Writing to file...')
    with open(args.o,'w') as file:
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.abspath(__file__)
        file.write(f'#{path}\n')
        for key,val in args.__dict__.items():
            if key not in ['j','s']:
                file.write(f'#\t-{key} {val}\n')
            elif len(val)<10:
                file.write(f'#\t-{key} {" ".join(val)}\n')
            else:
                file.write(f'#\t-{key} {" ".join(val[:10])}... and {len(val)-10} more\n')
    psi.drop('psi',axis=1).to_csv(args.o,mode='a')
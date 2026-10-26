"""Verify custom NicheNet ATAC weights against independently prepared priors."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-root',type=Path,required=True)
    parser.add_argument('--inference-root',type=Path,required=True)
    parser.add_argument('--case-list',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=[]
    for key in args.case_list.read_text().splitlines():
        provided=pd.read_csv(args.sim_root/key/'pre_grn_weighted_matrix.txt',sep='\t',index_col=0)
        provided.index=provided.index.astype(str);provided.columns=provided.columns.astype(str)
        prior=provided.to_numpy(float).copy();np.fill_diagonal(prior,0)
        if prior.max()>0:prior/=prior.max()
        edges=pd.read_csv(args.inference_root/key/'atac_tf_target_prior.tsv',sep='\t',dtype={'from':str,'to':str})
        built=np.zeros_like(prior)
        for edge in edges.to_dict('records'):
            built[provided.index.get_loc(edge['from']),provided.columns.get_loc(edge['to'])]=edge['weight']
        rows.append({'case':key,'max_abs_weight_difference':float(np.max(np.abs(prior-built))),
                     'nonzero_support_equal':bool(np.array_equal(prior>0,built>0))})
    if not rows:raise ValueError('empty case list')
    out={'cases':rows,'all_supports_equal':all(r['nonzero_support_equal'] for r in rows),
         'max_abs_difference':max(r['max_abs_weight_difference'] for r in rows),'truth_read':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()

"""Portable six-p runner preserving legacy generalized-array/export orientation."""
import argparse
import json
from pathlib import Path
import pandas as pd
import legacy_joint as joint

P={'p0':0.,'p1_3':1/3,'p1_2':.5,'p2_3':2/3,'p1':1.,'p2':2.}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-root',type=Path,required=True)
    parser.add_argument('--truth-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/simulation_strata.json'))
    parser.add_argument('--p-label',choices=list(P),action='append')
    parser.add_argument('--case-key',action='append')
    parser.add_argument('--budget-ledger',type=Path,default=Path(__file__).with_name('iteration_budgets.csv'))
    parser.add_argument('--max-iter',type=int,help='explicit new-run override; otherwise use recorded case budgets')
    args=parser.parse_args()
    if args.max_iter is not None and args.max_iter<1:raise ValueError('max-iter must be positive')
    cfg=json.loads(args.config.read_text(encoding='utf-8'))['strata']
    ledger=pd.read_csv(args.budget_ledger).set_index(['p_label','case'])
    if not ledger.index.is_unique:raise ValueError('duplicate case/p ledger records')
    cases=joint.enumerate_cases(args.sim_root)
    if args.case_key:
        cases=[c for c in cases if c.key in set(args.case_key)]
        if {c.key for c in cases}!=set(args.case_key):raise ValueError('requested case missing')
    original=joint.make_options
    for label in args.p_label or list(P):
        for case in cases:
            budget=args.max_iter if args.max_iter else int(ledger.loc[(label,case.key),'iteration_budget'])
            def options(*positional,**kwargs):
                result=original(*positional,**kwargs)
                result.update(p=P[label],maxIter=budget)
                return result
            joint.make_options=options
            root=args.output/label
            target=root/case.key/'metrics.json'
            if target.exists():raise FileExistsError(f'refusing to overwrite {target}')
            params={**cfg[joint.stratum(case.gene_num)]['parameters'],'p':P[label]}
            print(label,case.key,'budget',budget,flush=True)
            result=joint.run_case(case,args.truth_root,root,params,'adaptive-gabriel',4,'binary')
            result.update(regularization_label=label,regularization_p=P[label],max_iter=budget,
                          matrix_orientation='legacy_regulator_by_target_no_implicit_transpose',
                          result_policy='all_finite_outputs_including_nonconverged')
            joint.write_json_atomic(result,target)


if __name__=='__main__':main()

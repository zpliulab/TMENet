"""Independent evaluation after custom-prior inference; existing outputs read-only."""
from __future__ import annotations
import concurrent.futures
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path('results/nichenet_custom_prior_20261015_v1')
CODE=Path('reproducibility/nichenet_custom')
CANONICAL=Path(__file__).resolve().parents[1]/'benchmarks/benchmark_lr_contribution_v2.py'
KEYS=['sender_type','receiver_type','ligand','receptor']
TRUTH=['effect','eligible_edge_count','truth_edge_count','truth_present','truth_density','truth_effect_density']
SCORES={'tmenet_symmetric_lr_conserved_attribution':'TMENet-LR','cellchat_probability':'CellChat','cellphonedb_mean':'CellPhoneDB','mdic3_lr_official_neglog10_bonferroni_p':'MDIC3-LR'}
METRICS=['auroc','average_precision','f1_max']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load_evaluator():
    sys.path.insert(0,str(CANONICAL.parent))
    spec=importlib.util.spec_from_file_location('frozen_ccc_evaluator',CANONICAL)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.evaluate_scores

def align_custom(truth, scores, manifest):
    if manifest.get('method')!='NicheNet_custom_prior' or manifest.get('status') not in ('complete','complete_no_prediction'):raise ValueError('Inference not valid for evaluation')
    if not manifest.get('no_truth_inputs'):raise ValueError('Inference input audit missing')
    if not set(KEYS+TRUTH).issubset(truth) or not set(KEYS+['score']).issubset(scores):raise ValueError('Schema mismatch')
    for f in (truth,scores):
        if f[KEYS].isna().any().any() or f.duplicated(KEYS).any():raise ValueError('Invalid LR keys')
    if not np.isfinite(pd.to_numeric(scores['score'],errors='raise')).all():raise ValueError('Non-finite scores')
    inside=scores.merge(truth[KEYS],on=KEYS,how='inner',validate='one_to_one')
    merged=truth[KEYS+TRUTH].merge(inside[KEYS+['score']],on=KEYS,how='left',validate='one_to_one')
    missing=int(merged['score'].isna().sum())
    merged['nichenet_custom_prior_lr_priority']=merged.pop('score').fillna(0.0)
    return merged,{'universe_rows':len(merged),'native_predictions_in_universe':len(inside),'native_predictions_outside_universe':len(scores)-len(inside),'unpredicted_rows_zero':missing}

def evaluate_case(key):
    cfg=json.loads((ROOT/'protocol.json').read_text())
    inference=ROOT/'formal'/key
    info=json.loads((inference/'manifest.json').read_text())
    frozen=Path(cfg['frozen_eval_root'])/key
    truth_path=frozen/'lr_scores_and_truth.tsv';baseline_path=frozen/'metrics.tsv'
    source_hash={'universe_sha256':sha(truth_path),'existing_metrics_sha256':sha(baseline_path),'inference_manifest_sha256':sha(inference/'manifest.json'),'prediction_sha256':sha(inference/'lr_score.tsv')}
    scores=pd.read_csv(inference/'lr_score.tsv',sep='\t',dtype={k:str for k in KEYS})
    truth=pd.read_csv(truth_path,sep='\t',dtype={k:str for k in KEYS})
    table,audit=align_custom(truth,scores,info)
    metrics=load_evaluator()(table)
    dest=ROOT/'evaluation'/key;dest.mkdir(parents=True,exist_ok=True)
    table.to_csv(dest/'lr_scores_and_truth.tsv',sep='\t',index=False)
    metrics.to_csv(dest/'metrics.tsv',sep='\t',index=False)
    match=re.fullmatch(r'geneNum_(\d+)_id_(\d+)/cellType(\d+)/seed(\d+)',key)
    n,network,ct,seed=map(int,match.groups())
    common={'case':key,'network':f'geneNum_{n}_id_{network}','gene_num':n,'network_id':network,'cell_type_configuration':ct,'seed':seed,'network_size':'Small' if n<=40 else 'Medium' if n<=70 else 'Large'}
    nn=metrics.iloc[0].to_dict();nn.update(common,method='NicheNet-custom-prior')
    baseline=pd.read_csv(baseline_path,sep='\t')
    baseline=baseline[baseline.method_score.isin(SCORES)].copy()
    if len(baseline)!=4:raise ValueError('Missing frozen baseline metrics')
    rows=[nn]
    for row in baseline.to_dict('records'):
        row.update(common,method=SCORES[row['method_score']]);rows.append(row)
    if source_hash['universe_sha256']!=sha(truth_path) or source_hash['existing_metrics_sha256']!=sha(baseline_path):raise ValueError('Frozen source changed during evaluation')
    audit.update(source_hash,inference_status=info['status'],scored_receivers=info.get('scored_receivers',0),zero_prediction_receivers=info.get('zero_prediction_receivers',0),constant_prior_activity_rows=info.get('constant_prior_activity_rows',0),case=key)
    (dest/'evaluation_manifest.json').write_text(json.dumps(audit,indent=2))
    return rows,audit

def main():
    import argparse
    global ROOT, CODE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    ROOT=args.root
    CODE=Path(__file__).resolve().parent
    if args.workers<1:raise ValueError('workers must be positive')
    cfg=json.loads((ROOT/'protocol.json').read_text());keys=(ROOT/'all_cases.txt').read_text().splitlines()
    status=json.loads((ROOT/'formal/batch_status.json').read_text())
    if status['cases']!=1500 or set(status['statuses'])-{'complete','complete_no_prediction'}:raise ValueError('Inference batch is incomplete or failed; no incomplete comparison')
    launch=json.loads((ROOT/'formal/launch.json').read_text())
    for name,expected in launch['source_sha256'].items():
        if sha(CODE/name)!=expected:raise ValueError('Inference source changed after launch: '+name)
    for key in keys:
        m=ROOT/'formal'/key/'manifest.json'
        if not m.exists():raise ValueError('Missing inference '+key)
    all_rows=[];audits=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,(rows,audit) in enumerate(pool.map(evaluate_case,keys),1):
            all_rows.extend(rows);audits.append(audit)
            if i%100==0:print('EVALUATED',i,flush=True)
    frame=pd.DataFrame(all_rows)
    if frame.duplicated(['case','method']).any() or not (frame.groupby('method').size()==1500).all():raise ValueError('Unpaired evaluation')
    if not np.isfinite(frame[METRICS].to_numpy()).all() or (frame[METRICS].to_numpy()<0).any() or (frame[METRICS].to_numpy()>1).any():raise ValueError('Invalid benchmark metrics')
    counts=frame.groupby(['method','network','cell_type_configuration']).size()
    if not (counts==5).all():raise ValueError('Each network/configuration requires five seeds')
    summary=ROOT/'summary';summary.mkdir(exist_ok=True)
    frame.to_csv(summary/'paired_case_metrics.tsv',sep='\t',index=False)
    overall=frame.groupby('method')[METRICS].agg(['mean','min','max']).reset_index()
    overall.columns=['_'.join(c).rstrip('_') for c in overall.columns]
    overall.to_csv(summary/'overall_case_summary.tsv',sep='\t',index=False)
    sized=frame.groupby(['network_size','method'])[METRICS].agg(['mean','min','max']).reset_index()
    sized.columns=['_'.join(c).rstrip('_') for c in sized.columns]
    sized.to_csv(summary/'case_summary_by_network_size.tsv',sep='\t',index=False)
    network=frame.groupby(['method','network_size','network','gene_num','cell_type_configuration'],as_index=False)[METRICS].mean()
    network.to_csv(summary/'five_seed_network_metrics.tsv',sep='\t',index=False)
    macro=network.groupby(['network_size','cell_type_configuration','method'])[METRICS].agg(['mean','min','max']).reset_index()
    macro.columns=['_'.join(c).rstrip('_') for c in macro.columns]
    macro.to_csv(summary/'network_macro_by_size_configuration.tsv',sep='\t',index=False)
    aud=pd.DataFrame(audits);aud.to_csv(summary/'case_evaluation_audit.tsv',sep='\t',index=False)
    release={'status':'complete','cases':1500,'networks':frame.network.nunique(),'methods':sorted(frame.method.unique()),'paired_rows':len(frame),'inference_status':status,'constant_prior_activity_rows':int(aud.constant_prior_activity_rows.sum()),'zero_prediction_receiver_groups':int(aud.zero_prediction_receivers.sum()),'no_prediction_cases':int((aud.inference_status=='complete_no_prediction').sum()),'evaluation_module_sha256':sha(CANONICAL),'protocol_sha256':sha(ROOT/'protocol.json'),'overall_case_summary':overall.to_dict('records'),'case_summary_by_size':sized.to_dict('records'),'all_existing_baseline_files_read_only':True,'manuscript_or_supplementary_tables_changed':False}
    (summary/'evaluation_summary.json').write_text(json.dumps(release,indent=2))
    delivery_columns=['case','network','gene_num','cell_type_configuration','seed','network_size','method',*METRICS]
    (summary/'delivery_payload.json').write_text(json.dumps({'evaluation':release,'cases':frame[delivery_columns].to_dict('records'),'network_macro':macro.to_dict('records'),'protocol':cfg},separators=(',',':')))
    print(overall.to_string(index=False),flush=True)
    print('EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()

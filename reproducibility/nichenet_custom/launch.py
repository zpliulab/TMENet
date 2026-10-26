"""Launch the recorded NicheNet custom-prior protocol without server paths."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-root',required=True,type=Path)
    parser.add_argument('--frozen-eval-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--rscript',default='Rscript')
    args=parser.parse_args()
    if args.workers<1:raise ValueError('workers must be positive')
    if args.output.exists():raise FileExistsError(args.output)
    code=Path(__file__).resolve().parent
    keys=sorted(p.parent.relative_to(args.sim_root).as_posix() for p in args.sim_root.glob('geneNum_*/cellType*/seed*/COMPLETE.json'))
    if len(keys)!=1500:raise ValueError(f'formal protocol requires 1500 completed cases, found {len(keys)}')
    cfg=json.loads((code/'protocol.json').read_text(encoding='utf-8'))
    cfg.update(sim_root=str(args.sim_root.resolve()),frozen_eval_root=str(args.frozen_eval_root.resolve()))
    args.output.mkdir(parents=True);formal=args.output/'formal';formal.mkdir()
    (args.output/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    (args.output/'all_cases.txt').write_text('\n'.join(keys)+'\n',encoding='utf-8')
    source_sha={name:hashlib.sha256((code/name).read_bytes()).hexdigest() for name in ['run_custom_batch.R','custom_core.R']}
    (formal/'launch.json').write_text(json.dumps({'source_sha256':source_sha,'case_count':len(keys)},indent=2)+'\n',encoding='utf-8')
    subprocess.run([args.rscript,str(code/'run_custom_batch.R'),str(args.output/'protocol.json'),
                    str(formal),str(args.output/'all_cases.txt'),str(args.workers)],check=True)


if __name__=='__main__':main()

from __future__ import annotations
import argparse
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .model import fit_generalized
from .preprocessing import prepare_inputs
from .lr import attribute_lr
from .pathways import prioritize_pathways


def read_frame(path):
    return pd.read_csv(path,sep="\t" if str(path).endswith((".tsv",".txt")) else ",",index_col=0)


def read_table(path):
    return pd.read_csv(path,sep="\t" if str(path).endswith((".tsv",".txt")) else ",")


def fit(args):
    output=Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite {output}")
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    options=config.get("options",config).copy()
    if args.p is not None:options["p"]=float(Fraction(args.p))
    if args.max_iter is not None:options["maxIter"]=args.max_iter
    with np.load(args.input,allow_pickle=False) as bundle:
        for key in ["A","B_prior","C_prior","B_mask"]:
            if key not in bundle:raise ValueError(f"input bundle missing {key}")
        result=fit_generalized(bundle["A"],bundle["B_prior"],bundle["C_prior"],options=options,
                              B_mask=bundle["B_mask"],M_knn=bundle["M_knn"] if "M_knn" in bundle else None,
                              coordinates=bundle["coordinates"] if "coordinates" in bundle else None,
                              backend=args.backend,variant=config.get("variant","simulation"),
                              normalize=config.get("normalize",True))
        result.save(args.output)
        labels={k:bundle[k].tolist() for k in ["genes","cells","cell_types","block_types"] if k in bundle}
        (Path(args.output)/"labels.json").write_text(json.dumps(labels,indent=2)+"\n",encoding="utf-8")
    (Path(args.output)/"run_config.json").write_text(json.dumps({"variant":config.get("variant","simulation"),"backend":args.backend,"options":options},indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result.diagnostics,indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(prog="tmenet",description="Joint GRN / spatial coupling inference and post-hoc LR attribution")
    parser.add_argument("--version",action="version",version="TMENet 0.1.0")
    sub=parser.add_subparsers(dest="command",required=True)
    f=sub.add_parser("fit");f.add_argument("--input",required=True);f.add_argument("--config",required=True)
    f.add_argument("--output",required=True);f.add_argument("--backend",choices=["cpu","gpu"],default="cpu")
    f.add_argument("--p");f.add_argument("--max-iter",type=int)
    p=sub.add_parser("prepare");p.add_argument("--expression",required=True);p.add_argument("--metadata",required=True)
    p.add_argument("--coordinates",required=True);p.add_argument("--prior-edges",required=True);p.add_argument("--lr-pairs",required=True)
    p.add_argument("--output",required=True);p.add_argument("--k",type=int)
    l=sub.add_parser("lr");l.add_argument("--expression",required=True);l.add_argument("--coupling",required=True)
    l.add_argument("--metadata",required=True);l.add_argument("--pairs",required=True);l.add_argument("--output",required=True)
    l.add_argument("--mask");l.add_argument("--symmetrize",action="store_true")
    r=sub.add_parser("pathways");r.add_argument("--lr",required=True);r.add_argument("--expression",required=True)
    r.add_argument("--metadata",required=True);r.add_argument("--routes",required=True);r.add_argument("--grn",required=True)
    r.add_argument("--output",required=True);r.add_argument("--min-codetected",type=int,default=20)
    args=parser.parse_args(argv)
    if args.command=="fit":fit(args);return
    destination=Path(args.output)
    if destination.exists():raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True,exist_ok=True)
    if args.command=="prepare":
        prepared=prepare_inputs(read_frame(args.expression),read_table(args.metadata),read_frame(args.coordinates),
                                read_table(args.prior_edges),read_table(args.lr_pairs),args.k)
        spatial=prepared.pop("spatial_info")
        with destination.open("wb") as handle:np.savez_compressed(handle,**prepared)
        destination.with_suffix(".spatial.json").write_text(json.dumps(spatial,indent=2)+"\n",encoding="utf-8")
    elif args.command=="lr":
        expression=read_frame(args.expression)
        cp=Path(args.coupling)
        if cp.suffix==".npy":coupling=np.load(cp,allow_pickle=False)
        else:
            frame=read_frame(cp)
            frame.index=frame.index.astype(str);frame.columns=frame.columns.astype(str)
            expression.columns=expression.columns.astype(str)
            coupling=frame.loc[expression.columns,expression.columns].to_numpy(float)
        if cp.suffix==".npy":
            label_path=cp.parent/"labels.json"
            if not label_path.exists():raise ValueError("unlabeled coupling.npy requires adjacent labels.json")
            labels=json.loads(label_path.read_text(encoding="utf-8")).get("cells",[])
            expression.columns=expression.columns.astype(str)
            if set(labels)!=set(expression.columns) or len(labels)!=len(expression.columns):raise ValueError("saved coupling/expression cell labels differ")
            expression=expression.loc[:,labels]
        mask=np.load(args.mask,allow_pickle=False) if args.mask else None
        scores,audit=attribute_lr(expression,coupling,read_table(args.metadata),read_table(args.pairs),
                                  candidate_mask=mask,symmetrize=args.symmetrize)
        scores.to_csv(destination,sep="\t",index=False)
        destination.with_suffix(".audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(audit,indent=2))
    else:
        result=prioritize_pathways(read_table(args.lr),read_frame(args.expression),read_table(args.metadata),
                                    read_table(args.routes),read_table(args.grn),min_codetected=args.min_codetected)
        result.to_csv(destination,sep="\t",index=False)


if __name__=="__main__":main()

"""Create a deterministic API smoke fixture, not a paper benchmark dataset."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tmenet.preprocessing import prepare_inputs


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="examples/toy")
    args=p.parse_args();out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42)
    genes=["TF1","TF2","L1","L2","R1","R2","G1","G2"]
    cells=[f"cell{i:02d}" for i in range(36)]
    expr=pd.DataFrame(rng.poisson(4,(8,36)),index=genes,columns=cells)
    expr.iloc[2,:18]+=4;expr.iloc[4,18:]+=4
    expr.iloc[3,18:]+=3;expr.iloc[5,:18]+=3
    metadata=pd.DataFrame({"cell":cells,"cell_type":["type_A"]*18+["type_B"]*18})
    coords=pd.DataFrame([(i%6+(i//6%2)*0.17,i//6) for i in range(36)],index=cells,columns=["x","y"])
    prior=pd.DataFrame({"regulator":["TF1","TF1","TF2","TF2"],"target":["G1","L1","G2","R1"]})
    pairs=pd.DataFrame({"ligand":["L1","L1","L2"],"receptor":["R1","R2","R2"]})
    data=prepare_inputs(expr,metadata,coords,prior,pairs)
    data.pop("spatial_info")
    np.savez_compressed(out/"inputs.npz",**data)
    np.save(out/"candidate_mask.npy",data["M_knn"],allow_pickle=False)
    expr.loc[:,data["cells"]].to_csv(out/"expression.tsv",sep="\t")
    metadata.to_csv(out/"metadata.tsv",sep="\t",index=False)
    coords.to_csv(out/"coordinates.tsv",sep="\t")
    prior.to_csv(out/"prior_edges.tsv",sep="\t",index=False)
    pairs.to_csv(out/"lr_pairs.tsv",sep="\t",index=False)
    print(f"Toy fixture written to {out}")


if __name__=="__main__":main()

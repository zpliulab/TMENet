"""Join LR attribution to curated receptor-TF routes and receiver GRN edges."""
import numpy as np
import pandas as pd


def prioritize_pathways(lr_attribution,expression,metadata,receptor_tf_routes,grn_edges,
                        min_codetected=20,min_fraction=0.01):
    """Evidence-constrained candidates, not causal effects or validated pathways.

    Routes: receptor,tf,route,source_database (each row must be curated support).
    GRN edges: cell_type,regulator,target,weight in explicit edge-list orientation.
    """
    required={"receptor","tf","route","source_database"}
    if not required.issubset(receptor_tf_routes):raise ValueError(f"route table requires {sorted(required)}")
    if receptor_tf_routes[list(required)].isna().any().any():raise ValueError("route evidence cannot be missing")
    if not {"cell_type","regulator","target","weight"}.issubset(grn_edges):raise ValueError("invalid GRN edge table")
    if not {"cell","cell_type"}.issubset(metadata):raise ValueError("invalid metadata")
    if min_codetected<1 or not 0<=min_fraction<=1:raise ValueError("invalid co-detection thresholds")
    if (expression.values<0).any() or not np.isfinite(expression.values).all():raise ValueError("invalid expression values")
    expr=expression.copy();expr.columns=expr.columns.astype(str);expr.index=expr.index.astype(str)
    rows=[]
    for receiver,meta in metadata.groupby("cell_type",sort=True):
        cells=meta.cell.astype(str).tolist()
        if not set(cells).issubset(expr.columns):raise ValueError("receiver cells absent from expression")
        threshold=max(int(min_codetected),int(np.ceil(min_fraction*len(cells))))
        accepted=[]
        for route in receptor_tf_routes.itertuples(index=False):
            if str(route.receptor) not in expr.index or str(route.tf) not in expr.index:continue
            co=int(((expr.loc[str(route.receptor),cells]>0)&(expr.loc[str(route.tf),cells]>0)).sum())
            if co>=threshold:
                accepted.append({"receptor":str(route.receptor),"regulator":str(route.tf),"route":route.route,
                                 "source_database":route.source_database,"codetected_spots":co,
                                 "receiver_spots":len(cells)})
        if not accepted:continue
        routes=pd.DataFrame(accepted).drop_duplicates()
        lr=lr_attribution[(lr_attribution.receiver_type==receiver)&(lr_attribution.attributed_sum>0)]
        grn=grn_edges[(grn_edges.cell_type==receiver)&(grn_edges.weight>0)].copy()
        if not np.isfinite(grn.weight).all():raise ValueError("non-finite GRN weights")
        if len(grn):grn["grn_rank"]=grn.weight.rank(method="average",pct=True)
        merged=lr.merge(routes,on="receptor").merge(grn,on="regulator")
        if merged.empty:continue
        merged["lr_rank"]=merged.groupby(["sender_type","receiver_type"]).attributed_sum.rank(pct=True)
        merged["codetection_rank"]=merged.codetected_spots.rank(pct=True)
        merged["priority_score"]=np.cbrt(merged.lr_rank*merged.codetection_rank*merged.grn_rank)
        rows.append(merged)
    return pd.concat(rows,ignore_index=True).sort_values("priority_score",ascending=False) if rows else pd.DataFrame()

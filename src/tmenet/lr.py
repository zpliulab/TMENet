"""Coupling-conserving, post-inference ligand-receptor attribution."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .preprocessing import nonnegative_matrix


def positive_percentiles(expression):
    x=nonnegative_matrix(expression,"expression")
    out=np.zeros_like(x)
    for g in range(len(x)):
        positive=x[g]>0
        if positive.any():out[g,positive]=rankdata(x[g,positive],method="average")/positive.sum()
    return out


def attribute_lr(expression,coupling,metadata,pairs,*,candidate_mask=None,
                 symmetrize=False,chunk_size=1024):
    """Return type-pair LR sums/means/shares and a per-edge conservation audit.

    C is sender-row/receiver-column. Molecular support is r_L(i)*r_R(j).
    All supplied LR pairs compete in the denominator. Unsupported coupling is
    reported as unallocated, not silently assigned. No truth labels are used.
    """
    expr=expression.copy();expr.index=expr.index.astype(str);expr.columns=expr.columns.astype(str)
    if expr.empty or not expr.index.is_unique or not expr.columns.is_unique:
        raise ValueError("expression must have unique, nonempty gene/cell labels")
    x=positive_percentiles(expr.to_numpy())
    cells=expr.columns.tolist();n=len(cells)
    c=nonnegative_matrix(coupling,"coupling").copy()
    if c.shape!=(n,n):raise ValueError("coupling is not aligned to expression")
    if symmetrize:c=(c+c.T)/2
    if not {"cell","cell_type"}.issubset(metadata.columns):raise ValueError("metadata needs cell/cell_type")
    meta=metadata.copy();meta["cell"]=meta.cell.astype(str)
    if meta.cell.duplicated().any() or set(meta.cell)!=set(cells):raise ValueError("metadata cell mismatch")
    types=meta.set_index("cell").loc[cells,"cell_type"].astype(str).to_numpy()
    if candidate_mask is None:mask=c>0
    else:
        mask=np.asarray(candidate_mask)
        if mask.shape!=c.shape or not np.isin(mask,[0,1]).all():raise ValueError("candidate_mask must be binary and aligned")
        mask=mask.astype(bool)
    if np.any(np.diag(mask)):raise ValueError("self-coupling is outside this LR candidate universe")
    if chunk_size<1:raise ValueError("chunk_size must be positive")
    lr=pairs.copy()
    if not {"ligand","receptor"}.issubset(lr.columns):raise ValueError("pairs needs ligand/receptor")
    lr=lr[["ligand","receptor"]].astype(str)
    for col in lr:lr[col]=lr[col].str.strip()
    supplied=len(lr)
    lr=lr[lr.ligand.isin(expr.index)&lr.receptor.isin(expr.index)&(lr.ligand!=lr.receptor)].drop_duplicates().sort_values(["ligand","receptor"]).reset_index(drop=True)
    if lr.empty:raise ValueError("no LR candidates match the expression feature space")
    gene_position={g:i for i,g in enumerate(expr.index)}
    li=np.array([gene_position[g] for g in lr.ligand]);ri=np.array([gene_position[g] for g in lr.receptor])
    send,recv=np.where(mask)
    if len(send)==0:raise ValueError("empty spatial candidate mask")
    groups=sorted(set(zip(types[send],types[recv])))
    group_idx={g:i for i,g in enumerate(groups)}
    sums=np.zeros((len(groups),len(lr)));counts=np.zeros_like(sums,dtype=int)
    candidate_counts=np.zeros(len(groups),dtype=int)
    total=np.zeros(len(groups));supported_total=np.zeros(len(groups))
    max_error=0.;unallocated=0.
    for first in range(0,len(send),chunk_size):
        si=send[first:first+chunk_size];rj=recv[first:first+chunk_size]
        q=x[li[:,None],si]*x[ri[:,None],rj]
        denom=q.sum(axis=0);supported=denom>0
        share=np.divide(q,denom[None,:],out=np.zeros_like(q),where=denom[None,:]>0)
        weights=c[si,rj];attribution=share*weights[None,:]
        max_error=max(max_error,float(np.max(np.abs(attribution.sum(axis=0)-weights*supported))))
        unallocated+=float(weights[~supported].sum())
        for g in set(zip(types[si],types[rj])):
            gi=group_idx[g];keep=(types[si]==g[0])&(types[rj]==g[1])
            sums[gi]+=attribution[:,keep].sum(axis=1)
            counts[gi]+=(q[:,keep]>0).sum(axis=1)
            candidate_counts[gi]+=int(keep.sum())
            total[gi]+=float(weights[keep].sum());supported_total[gi]+=float(weights[keep&supported].sum())
    rows=[]
    for gi,(sender,receiver) in enumerate(groups):
        for pi,pair in enumerate(lr.itertuples(index=False)):
            rows.append(dict(sender_type=sender,receiver_type=receiver,ligand=pair.ligand,receptor=pair.receptor,
                             attributed_sum=float(sums[gi,pi]),attributed_mean=float(sums[gi,pi]/candidate_counts[gi]),
                             attribution_share_percent=float(100*sums[gi,pi]/supported_total[gi]) if supported_total[gi]>0 else 0.,
                             support_edges=int(counts[gi,pi]),candidate_edges=int(candidate_counts[gi]),
                             total_coupling=float(total[gi]),supported_coupling=float(supported_total[gi])))
    audit=dict(candidate_pairs_supplied=supplied,candidate_pairs_matched=len(lr),candidate_edges=len(send),
               symmetrized=bool(symmetrize),max_conservation_error=max_error,unallocated_coupling=unallocated,
               score_definition="coupling-conserving expression-supported attribution; not an LR binding probability")
    if max_error>1e-8:raise FloatingPointError("LR attribution failed conservation")
    return pd.DataFrame(rows),audit

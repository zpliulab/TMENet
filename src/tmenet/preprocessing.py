"""Labeled inputs and spatial masks; no ground-truth inputs are used here."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.distance import cdist


def nonnegative_matrix(value, name):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError(f"{name} must be a finite nonnegative matrix")
    return a


def spatial_neighborhood(coordinates, k=None):
    """Gabriel-mean adaptive K and directed KNN retaining kth-distance ties.

    Mirrors the paper's closed-Gabriel-ball rule and np.rint rounding.
    Small/collinear inputs use an exact all-pairs Gabriel check, not a new K rule.
    """
    xy = np.asarray(coordinates, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError("coordinates must be a finite n-by-2 array with n >= 2")
    n = len(xy)
    if len(np.unique(xy, axis=0)) != n:
        raise ValueError("duplicate spatial coordinates are not supported")
    d = cdist(xy, xy)
    details = {"rule": "fixed", "requested_k": k}
    if k is None:
        try:
            triangles = Delaunay(xy).simplices
            candidates = {tuple(sorted((int(t[a]), int(t[b])))) for t in triangles
                          for a,b in [(0,1),(0,2),(1,2)]}
        except QhullError:
            if n > 256:
                raise ValueError("degenerate geometry: supply an explicit K or nondegenerate coordinates")
            candidates = {(i,j) for i in range(n) for j in range(i+1,n)}
        degrees = np.zeros(n, dtype=int)
        for i,j in candidates:
            midpoint = (xy[i] + xy[j])/2
            radius2 = float(np.sum((xy[i]-xy[j])**2)/4)
            squared = np.sum((xy-midpoint)**2, axis=1)
            others = np.ones(n, dtype=bool); others[[i,j]] = False
            if not np.any(squared[others] <= radius2 + 1e-10*max(1., radius2)):
                degrees[i] += 1; degrees[j] += 1
        if np.any(degrees == 0):
            raise ValueError("Gabriel graph contains isolated nodes; inspect coordinates")
        k = int(np.clip(np.rint(degrees.mean()),3,12))
        details = {"rule": "clip(round(mean_Gabriel_degree),3,12)",
                   "gabriel_mean_degree": float(degrees.mean()),
                   "gabriel_edges": int(degrees.sum()//2)}
    if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or k < 1:
        raise ValueError("K must be a positive integer")
    effective = min(int(k), n-1)
    safe = d.copy(); np.fill_diagonal(safe, np.inf)
    threshold = np.sort(safe, axis=1)[:,effective-1]
    mask = (safe <= threshold[:,None] + 1e-12).astype(float)
    np.fill_diagonal(mask,0)
    return d, mask, {**details,"selected_k":int(k),"effective_k":effective,
                     "min_neighbors":int(mask.sum(axis=1).min()),
                     "max_neighbors":int(mask.sum(axis=1).max())}


def prepare_inputs(expression, metadata, coordinates, prior_edges, lr_pairs, k=None):
    """Build A/B*/C* from labeled genes-by-spots expression.

    prior_edges: regulator,target plus optional cell_type; output B* uses
    target rows and regulator columns. Metadata has columns cell,cell_type.
    Coordinate index is cell ID and its first two columns are x,y.
    """
    if not isinstance(expression,pd.DataFrame) or expression.empty:
        raise ValueError("expression must be a nonempty pandas DataFrame")
    expression = expression.copy()
    expression.index = expression.index.astype(str)
    expression.columns = expression.columns.astype(str)
    if not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError("gene and cell labels must be unique")
    values = nonnegative_matrix(expression.values,"expression")
    meta = metadata.copy()
    if not {"cell","cell_type"}.issubset(meta.columns):
        raise ValueError("metadata needs cell and cell_type columns")
    meta["cell"] = meta["cell"].astype(str)
    meta["cell_type"] = meta["cell_type"].astype(str)
    if meta["cell"].duplicated().any() or set(meta.cell) != set(expression.columns):
        raise ValueError("metadata cells must match expression exactly")
    types = sorted(meta.cell_type.unique())
    mapping = meta.set_index("cell").cell_type
    order = [c for ct in types for c in expression.columns if mapping[c] == ct]
    expr = expression.loc[:,order]
    if not coordinates.index.is_unique or not set(order).issubset(set(coordinates.index.astype(str))):
        raise ValueError("coordinate labels must cover all expression cells uniquely")
    coord = coordinates.copy(); coord.index = coord.index.astype(str)
    d,mask,spatial = spatial_neighborhood(coord.loc[order].iloc[:,:2].to_numpy(),k)
    genes = expr.index.tolist(); m,n = expr.shape; t = len(types)
    a = np.zeros((t*m,n)); b = np.zeros((t*m,t*m)); bmask = np.zeros_like(b)
    gpos = {g:i for i,g in enumerate(genes)}
    if not {"regulator","target"}.issubset(prior_edges.columns):
        raise ValueError("prior_edges needs regulator and target columns")
    for qi,ct in enumerate(types):
        sl = slice(qi*m,(qi+1)*m)
        cols = [i for i,c in enumerate(order) if mapping[c] == ct]
        a[sl,cols] = expr.iloc[:,cols].values
        bmask[sl,sl] = 1
        edges = prior_edges if "cell_type" not in prior_edges else prior_edges[prior_edges.cell_type.astype(str)==ct]
        for edge in edges.itertuples(index=False):
            reg,target = str(edge.regulator),str(edge.target)
            if reg in gpos and target in gpos:
                b[qi*m+gpos[target],qi*m+gpos[reg]] = 1.
    if not {"ligand","receptor"}.issubset(lr_pairs.columns):
        raise ValueError("lr_pairs needs ligand and receptor columns")
    compatible = np.zeros((n,n),dtype=bool)
    for lr in lr_pairs.itertuples(index=False):
        l,r = str(lr.ligand),str(lr.receptor)
        if l in gpos and r in gpos:
            compatible |= np.outer(expr.loc[l].values>0,expr.loc[r].values>0)
    cstar = compatible.astype(float)*mask
    return {"A":a,"B_prior":b,"C_prior":cstar,"B_mask":bmask,"M_knn":mask,
            "D":d,"coordinates":coord.loc[order].iloc[:,:2].to_numpy(float),
            "genes":np.asarray(genes,dtype=str),"cells":np.asarray(order,dtype=str),
            "cell_types":np.asarray([mapping[c] for c in order],dtype=str),
            "block_types":np.asarray(types,dtype=str),"spatial_info":spatial}

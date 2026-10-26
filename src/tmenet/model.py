"""Portable validation/output adapter around the audited optimization cores."""
from __future__ import annotations
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import warnings
import numpy as np
from .preprocessing import nonnegative_matrix, spatial_neighborhood
from ._lp_prox import fractional_prox


@dataclass
class FitResult:
    B: np.ndarray
    X: np.ndarray
    C: np.ndarray
    objective: list
    augmented_objective: list
    diagnostics: dict

    def save(self,directory):
        path=Path(directory)
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite a nonempty output directory: {path}")
        path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/"factors.npz",B=self.B,X=self.X,C=self.C)
        np.save(path/"B.npy",self.B,allow_pickle=False)
        np.save(path/"C.npy",self.C,allow_pickle=False)
        np.save(path/"X.npy",self.X,allow_pickle=False)
        (path/"diagnostics.json").write_text(json.dumps(self.diagnostics,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        (path/"objective.json").write_text(json.dumps({"raw":self.objective,"augmented":self.augmented_objective},indent=2)+"\n",encoding="utf-8")


def _host(value,xp):
    if xp is not np and isinstance(value,xp.ndarray):
        value=xp.asnumpy(value)
    if isinstance(value,np.ndarray):return value.item() if value.ndim==0 else value
    if isinstance(value,np.generic):return value.item()
    if xp is not np and isinstance(value,xp.generic):return value.item()
    if isinstance(value,dict):return {k:_host(v,xp) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_host(v,xp) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def fit_generalized(A,B_prior,C_prior,*,options,coordinates=None,B_mask=None,M_knn=None,
                    backend="cpu",variant="simulation",normalize=True):
    """Jointly infer B, X and C; save/return every finite result, even max-iter stops.

    A: generalized genes-by-cells; B*: target-by-regulator; C*: sender-by-receiver.
    Already assembled legacy matrices are never silently transposed.
    variant is simulation (frozen six-p core) or real (AMI/HCC core).
    """
    if backend not in {"cpu","gpu"} or variant not in {"simulation","real"}:
        raise ValueError("invalid backend or core variant")
    a=nonnegative_matrix(A,"A").copy()
    b=nonnegative_matrix(B_prior,"B_prior").copy()
    c=nonnegative_matrix(C_prior,"C_prior").copy()
    m,n=a.shape
    if b.shape!=(m,m) or c.shape!=(n,n) or n<2:
        raise ValueError("incompatible A/B_prior/C_prior shapes")
    opts=dict(options)
    required={"rho_star","eta","p","lambda_B","lambda_C","beta_B","beta_C"}
    missing=required-set(opts)
    if missing:raise ValueError(f"missing optimizer options: {sorted(missing)}")
    p=float(opts["p"])
    if p not in [0.,1/3,0.5,2/3,1.,2.]:
        raise ValueError("supported p values are 0,1/3,1/2,2/3,1,2")
    if int(opts.get("maxIter",1000))<1:raise ValueError("maxIter must be positive")
    for key in ["rho_star","eta"]:
        if not np.isfinite(opts[key]) or opts[key]<=0:raise ValueError(f"{key} must be positive")
    for key in ["lambda_B","lambda_C","beta_B","beta_C","lambda_X"]:
        if key in opts and (not np.isfinite(opts[key]) or opts[key]<0):raise ValueError(f"{key} must be nonnegative")
    if B_mask is None:B_mask=opts.pop("B_mask",None)
    if B_mask is None:raise ValueError("explicit cell-type block B_mask is required")
    bm=nonnegative_matrix(B_mask,"B_mask")
    if bm.shape!=b.shape or not np.isin(bm,[0,1]).all() or np.any(b[bm==0]!=0):
        raise ValueError("invalid B block mask or off-block prior values")
    spatial={}
    if M_knn is None:M_knn=opts.pop("M_knn",None)
    if M_knn is None:
        if coordinates is None:raise ValueError("coordinates or an explicit spatial mask is required")
        d,M_knn,spatial=spatial_neighborhood(coordinates,opts.get("K"))
        opts.update(D=d,K=spatial["selected_k"])
    mask=nonnegative_matrix(M_knn,"M_knn")
    if mask.shape!=c.shape or not np.isin(mask,[0,1]).all() or np.any(np.diag(mask)):
        raise ValueError("spatial mask must be binary, square and zero-diagonal")
    if np.any(c[mask==0]!=0):
        raise ValueError("C_prior conflicts with spatial mask; project it explicitly before fitting")
    opts.update(B_mask=bm,M_knn=mask,return_info=True,use_gpu=backend=="gpu")
    opts.setdefault("maxIter",1000);opts.setdefault("scale_every",0)
    opts.setdefault("adaptive_step",True);opts.setdefault("stationarity_tol",0.02)
    opts.setdefault("display",False);opts.setdefault("lambda_X",0.)
    opts.setdefault("row_normalize_C",False);opts.setdefault("zero_diag_C",False)
    opts.setdefault("mask_conflict_policy","error");opts.setdefault("mask_conflict_tolerance",0.)
    if normalize:
        a=np.log1p(a);a=(a-a.min())/(a.max()-a.min()+1e-12)
    xp=np
    if backend=="gpu":
        try:
            import cupy as xp
            if xp.cuda.runtime.getDeviceCount()<1:raise RuntimeError("no CUDA device")
        except Exception as exc:
            raise RuntimeError("GPU requested but CuPy/CUDA is unavailable; install a compatible CuPy build or use CPU") from exc
    # Isolated module state prevents CPU/GPU calls changing each other's namespace.
    filename="_core_cpu.py" if variant=="simulation" else "_core_real.py"
    spec=importlib.util.spec_from_file_location("_tmenet_isolated_core",Path(__file__).with_name(filename))
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    if variant=="simulation":core.np=xp
    core.prox_p=lambda d,alpha,p,eps=1e-12:fractional_prox(d,alpha,p,xp)
    if opts.pop("hcc_corrected_projection",False):
        if variant!="real":raise ValueError("HCC projection requires the real-data core")
        from ._hcc_projection import weighted_ball_projection
        core.weighted_ball_projection=weighted_ball_projection
    if backend=="gpu":
        opts={k:xp.asarray(v) if isinstance(v,np.ndarray) else v for k,v in opts.items()}
    raw=core.moNMTF(xp.asarray(a),xp.asarray(b),xp.asarray(c),opts)
    if backend=="gpu":xp.cuda.Stream.null.synchronize()
    result=_host(raw,xp)
    bfit,xfit,cfit,aug,obj,_,_,info=result
    if not all(np.isfinite(v).all() for v in [bfit,xfit,cfit]):
        raise FloatingPointError("optimizer returned non-finite factors")
    info.update(backend=backend,variant=variant,p=p,normalized_expression=bool(normalize),
                matrix_orientation="B:target_by_regulator; C:sender_by_receiver",spatial=spatial)
    if not info.get("converged",False):
        warnings.warn(f"result retained with stop_reason={info.get('stop_reason')}",RuntimeWarning)
    return FitResult(np.asarray(bfit),np.asarray(xfit),np.asarray(cfit),list(obj),list(aug),info)

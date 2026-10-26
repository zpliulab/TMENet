"""Lazy CPU/CuPy execution of the frozen simulation optimizer; raw eight outputs."""
import importlib.util
from pathlib import Path
import numpy as np
from tmenet import _core_cpu
from tmenet._lp_prox import fractional_prox, fractional_prox_gpu
from tmenet.model import _host


def run(a,b,c,options,gpu=False):
    xp=np
    if gpu:
        import cupy as xp
        if xp.cuda.runtime.getDeviceCount()<1:raise RuntimeError('GPU requested but no CUDA device available')
    spec=importlib.util.spec_from_file_location('_legacy_isolated_core',Path(_core_cpu.__file__))
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    core.np=xp
    original=core.prox_p
    core.prox_p=lambda d,alpha,p,eps=1e-12: (
        fractional_prox_gpu(d,alpha,p) if gpu else fractional_prox(d,alpha,p,np)
    ) if 0<p<1 else original(d,alpha,p,eps)
    opts={k:xp.asarray(v) if isinstance(v,np.ndarray) else v for k,v in options.items()}
    result=core.moNMTF(xp.asarray(a),xp.asarray(b),xp.asarray(c),opts)
    if gpu:xp.cuda.Stream.null.synchronize()
    return _host(result,xp)


def cpu_moNMTF(a,b,c,options):return run(a,b,c,options,False)
def gpu_moNMTF(a,b,c,options):return run(a,b,c,options,True)

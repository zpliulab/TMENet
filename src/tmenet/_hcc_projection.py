"""Correct weighted Euclidean projection, isolated from the shared optimizer."""
import numpy as np

def weighted_ball_projection(M, M_star, beta, W, eps=1e-12):
    from ._core_real import get_array_module
    xp=get_array_module(M)
    if beta < 0:
        raise ValueError('beta must be nonnegative')
    delta=M-M_star
    w2=W**2
    initial=float(xp.sum(w2*delta**2))
    if initial <= beta**2:
        return M.copy()
    positive=w2[w2>0]
    if positive.size == 0:
        return M.copy()
    if beta == 0:
        return M_star+xp.where(w2>0,0,delta)
    lower=0.0
    upper=max(1.0,(np.sqrt(initial)/beta-1.0)/float(xp.min(positive)))
    multiplier=upper
    for _ in range(80):
        projected=delta/(1+multiplier*w2)
        residual=float(xp.sum(w2*projected**2))-beta**2
        # The converged multiplier, not a stale upper endpoint, defines the projection.
        if abs(residual) <= 1e-12*max(beta**2,1e-24):
            return M_star+projected
        if residual>0:
            lower=multiplier
        else:
            upper=multiplier
        derivative=-2*float(xp.sum(w2**2*delta**2/(1+multiplier*w2)**3))
        proposal=multiplier-residual/derivative if derivative<0 else np.nan
        multiplier=proposal if np.isfinite(proposal) and lower<proposal<upper else (lower+upper)/2
    return M_star+delta/(1+upper*w2)

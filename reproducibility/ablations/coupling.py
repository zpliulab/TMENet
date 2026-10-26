"""Fixed lognormal stress grid; no metric-dependent choices."""
import hashlib
import numpy as np

PROTOCOL='randomized_c_lognormal_noise_mass_matched_stress_v1'
SIGMAS=(0.0,0.25,0.5,1.0,2.0)
REPEATS=10

def add_n(coupling,mask,case,repeat,sigma):
    if sigma not in SIGMAS or not 0<=repeat<REPEATS:
        raise ValueError('outside prespecified sigma/repeat grid')
    c=np.asarray(coupling,dtype=float);m=np.asarray(mask,dtype=bool)
    if c.shape!=m.shape or c.shape[0]!=c.shape[1] or np.any(np.diag(m)):
        raise ValueError('invalid coupling or candidate mask')
    if not np.isfinite(c).all() or np.any(c<0):raise ValueError('invalid coupling values')
    np.testing.assert_allclose(c,c.T,rtol=0,atol=1e-12)
    digest=hashlib.sha256(f'{PROTOCOL}|{case}|{repeat}'.encode()).hexdigest()
    rng=np.random.Generator(np.random.PCG64(int(digest[:16],16)))
    union=m|m.T
    first,second=np.where(np.triu(union,k=1))
    z=rng.standard_normal(len(first))
    result=c.copy()
    total=float(c[m].sum())
    factor=1.0
    if sigma>0:
        noisy=c[first,second]*np.exp(sigma*z-0.5*sigma*sigma)
        result[first,second]=noisy;result[second,first]=noisy
        before=float(result[m].sum())
        if total>0:
            if before<=0 or not np.isfinite(before):raise ValueError('invalid perturbed mass')
            factor=total/before
            result[first,second]*=factor;result[second,first]*=factor
    np.testing.assert_allclose(result,result.T,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(result[~union],c[~union])
    np.testing.assert_array_equal(result[m]==0,c[m]==0)
    np.testing.assert_allclose(result[m].sum(),total,rtol=1e-12,atol=1e-10)
    if not np.isfinite(result).all():raise ValueError('nonfinite noise output')
    relative=float(np.linalg.norm(result[m]-c[m])/np.linalg.norm(c[m])) if total else 0.
    values=result[m]
    audit={'sigma':sigma,'repeat':repeat,'noise_seed_hex':digest,
        'same_standard_normal_draw_across_sigmas':True,'normalization_factor':factor,
        'mass_before':total,'mass_after':float(values.sum()),
        'absolute_mass_error':abs(float(values.sum())-total),
        'relative_l2_change_from_permuted':relative,
        'candidate_weight_cv':float(values.std()/values.mean()) if total else 0.,
        'max_candidate_weight':float(values.max()),'value_distribution_preserved':sigma==0,
        'symmetry_preserved':True,'zero_support_preserved':True}
    return result,audit

def classification_metrics(y,score,lr):
    y=np.asarray(y,dtype=int);score=np.asarray(score,dtype=float)
    if not np.isfinite(score).all() or np.unique(y).size!=2:raise ValueError('invalid evaluation input')
    precision,recall,_=lr.precision_recall_curve(y,score)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(precision),where=(precision+recall)>0)
    return {'auroc':float(lr.roc_auc_score(y,score)),
        'average_precision':float(lr.average_precision_score(y,score)),
        'f1_max':float(f1.max())}

"""Exact scalar/separable proximal map for alpha*|x|**p, 0<p<1.

For 0.5*(x-y)^2 + alpha*|x|^p the nonzero global minimizer lies
above x0=(2*alpha*(1-p))**(1/(2-p)). It is preferred to zero only
when |y|>x0+alpha*p*x0**(p-1). Newton solves the larger stationary
root on this monotone, convex branch; zero is selected at the tie.
"""

def fractional_prox(values, alpha, p, xp, iterations=16):
    if not 0 < p < 1:
        raise ValueError('fractional_prox requires 0<p<1')
    if alpha < 0:
        raise ValueError('alpha must be nonnegative')
    if alpha == 0:
        return values.copy()
    magnitude=xp.abs(values)
    x0=(2.0*alpha*(1.0-p))**(1.0/(2.0-p))
    threshold=x0+alpha*p*x0**(p-1.0)
    root=xp.maximum(magnitude,x0)
    for _ in range(iterations):
        residual=root+alpha*p*root**(p-1.0)-magnitude
        derivative=1.0+alpha*p*(p-1.0)*root**(p-2.0)
        root=xp.maximum(root-residual/derivative,x0)
    return xp.where(magnitude>threshold,xp.sign(values)*root,0.0)

_GPU_KERNEL=None
def fractional_prox_gpu(values, alpha, p):
    """Same float64 scalar map, fused to avoid 16 rounds of kernel launches."""
    global _GPU_KERNEL
    import cupy as cp
    if not 0<p<1 or alpha<0: raise ValueError('invalid fractional proximal parameters')
    if _GPU_KERNEL is None:
        _GPU_KERNEL=cp.ElementwiseKernel(
            'float64 value, float64 alpha, float64 p','float64 output',
            '''
            if (alpha == 0.0) { output = value; }
            else {
              double magnitude = fabs(value);
              double x0 = pow(2.0 * alpha * (1.0-p), 1.0/(2.0-p));
              double threshold = x0 + alpha*p*pow(x0,p-1.0);
              if (magnitude <= threshold) { output = 0.0; }
              else {
                double root = magnitude;
                for (int iteration=0; iteration<16; ++iteration) {
                  double residual = root + alpha*p*pow(root,p-1.0) - magnitude;
                  double derivative = 1.0 + alpha*p*(p-1.0)*pow(root,p-2.0);
                  root = fmax(root-residual/derivative,x0);
                }
                output = copysign(root,value);
              }
            }
            ''','six_p_exact_fractional_prox_float64')
    return _GPU_KERNEL(values,alpha,p)

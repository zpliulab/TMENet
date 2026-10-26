"""Truth-blind, symmetry-preserving permutation on a frozen spatial graph."""
import hashlib
import numpy as np

PROTOCOL='symmetric_c_permutation_by_candidate_reciprocity_v1'
REPEATS=10

def permute_coupling(coupling, mask, case, repeat):
    if not 0 <= repeat < REPEATS:
        raise ValueError('repeat outside frozen 0..9 grid')
    c=np.asarray(coupling,dtype=float)
    m=np.asarray(mask,dtype=bool)
    if c.shape!=m.shape or c.ndim!=2 or c.shape[0]!=c.shape[1]:
        raise ValueError('coupling and mask shapes differ')
    if np.any(np.diag(m)) or not np.isfinite(c).all() or np.any(c<0):
        raise ValueError('invalid diagonal, nonfinite value, or negative coupling')
    np.testing.assert_allclose(c,c.T,rtol=0,atol=1e-12)
    seed_hex=hashlib.sha256(f'{PROTOCOL}|{case}|{repeat}'.encode()).hexdigest()
    seed=int(seed_hex[:16],16)
    rng=np.random.Generator(np.random.PCG64(seed))
    result=c.copy()
    union=m|m.T
    reciprocal=m&m.T
    info=[]
    for label,admitted in [('reciprocal',reciprocal),('one_way',union&~reciprocal)]:
        first,second=np.where(np.triu(admitted,k=1))
        original=c[first,second]
        order=rng.permutation(len(original))
        shuffled=original[order]
        result[first,second]=shuffled
        result[second,first]=shuffled
        info.append({'stratum':label,'unordered_pairs':len(original),
                     'changed_values':int(np.count_nonzero(shuffled!=original))})
    # Reciprocal pairs occur twice in evaluated M, one-way pairs once. Keeping
    # these strata separate preserves BOTH unordered and evaluated histograms.
    np.testing.assert_array_equal(np.sort(result[m]),np.sort(c[m]))
    np.testing.assert_array_equal(result[~union],c[~union])
    np.testing.assert_array_equal(np.diag(result),np.diag(c))
    np.testing.assert_allclose(result,result.T,rtol=0,atol=1e-12)
    audit={'repeat':repeat,'seed_hex':seed_hex,'pcg64_seed':seed,
           'candidate_values':int(m.sum()),'changed_candidate_values':int(np.count_nonzero(result[m]!=c[m])),
           'unchanged_candidate_values':bool(np.array_equal(result[m],c[m])),
           'strata':info,'candidate_value_multiset_preserved':True,
           'symmetry_preserved':True,'outside_candidate_union_unchanged':True}
    return result,audit

class AttributionCache:
    """The frozen competitive LR formula, with expression shares cached once."""
    def __init__(self,table,expression,metadata,mask,rank_function):
        ranked=rank_function(expression)
        genes={g:i for i,g in enumerate(ranked.index)}
        pairs=table[['ligand','receptor']].drop_duplicates().reset_index(drop=True)
        index={(r.ligand,r.receptor):i for i,r in enumerate(pairs.itertuples(index=False))}
        self.sender,self.receiver=np.where(mask)
        values=ranked.to_numpy(dtype=float)
        ligand=np.array([genes[g] for g in pairs['ligand']])
        receptor=np.array([genes[g] for g in pairs['receptor']])
        support=values[ligand[:,None],self.sender]*values[receptor[:,None],self.receiver]
        total=support.sum(axis=0)
        self.shares=np.divide(support,total[None,:],out=np.zeros_like(support),where=total[None,:]>0)
        self.supported=total>0
        self.groups=[]
        types=metadata['cell_type'].to_numpy()
        for (sender_type,receiver_type),rows in table.groupby(['sender_type','receiver_type'],sort=False).groups.items():
            edge=(types[self.sender]==sender_type)&(types[self.receiver]==receiver_type)
            if not edge.any():raise ValueError('empty type-pair candidate set')
            row_indices=np.asarray(list(rows),dtype=int)
            pair_indices=np.asarray([index[(table.loc[r,'ligand'],table.loc[r,'receptor'])] for r in row_indices])
            # C-contiguous rows use the same per-LR reduction as the archived
            # evaluator. Numerical agreement is gated against its slow formula.
            block=np.ascontiguousarray(self.shares[:,edge])
            self.groups.append((row_indices,pair_indices,edge,block))
        self.rows=len(table)

    def score(self,coupling):
        edge_values=np.asarray(coupling)[self.sender,self.receiver]
        output=np.zeros(self.rows,dtype=float)
        for rows,indices,edges,shares in self.groups:
            means=(shares*edge_values[edges][None,:]).mean(axis=1)
            output[rows]=means[indices]
        reconstructed=self.shares.sum(axis=0)*edge_values
        expected=edge_values*self.supported
        error=float(np.max(np.abs(reconstructed-expected)))
        if error>1e-8 or not np.isfinite(output).all():
            raise ValueError('cached attribution failed conservation or finiteness')
        return output,error

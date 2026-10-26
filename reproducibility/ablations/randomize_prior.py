"""Truth-independent weighted, degree-preserving ATAC-prior null model."""
import hashlib
import numpy as np

SCHEME='directed_double_edge_swap_preserve_degrees_row_weight_multisets_shared_blocks_v1'

def stable_seed(key):
    return int.from_bytes(hashlib.sha256(('random_ATAC_simdata_v3_v1|'+key).encode()).digest()[:4],'little')

def array_hash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode())
    h.update(a.tobytes())
    return h.hexdigest()

def randomize_block(block,rng):
    b=np.asarray(block,dtype=float)
    if b.ndim!=2 or b.shape[0]!=b.shape[1] or not np.isfinite(b).all() or np.any(b<0):
        raise ValueError('prior block must be finite, nonnegative and square')
    n=len(b)
    support=b!=0
    off=support.copy();np.fill_diagonal(off,False)
    edges=np.argwhere(off).astype(np.int64)
    weights=b[off].copy()
    occupied=off.copy()
    m=len(edges);accepted=attempted=0
    target=10*m
    budget=max(500,200*m)
    if m>=2:
        while attempted<budget and accepted<target:
            proposals=rng.randint(m,size=(min(4096,budget-attempted),2))
            for first,second in proposals:
                attempted+=1
                if first==second: continue
                a,x=edges[first];c,y=edges[second]
                if a==c or x==y or a==y or c==x: continue
                if occupied[a,y] or occupied[c,x]: continue
                occupied[a,x]=False;occupied[c,y]=False
                occupied[a,y]=True;occupied[c,x]=True
                edges[first,1]=y;edges[second,1]=x
                accepted+=1
                if accepted>=target: break
    result=np.zeros_like(b)
    if m:result[edges[:,0],edges[:,1]]=weights
    result[np.diag_indices(n)]=np.diag(b)
    after=result!=0
    np.testing.assert_array_equal(support.sum(0),after.sum(0))
    np.testing.assert_array_equal(support.sum(1),after.sum(1))
    np.testing.assert_array_equal(np.diag(b),np.diag(result))
    np.testing.assert_array_equal(np.sort(b.ravel()),np.sort(result.ravel()))
    for row in range(n):
        np.testing.assert_array_equal(np.sort(b[row]),np.sort(result[row]))
    union=np.count_nonzero(support|after)
    return result,{'accepted_swaps':accepted,'attempts':attempted,'target_swaps':target,
        'mixing_target_reached':accepted>=target,'off_diagonal_edges':m,
        'changed_support_positions':int(np.count_nonzero(support!=after)),
        'changed_weight_positions':int(np.count_nonzero(b!=result)),
        'support_jaccard':float(np.count_nonzero(support&after)/union) if union else 1.0,
        'unchanged':bool(np.array_equal(b,result))}

def randomize_prior(prior,gene_num,key):
    b=np.asarray(prior,dtype=float)
    if b.ndim!=2 or b.shape[0]!=b.shape[1] or len(b)%gene_num:
        raise ValueError('incompatible block prior')
    blocks=len(b)//gene_num
    for i in range(blocks):
        for j in range(blocks):
            if i!=j and np.any(b[i*gene_num:(i+1)*gene_num,j*gene_num:(j+1)*gene_num]):
                raise ValueError('unexpected nonzero off-block prior')
    original=[b[i*gene_num:(i+1)*gene_num,i*gene_num:(i+1)*gene_num] for i in range(blocks)]
    shared=all(np.array_equal(original[0],v) for v in original[1:])
    rng=np.random.RandomState(stable_seed(key))
    replacements=[];details=[]
    for original_block in original[:1] if shared else original:
        value,meta=randomize_block(original_block,rng)
        replacements.append(value);details.append(meta)
    if shared: replacements=replacements*blocks
    result=np.zeros_like(b)
    for i,value in enumerate(replacements):
        result[i*gene_num:(i+1)*gene_num,i*gene_num:(i+1)*gene_num]=value
    return result,{'scheme':SCHEME,'seed':stable_seed(key),'case':key,'blocks':blocks,
        'identical_input_blocks':shared,'shared_randomization_across_identical_blocks':shared,
        'input_hash':array_hash(b),'randomized_hash':array_hash(result),
        'source_nonzero_count':int(np.count_nonzero(b)),
        'randomized_nonzero_count':int(np.count_nonzero(result)),
        'input_Frobenius_norm':float(np.linalg.norm(b)),
        'randomized_Frobenius_norm':float(np.linalg.norm(result)),
        'block_diagnostics':details,'truth_accessed_for_randomization':False}

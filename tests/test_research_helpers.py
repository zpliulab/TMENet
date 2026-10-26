"""Tests requiring neither patient data nor optional baseline runtimes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from tmenet import fit_generalized
from tmenet._hcc_projection import weighted_ball_projection
from test_public_api import fixture, options

ROOT=Path(__file__).resolve().parents[1]
ABLATIONS=ROOT/'reproducibility/ablations'
sys.path.insert(0,str(ABLATIONS))
from randomize_prior import randomize_prior
from permutation import permute_coupling
from coupling_noise import add_noise


@pytest.mark.parametrize('corrected',[False,True])
def test_real_core_joint_finite_outputs(corrected):
    *_,data=fixture()
    with pytest.warns(RuntimeWarning):
        result=fit_generalized(data['A'],data['B_prior'],data['C_prior'],
                              options=options(hcc_corrected_projection=corrected),
                              B_mask=data['B_mask'],M_knn=data['M_knn'],variant='real')
    assert result.diagnostics['variant']=='real'
    assert np.isfinite(result.B).all() and np.isfinite(result.C).all()
    assert np.all(result.B[data['B_mask']==0]==0)


def test_corrected_projection_satisfies_weighted_boundary():
    original=np.arange(16,dtype=float).reshape(4,4)/3
    prior=np.zeros((4,4));weights=np.linspace(.1,2,16).reshape(4,4)
    fitted=weighted_ball_projection(original,prior,.7,weights)
    assert np.isclose(np.linalg.norm(weights*(fitted-prior)),.7,rtol=1e-9)
    assert np.linalg.norm(fitted-original)<np.linalg.norm(original)
    np.testing.assert_array_equal(weighted_ball_projection(original,prior,0,weights),prior)


def test_random_atac_invariants_and_determinism():
    n=7;block=np.zeros((n,n))
    for i in range(n):
        for shift in [1,2]:block[i,(i+shift)%n]=(i+shift)/10
    full=np.kron(np.eye(2),block)
    randomized,audit=randomize_prior(full,n,'fixture')
    repeat,_=randomize_prior(full,n,'fixture')
    np.testing.assert_array_equal(randomized,repeat)
    np.testing.assert_array_equal((full>0).sum(0),(randomized>0).sum(0))
    np.testing.assert_array_equal((full>0).sum(1),(randomized>0).sum(1))
    np.testing.assert_array_equal(np.sort(full,axis=1),np.sort(randomized,axis=1))
    np.testing.assert_array_equal(randomized[:n,:n],randomized[n:,n:])
    assert audit['truth_accessed_for_randomization'] is False
    assert not np.array_equal(full,randomized)


def test_coupling_permutation_noise_mass_and_support():
    rng=np.random.default_rng(77)
    values=rng.uniform(0,2,(9,9));c=(values+values.T)/2
    np.fill_diagonal(c,0)
    mask=rng.random((9,9))>.4;np.fill_diagonal(mask,False)
    permuted,audit=permute_coupling(c,mask,'fixture',0)
    np.testing.assert_array_equal(np.sort(permuted[mask]),np.sort(c[mask]))
    noisy,noise_audit=add_noise(permuted,mask,'fixture',0,2.)
    repeat,_=add_noise(permuted,mask,'fixture',0,2.)
    np.testing.assert_array_equal(noisy,repeat)
    np.testing.assert_allclose(noisy,noisy.T,atol=1e-12)
    assert np.isclose(noisy[mask].sum(),c[mask].sum(),rtol=1e-12)
    assert noise_audit['value_distribution_preserved'] is False
    assert audit['candidate_value_multiset_preserved'] is True


def test_legacy_adapter_preserves_closed_form_p1():
    sys.path.insert(0,str(ROOT/'reproducibility/simulation'))
    from optimizer_adapter import cpu_moNMTF
    from tmenet import _core_cpu
    *_,data=fixture()
    opts=options(B_mask=data['B_mask'],M_knn=data['M_knn'],return_info=True)
    actual=cpu_moNMTF(data['A'],data['B_prior'],data['C_prior'],opts)
    expected=_core_cpu.moNMTF(data['A'],data['B_prior'],data['C_prior'],opts)
    for x,y in zip(actual[:3],expected[:3]):np.testing.assert_array_equal(x,y)


def test_recorded_six_p_ledger_is_complete():
    import pandas as pd
    ledger=pd.read_csv(ROOT/'reproducibility/simulation/iteration_budgets.csv')
    assert len(ledger)==9000 and not ledger.duplicated(['p_label','case']).any()
    assert set(ledger.p_label)=={'p0','p1_3','p1_2','p2_3','p1','p2'}
    assert (ledger.groupby('p_label').size()==1500).all()
    assert (ledger.iterations<=ledger.iteration_budget).all()


def test_public_lr_matches_reference_benchmark():
    from tmenet import attribute_lr
    sys.path.insert(0,str(ROOT/'reproducibility/benchmarks'))
    from benchmark_lr_contribution_v2 import add_conserved_coupling_attribution
    expression,metadata,_,_,pairs,data=fixture()
    public,_=attribute_lr(expression,data['C_prior'],metadata,pairs,candidate_mask=data['M_knn'])
    keys=['sender_type','receiver_type','ligand','receptor']
    reference,_=add_conserved_coupling_attribution(public[keys],expression,data['C_prior'],
                                                   metadata.set_index('cell'),data['M_knn'].astype(bool),'reference')
    np.testing.assert_allclose(public.attributed_mean,reference.reference_lr_conserved_attribution,rtol=1e-13)


def test_cli_fit_and_lr_with_reordered_expression(tmp_path):
    import pandas as pd
    from tmenet.cli import main
    fit_dir=tmp_path/'fit'
    main(['fit','--input',str(ROOT/'examples/toy/inputs.npz'),'--config',
          str(ROOT/'configs/toy.json'),'--max-iter','10','--output',str(fit_dir)])
    assert (fit_dir/'B.npy').exists() and (fit_dir/'C.npy').exists()
    expr=pd.read_csv(ROOT/'examples/toy/expression.tsv',sep='\t',index_col=0)
    reverse=tmp_path/'reverse.tsv';expr.iloc[:,::-1].to_csv(reverse,sep='\t')
    outputs=[]
    for i,expression in enumerate([ROOT/'examples/toy/expression.tsv',reverse]):
        out=tmp_path/f'lr_{i}.tsv'
        main(['lr','--expression',str(expression),'--coupling',str(fit_dir/'C.npy'),
              '--metadata',str(ROOT/'examples/toy/metadata.tsv'),'--pairs',
              str(ROOT/'examples/toy/lr_pairs.tsv'),'--mask',str(ROOT/'examples/toy/candidate_mask.npy'),
              '--output',str(out)])
        outputs.append(pd.read_csv(out,sep='\t'))
    pd.testing.assert_frame_equal(*outputs)

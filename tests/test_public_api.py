import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from tmenet import fit_generalized,prepare_inputs,attribute_lr,spatial_neighborhood
from tmenet._lp_prox import fractional_prox
from tmenet.lr import positive_percentiles
from tmenet.metrics import binary_metrics
from tmenet.pathways import prioritize_pathways

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    rng=np.random.default_rng(7)
    genes=["TF","L","R","G"];cells=[f"c{i}" for i in range(8)]
    expression=pd.DataFrame(rng.uniform(.1,5,(4,8)),index=genes,columns=cells)
    metadata=pd.DataFrame({"cell":cells,"cell_type":["A"]*4+["B"]*4})
    coordinates=pd.DataFrame([(0,0),(1,.1),(0,1),(1,1),(2,.2),(2,1),(3,0),(3,1)],index=cells)
    prior=pd.DataFrame({"regulator":["TF"],"target":["G"]})
    pairs=pd.DataFrame({"ligand":["L"],"receptor":["R"]})
    data=prepare_inputs(expression,metadata,coordinates,prior,pairs)
    return expression,metadata,coordinates,prior,pairs,data


def options(**extra):
    config=json.loads((ROOT/"configs/toy.json").read_text())["options"]
    config.update(maxIter=5,minIter=0,adaptive_step=False)
    config.update(extra)
    return config


def test_preparation_orientation_and_masks():
    *_,data=fixture()
    assert data["A"].shape==(8,8)
    assert data["B_prior"][3,0]==1
    assert data["B_prior"][0,3]==0
    assert np.all(data["C_prior"][data["M_knn"]==0]==0)
    assert data["spatial_info"]["selected_k"]>=3


def test_wrapper_matches_reference_core_p1():
    *_,data=fixture();opts=options()
    with pytest.warns(RuntimeWarning):
        result=fit_generalized(data["A"],data["B_prior"],data["C_prior"],options=opts,
                              B_mask=data["B_mask"],M_knn=data["M_knn"])
    from tmenet import _core_cpu
    reference_opts={**opts,"B_mask":data["B_mask"],"M_knn":data["M_knn"],"return_info":True,
                    "use_gpu":False,"zero_diag_C":False,"row_normalize_C":False,"mask_conflict_policy":"error","mask_conflict_tolerance":0.}
    a=np.log1p(data["A"]);a=(a-a.min())/(a.max()-a.min()+1e-12)
    reference=_core_cpu.moNMTF(a,data["B_prior"],data["C_prior"],reference_opts)
    for got,expected in zip([result.B,result.X,result.C],reference[:3]):np.testing.assert_array_equal(got,expected)


@pytest.mark.parametrize("p",[0,1/3,.5,2/3,1,2])
def test_all_p_return_joint_finite_factors(p):
    *_,data=fixture()
    with pytest.warns(RuntimeWarning):
        result=fit_generalized(data["A"],data["B_prior"],data["C_prior"],options=options(p=p),
                              B_mask=data["B_mask"],M_knn=data["M_knn"])
    assert np.isfinite(result.B).all() and np.isfinite(result.C).all()
    assert result.diagnostics["iterations"]==5
    assert np.all(result.B[data["B_mask"]==0]==0)
    assert np.all(result.C[data["M_knn"]==0]==0)


@pytest.mark.parametrize("p",[1/3,.5,2/3])
def test_fractional_prox_against_scalar_minimizer(p):
    alpha=.2
    for y in [.01,.2,1.,3.]:
        got=float(fractional_prox(np.array([y]),alpha,p,np)[0])
        objective=lambda x:.5*(x-y)**2+alpha*abs(x)**p
        opt=minimize_scalar(objective,bounds=(1e-12,max(4.,y*2)),method="bounded")
        expected=min(objective(0.),opt.fun)
        assert abs(objective(got)-expected)<1e-7


def test_rank_and_lr_conservation():
    np.testing.assert_array_equal(positive_percentiles([[0,2,2,4]]),[[0,.5,.5,1]])
    expression,metadata,_,_,pairs,data=fixture()
    result,audit=attribute_lr(expression,data["C_prior"],metadata,pairs,candidate_mask=data["M_knn"])
    assert audit["max_conservation_error"]<1e-12
    assert audit["unallocated_coupling"]==0
    for _,group in result.groupby(["sender_type","receiver_type"]):
        assert abs(group.attribution_share_percent.sum()-100)<1e-10


def test_unsupported_coupling_is_reported():
    expression,metadata,_,_,pairs,data=fixture();expression.loc["L"]=0
    result,audit=attribute_lr(expression,data["C_prior"],metadata,pairs,candidate_mask=data["M_knn"])
    assert result.attributed_sum.sum()==0 and audit["unallocated_coupling"]>0


def test_invalid_masks_and_duplicate_coordinates_rejected():
    *_,data=fixture();bad=data["C_prior"].copy();np.fill_diagonal(bad,1)
    with pytest.raises(ValueError,match="conflicts"):
        fit_generalized(data["A"],data["B_prior"],bad,options=options(),B_mask=data["B_mask"],M_knn=data["M_knn"])
    with pytest.raises(ValueError,match="duplicate"):
        spatial_neighborhood([[0,0],[0,0],[1,1]])


def test_saved_output_keeps_b_and_c_and_stop_status(tmp_path):
    *_,data=fixture()
    with pytest.warns(RuntimeWarning):
        result=fit_generalized(data["A"],data["B_prior"],data["C_prior"],options=options(),B_mask=data["B_mask"],M_knn=data["M_knn"])
    out=tmp_path/"run";result.save(out)
    assert (out/"B.npy").exists() and (out/"C.npy").exists()
    assert json.loads((out/"diagnostics.json").read_text())["stop_reason"]=="max_iter"
    with pytest.raises(FileExistsError):result.save(out)


def test_metrics_are_explicit_max_f1():
    out=binary_metrics([0,1,0,1],[.1,.9,.2,.8])
    assert out["auroc"]==out["average_precision"]==out["f1_max"]==1
    with pytest.raises(ValueError):binary_metrics([1,1],[.1,.2])


def test_pathways_require_curated_route_and_codetection():
    expression,metadata,_,_,pairs,data=fixture()
    lr,_=attribute_lr(expression,data["C_prior"],metadata,pairs,candidate_mask=data["M_knn"])
    grn=pd.DataFrame({"cell_type":["B"],"regulator":["TF"],"target":["G"],"weight":[.7]})
    routes=pd.DataFrame({"receptor":["R"],"tf":["TF"],"route":["R;signal;TF"],"source_database":["toy_evidence"]})
    assert len(prioritize_pathways(lr,expression,metadata,routes,grn,min_codetected=1))>0
    assert prioritize_pathways(lr,expression,metadata,routes,grn,min_codetected=20).empty

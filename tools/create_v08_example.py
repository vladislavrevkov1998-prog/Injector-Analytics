"""Reproducible synthetic examples; never overwrite the user's measured field."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import Settings,default_nozzles,project_to_json
from fuelfield.air_sources import default_air_model

ROOT=Path(__file__).resolve().parents[1]


def synthetic():
    cfg=Settings(z_mm=[0.,20.,40.,60.,100.],n_sample_radial=16,n_sample_angular=144)
    ns=default_nozzles(cfg)
    air=default_air_model();air.update(enabled=True,total_kg_s=1.,l0=15.,distribution='directed',
        interaction_strength=.5,step_mm=2.,max_path_mm=140.,
        provenance={'origin':'SYNTHETIC: all numerical inputs are educational; no experiment'})
    for n in ns:
        n.flow_kg_h=1.;n.sector_ml=[float(1+.25*np.cos(np.deg2rad(10*k+n.id))) for k in range(36)]
        n.sector_source='SYNTHETIC 36-bin test distribution; not experimental'
        air['sources'].append(dict(id=f'S{n.id}',kind='swirler',nozzle_id=n.id,method='fixed',
            flow_kg_s=float(.003*(1+.2*np.cos(np.deg2rad(n.theta_deg)))),direction=[1.,0.,0.],
            sigma_r_mm=12.,sigma_theta_deg=5.,mixed_fraction=.15,mixing_length_mm=60.,
            confidence='synthetic',provenance={'origin':'SYNTHETIC individual swirler flow; not measured'}))
    for index,theta in enumerate([0.,40.,100.,180.,250.,300.]):
        air['sources'].append(dict(id=f'H{index+1}',kind='hole',method='fraction',share=.04,
            z_mm=20.,radius_mm=295.,theta_deg=theta,direction=[.9,-.25,.1],area_mm2=20.,
            sigma_r_mm=8.,sigma_theta_deg=8.,spread_mm_per_mm=.08,mixed_fraction=.1,mixing_length_mm=40.,
            confidence='synthetic',provenance={'origin':'SYNTHETIC oblique hole, empirical widths'}))
    air['sources'].append(dict(id='C1',kind='cooling',method='fraction',share=.1,z_mm=40.,radius_mm=202.,theta_deg=20.,
        direction=[1.,0.,0.],kernel='wall',wall='free',sigma_r_mm=3.,sigma_theta_deg=25.,
        mixed_fraction=.2,mixing_length_mm=60.,confidence='synthetic',provenance={'origin':'SYNTHETIC wall layer'}))
    cfg.air_model=air;cfg.validate()
    return project_to_json(cfg,ns)


def didenko():
    cfg=Settings(z_mm=[float(k*10) for k in range(11)],n_sample_radial=8,n_sample_angular=72)
    ns=default_nozzles(cfg)
    for n in ns:
        n.flow_kg_h=3600/(140*2.6*15.)
        # The 1D book check needs a constant eligible fuel flow. An explicitly
        # synthetic narrow nonexpanding kernel stays wholly within the annulus.
        n.alpha_deg=0.;n.k_linear=0.
        n.sector_source='SYNTHETIC uniform 36 bins for the 1D ledger; not experimental'
    air=default_air_model();air.update(enabled=True,total_kg_s=1.,l0=15.,distribution='uniform',max_path_mm=120.,
        provenance={'origin':'Didenko 2012 III printed pp8-13/19-20; spatial uniform extension SYNTHETIC; L0=15 is educational'})
    def source(identity,kind,flow,points):
        return dict(id=identity,kind=kind,method='fixed',flow_kg_s=flow,z_mm=points[0][0],radius_mm=250.,
            kernel='uniform',entrainment_points=points,provenance={'origin':'Didenko printed pp8-13 conditional involvement'},confidence='reference arithmetic')
    air['sources'].append(source('front','other',.25,[[0.,1.]]))
    for identity,points in [('jet3',[[10.,.2],[20.,.4],[30.,1.]]),('jet5',[[40.,.4],[50.,1.]]),('jet7',[[60.,.4],[70.,1.]])]:
        air['sources'].append(source(identity,'hole',.1346,points))
    air['sources'].append(source('jet9','insert',.0962,[[80.,.4],[90.,1.]]))
    for k,z in enumerate([10.,30.,50.,70.,90.]):air['sources'].append(source(f'cooling{k+1}','cooling',.05,[[z,.4],[z+10,1.]]))
    cfg.air_model=air;cfg.validate();return project_to_json(cfg,ns)


if __name__=='__main__':
    for name,data in [('demo_project_v08_air_synthetic.json',synthetic()),('demo_project_v08_didenko_reference.json',didenko())]:
        path=ROOT/'examples'/name;path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(path)

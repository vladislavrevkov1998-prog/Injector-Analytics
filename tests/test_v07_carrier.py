"""Independent invariants of the explicitly prescribed axisymmetric carrier.

The numerical conservation checks integrate actual element fluxes; the
curved-wall checks include a vertical tangent where radial and normal chord
errors differ substantially. No experimental validation is implied.
"""
from copy import deepcopy
import math
import unittest

import numpy as np

from fuelfield.carrier import build_carrier, clear_carrier_cache
from fuelfield.chamber import Chamber, UNITS


def duct(inner=(100., 100.), outer=(160., 160.), length=160., unit='mm'):
    factor=UNITS[unit]
    primitives=[]
    for wall,ends in (('inner',inner),('outer',outer)):
        primitives.append(dict(id=wall,kind='line',wall=wall,unit=unit,
            start=[0.,ends[0]/factor],end=[length/factor,ends[1]/factor]))
    return dict(schema_version=2,enabled=True,base_z_mm=0.,outlet_z_mm=length,
                primitives=primitives,objects=[],groups=[],constraints=[],
                sections=[dict(id='S1',z_mm=length,is_outlet=True)])


def vertical_arc_duct(unit='mm'):
    raw=duct((210.,200.),(300.,300.),length=10.,unit=unit)
    factor=UNITS[unit]
    raw['primitives'][0]=dict(id='inner',kind='arc',wall='inner',unit=unit,
        start=[0.,210./factor],end=[10./factor,200./factor],
        center=[10./factor,210./factor],clockwise=False)
    return raw


def discrete_flux(field,z):
    """Exact quadrature of r*uz on a linear-wall cross-section.

    A vertical cut crosses each radial grid line and each triangle diagonal.
    Within each resulting segment r*uz is constant for this P1 streamfunction.
    """
    i=int(np.clip(np.searchsorted(field.z_mm,z,side='right')-1,0,len(field.z_mm)-2))
    t=(z-field.z_mm[i])/(field.z_mm[i+1]-field.z_mm[i])
    radial=(1-t)*field.r_mm[i]+t*field.r_mm[i+1]
    diagonal=(1-t)*field.r_mm[i,:-1]+t*field.r_mm[i+1,1:]
    cuts=np.unique(np.r_[radial,diagonal])
    midpoint=(cuts[1:]+cuts[:-1])/2
    velocity=field.meridional_velocity(z,midpoint)[:,0]
    return float(2*math.pi*np.sum((midpoint*.001)*velocity*np.diff(cuts*.001)))


class TestCarrier(unittest.TestCase):
    def setUp(self):clear_carrier_cache()

    def test_cylinder_is_exact_uniform_flow_and_analytic_streamfunction(self):
        field=build_carrier(duct(),20.,max_step_mm=37.,radial_cells=3)
        z=np.array([0.,28.,160.])[:,None]
        r=np.array([100.,113.,141.,160.])[None,:]
        v=field.meridional_velocity(z,r)
        np.testing.assert_allclose(v[...,0],20.,atol=1e-14)
        np.testing.assert_allclose(v[...,1],0.,atol=0)
        expected=.5*20.*((np.broadcast_to(r,v.shape[:-1])*.001)**2-.1**2)
        np.testing.assert_allclose(field.streamfunction_at(z,r),expected,atol=1e-16)
        self.assertAlmostEqual(field.inlet_flux_m3_s,20.*math.pi*(.16**2-.1**2),places=14)
        self.assertEqual(field.diagnostics['solution_kind'],'analytic_cylindrical')

    def test_narrowing_and_widening_preserve_total_volume_flux_at_every_cut(self):
        for inner,outer in (((100.,115.),(160.,145.)),((115.,100.),(145.,160.))):
            with self.subTest(inner=inner,outer=outer):
                field=build_carrier(duct(inner,outer),20.,max_step_mm=17.,radial_cells=9)
                for z in (0.,2.3,17.,49.2,80.,133.7,160.):
                    self.assertAlmostEqual(discrete_flux(field,z),field.inlet_flux_m3_s,places=12)
                self.assertLess(field.weak_residual,1e-12)

    def test_sidewalls_have_no_pointwise_normal_flow(self):
        raw=duct((100.,115.),(160.,145.));model=Chamber(raw)
        field=build_carrier(raw,20.,max_step_mm=13.,radial_cells=7)
        z=np.linspace(0.,160.,31)
        for wall in ('inner','outer'):
            primitive=model.profiles[wall][0];r=primitive.at(z)
            v=field.meridional_velocity(z,r)
            np.testing.assert_allclose(v@primitive.normal(50.),0.,atol=3e-12)
        self.assertLess(field.diagnostics['max_discrete_wall_normal_velocity_m_s'],1e-11)

    def test_normal_flux_is_continuous_on_internal_triangle_edges(self):
        field=build_carrier(duct((100.,115.),(160.,145.)),20.,max_step_mm=20.,radial_cells=5)
        owners={}
        for triangle,nodes in enumerate(field.triangles):
            for a,b in ((nodes[0],nodes[1]),(nodes[1],nodes[2]),(nodes[2],nodes[0])):
                owners.setdefault(tuple(sorted((int(a),int(b)))),[]).append(triangle)
        for edge,triangles in owners.items():
            if len(triangles)!=2:continue
            points=field._coordinates_m[list(edge)];tangent=points[1]-points[0]
            normal=np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent)
            g=field._psi_gradient[triangles]
            weighted=np.column_stack((g[:,1],-g[:,0]))
            self.assertAlmostEqual(float(weighted[0]@normal),float(weighted[1]@normal),places=12)

    def test_curved_wall_sliver_and_vertical_tangent_do_not_reject_valid_points(self):
        raw=vertical_arc_duct();field=build_carrier(raw,20.,max_step_mm=2.,radial_cells=16,wall_tolerance_mm=.01)
        primitive=Chamber(raw).profiles['inner'][0]
        z=np.linspace(0.,10.,501);r=primitive.at(z)
        lower,_=field.mesh_bounds(z)
        # A sagitta <=.01 mm can have a larger radial gap near vertical.
        self.assertGreater(float(np.max(lower-r)),.01)
        velocity=field.meridional_velocity(z,r)
        self.assertTrue(np.isfinite(velocity).all())
        for point,zz in zip(velocity,z):self.assertLess(abs(float(point@primitive.normal(float(zz)))),1e-11)
        self.assertAlmostEqual(float(velocity[0,0]),0.,places=12)
        np.testing.assert_allclose(field.streamfunction_at(z,r),0.,atol=3e-16)
        # This is a point inside the exact wall but outside its coarse chord.
        near=r+1e-5
        self.assertTrue(np.isfinite(field.meridional_velocity(z,near)).all())

    def test_finite_points_outside_exact_walls_are_not_projected_inside(self):
        field=build_carrier(vertical_arc_duct(),20.,wall_tolerance_mm=.1)
        inner=Chamber(vertical_arc_duct()).bounds(.5)[0]
        for z,r in ((.5,inner-.001),(.5,300.001),(-.001,250.),(10.001,250.)):
            with self.subTest(z=z,r=r),self.assertRaises(ValueError):field.meridional_velocity(z,r)
        with self.assertRaises(ValueError):field.meridional_velocity(float('nan'),250.)
        with self.assertRaises(ValueError):field.velocity(.5,float('inf'),0.)

    def test_physical_units_and_endpoint_orientation_do_not_change_solution(self):
        reference=build_carrier(vertical_arc_duct(),20.,max_step_mm=1.,radial_cells=12)
        z=np.array([.2,2.3,7.9]);r=np.array([245.,260.,285.])
        expected=reference.meridional_velocity(z,r)
        for unit in ('cm','m'):
            raw=vertical_arc_duct(unit)
            for p in raw['primitives']:
                p['start'],p['end']=p['end'],p['start']
                if p['kind']=='arc':p['clockwise']=not p['clockwise']
            field=build_carrier(raw,20.,max_step_mm=1.,radial_cells=12)
            np.testing.assert_allclose(field.meridional_velocity(z,r),expected,rtol=2e-11,atol=2e-10)

    def test_cartesian_rotation_and_broadcast_follow_axisymmetric_velocity(self):
        field=build_carrier(duct((100.,115.),(160.,145.)),20.)
        angles=np.array([0.,.3,1.8,3.1]);radius=130.
        v=field.velocity(81.,radius*np.cos(angles),radius*np.sin(angles))
        meridional=field.meridional_velocity(81.,radius)
        np.testing.assert_allclose(v[:,0],meridional[0],atol=1e-13)
        np.testing.assert_allclose(v[:,1],meridional[1]*np.cos(angles),atol=1e-13)
        np.testing.assert_allclose(v[:,2],meridional[1]*np.sin(angles),atol=1e-13)

    def test_scaling_geometry_and_inlet_velocity_obey_similarity(self):
        raw=duct((100.,115.),(160.,145.));base=build_carrier(raw,20.,max_step_mm=10.,radial_cells=12)
        scaled=deepcopy(raw);scale=10.
        scaled['outlet_z_mm']*=scale;scaled['sections'][0]['z_mm']*=scale
        for p in scaled['primitives']:
            for key in ('start','end'):p[key]=(np.array(p[key])*scale).tolist()
        field=build_carrier(scaled,70.,max_step_mm=100.,radial_cells=12,wall_tolerance_mm=.2)
        z=np.array([24.3,63.2,126.7]);r=np.array([125.,135.,140.])
        np.testing.assert_allclose(field.meridional_velocity(z*scale,r*scale),base.meridional_velocity(z,r)*3.5,rtol=3e-12,atol=3e-10)
        self.assertAlmostEqual(field.inlet_flux_m3_s/base.inlet_flux_m3_s,scale**2*3.5,places=10)

    def test_mesh_refinement_reduces_streamfunction_error(self):
        raw=duct((100.,115.),(160.,145.))
        z=np.array([27.3,42.7,71.9,102.3,138.7])
        fraction=np.array([.23,.41,.67,.37,.74]);r=100.+15*z/160+fraction*(60.-30*z/160)
        reference=build_carrier(raw,20.,max_step_mm=1.25,radial_cells=48).streamfunction_at(z,r)
        errors=[]
        for step,n in ((40.,4),(20.,8),(10.,16)):
            values=build_carrier(raw,20.,max_step_mm=step,radial_cells=n).streamfunction_at(z,r)
            errors.append(float(np.linalg.norm(values-reference)))
        self.assertLess(errors[1],errors[0]*.4)
        self.assertLess(errors[2],errors[1]*.5)

    def test_cache_uses_geometry_not_ids_output_sections_or_metadata(self):
        raw=duct((100.,115.),(160.,145.));before=deepcopy(raw)
        field=build_carrier(raw,20.)
        changed=deepcopy(raw);changed['future']={'keep':[1,2]}
        changed['sections'].insert(0,dict(id='S2',z_mm=48.,is_outlet=False))
        for p in changed['primitives']:p['id']='other-'+p['id'];p['fixed']=True
        self.assertIs(build_carrier(changed,20.),field)
        self.assertEqual(raw,before)
        for value in (field.z_mm,field.r_mm,field.eta,field.streamfunction,field.triangles):
            self.assertFalse(value.flags.writeable)
            with self.assertRaises(ValueError):value.flat[0]=999.
        with self.assertRaises(TypeError):field.diagnostics['changed']=True
        clear_carrier_cache();self.assertIsNot(build_carrier(raw,20.),field)

    def test_invalid_parameters_and_excessive_mesh_have_clear_errors(self):
        for velocity in (0.,-1.,float('nan'),float('inf'),True,np.bool_(True)):
            with self.subTest(velocity=velocity),self.assertRaises(ValueError):build_carrier(duct(),velocity)
        for step in (0.,-1.,float('nan'),True,1e-300):
            with self.subTest(step=step),self.assertRaises(ValueError):build_carrier(duct(),20.,max_step_mm=step)
        for count in (True,1,129,3.5):
            with self.subTest(count=count),self.assertRaises(ValueError):build_carrier(duct(),20.,radial_cells=count)
        for tolerance in (0.,float('inf'),False):
            with self.subTest(tolerance=tolerance),self.assertRaises(ValueError):build_carrier(duct(),20.,wall_tolerance_mm=tolerance)
        disabled=duct();disabled['enabled']=False
        with self.assertRaises(ValueError):build_carrier(disabled,20.)


if __name__=='__main__':unittest.main()

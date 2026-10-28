"""One mode, atomic worker results, circular maps and real-time alpha."""
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QFileDialog
from fuelfield.core import project_to_json, calculate_plane
from fuelfield.gui import CalcWorker, MainWindow
from fuelfield.quasi import calculate_quasi, CalculationOutput, CalculationCancelled
from fuelfield.unified_results import build_air_results, air_cell_values
from test_quasi import case as fuel_case


def case(quasi=False):
    cfg, nozzles, _ = fuel_case(count=8)
    cfg.quasistationary = quasi
    cfg.preview_resolution = 200
    cfg.air_model = dict(enabled=True, total_kg_s=1., l0=15., distribution='uniform',
        max_path_mm=80., step_mm=2., sources=[dict(id='SYNTHETIC_AIR', kind='hole',
            method='fixed', flow_kg_s=1., z_mm=0., radius_mm=float(np.mean(cfg.bounds_at(0.))),
            direction=[1.,0.,0.], mixed_fraction=.5)])
    return cfg, nozzles


class TestJointCalculation(unittest.TestCase):
    def test_stationary_worker_emits_both_fields_with_one_mode(self):
        cfg, ns = case()
        before = project_to_json(cfg, ns)
        outputs, errors = [], []
        worker = CalcWorker(cfg, ns)
        worker.completed.connect(outputs.append); worker.failed.connect(errors.append)
        worker.run()
        self.assertEqual(errors, []); self.assertEqual(len(outputs), 1)
        self.assertIsNone(outputs[0].quasi_result)
        self.assertIsNotNone(outputs[0].air_results)
        for i, (fuel, _) in enumerate(outputs[0]):
            plane, mixed = outputs[0].air_results.state(i)
            self.assertIs(plane, fuel)
            self.assertAlmostEqual(mixed.air_total_kg_s.sum(), 1.)
            np.testing.assert_array_equal(fuel.cell_mass_kg_s, calculate_plane(ns,cfg,fuel.z_mm).cell_mass_kg_s)
        self.assertEqual(project_to_json(cfg, ns), before)

    def test_air_error_never_emits_partial_completed_fuel(self):
        cfg, ns = case(); cfg.air_model['sources'][0]['flow_kg_s'] = 2.
        completed, errors = [], []
        worker = CalcWorker(cfg, ns)
        worker.completed.connect(completed.append); worker.failed.connect(errors.append)
        worker.run()
        self.assertEqual(completed, []); self.assertEqual(len(errors), 1)
        self.assertIn('превышают', errors[0])

    def test_quasi_snapshot_alpha_uses_real_pressure_time_and_composite_columns(self):
        cfg, ns = case(True)
        fuel = calculate_quasi(ns, cfg, preview=False)
        before = [p.cell_mass_kg_s.copy() for p in fuel.composites]
        air = build_air_results(cfg, ns, fuel.composites, fuel)
        _, first = air.state(1, 0); _, last = air.state(1, 7)
        self.assertNotEqual(first.alpha_section_total, last.alpha_section_total)
        composite, mixed = air.state(1)
        self.assertTrue(composite.is_composite); self.assertTrue(np.isnan(mixed.alpha_section_total))
        for column, step in enumerate(composite.scan_steps):
            snap, result = air.state(1, int(step))
            np.testing.assert_allclose(mixed.alpha_total[:,column], result.alpha_total[:,column], equal_nan=True)
        for saved, plane in zip(before, fuel.composites):
            np.testing.assert_array_equal(saved, plane.cell_mass_kg_s)
        np.testing.assert_allclose(air_cell_values(composite,mixed,'air_flux')*composite.cell_area_m2, mixed.air_total_kg_s)

    def test_cancelled_air_build_and_disabled_model_do_not_trace(self):
        cfg, ns = case(); plane = calculate_plane(ns,cfg,30.)
        with patch('fuelfield.unified_results.compute_air_field') as compute:
            with self.assertRaises(CalculationCancelled):build_air_results(cfg,ns,[plane],cancel=lambda:True)
            compute.assert_not_called()
            cfg.air_model['enabled'] = False
            self.assertIsNone(build_air_results(cfg,ns,[plane]));compute.assert_not_called()

    def test_unsorted_source_cache_keeps_spatial_fields_and_refreshed_metadata(self):
        from fuelfield.air_distribution import clear_air_cache, compute_air_field
        from fuelfield.air_jets import trace_jets
        cfg, ns = case()
        cfg.air_model['distribution'] = 'directed'
        radius = float(np.mean(cfg.bounds_at(0.)))
        cfg.air_model['sources'] = [dict(id=identity, kind='hole', method='fixed',
            flow_kg_s=flow, z_mm=0., radius_mm=radius+offset, theta_deg=theta,
            direction=[1.,0.,0.], mixed_fraction=mixed,
            sigma_r_mm=8., sigma_theta_deg=6.) for identity,flow,offset,theta,mixed in
            [('S1',.2,-10.,0.,.1),('S2',.3,10.,90.,.5),('S10',.5,0.,210.,.9)]]
        plane = calculate_plane(ns,cfg,30.)
        clear_air_cache()
        try:
            with patch('fuelfield.air_distribution.trace_jets', wraps=trace_jets) as trace:
                cold = compute_air_field(plane,cfg,ns)
                warm = compute_air_field(plane,cfg,ns)
                self.assertEqual(trace.call_count,1)
                for field in ('air_total_kg_s','air_core_kg_s','air_mixed_kg_s'):
                    np.testing.assert_array_equal(getattr(cold,field),getattr(warm,field))
                # Registry reorder and mixing laws must reuse the trajectory,
                # but each source must retain its own new mixing metadata.
                cfg.air_model['sources'].reverse()
                next(s for s in cfg.air_model['sources'] if s['id']=='S2')['mixed_fraction']=.8
                changed = compute_air_field(plane,cfg,ns)
                self.assertEqual(trace.call_count,1)
            clear_air_cache()
            fresh = compute_air_field(plane,cfg,ns)
            for field in ('air_total_kg_s','air_core_kg_s','air_mixed_kg_s'):
                np.testing.assert_array_equal(getattr(changed,field),getattr(fresh,field))
            self.assertFalse(np.array_equal(cold.air_mixed_kg_s,changed.air_mixed_kg_s))
            self.assertAlmostEqual(changed.air_total_kg_s.sum(),1.)
        finally:
            clear_air_cache()


class TestUnifiedResultsUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.w = MainWindow(); self.w.show()
        cfg, ns = case(True)
        self.w.apply_project_state(project_to_json(cfg,ns),'SYNTHETIC test')
        out = calculate_quasi(ns,cfg,preview=True)
        bundle = CalculationOutput([(p,out.smooth(i,0)) for i,p in enumerate(out.composites)],out)
        bundle.air_results = build_air_results(cfg,ns,out.composites,out)
        self.w._done(bundle);self.w.tabs.setCurrentIndex(3)
        self.w.plane_select.setCurrentIndex(1);self.app.processEvents()

    def tearDown(self):
        self.w.close();self.w.deleteLater();self.app.processEvents()

    def test_common_metric_has_circular_alpha_no_extra_window_and_true_geometry(self):
        self.assertFalse(hasattr(self.w,'air_results_button'))
        self.w.metric.setCurrentIndex(self.w.metric.findData('alpha_total'))
        self.app.processEvents()
        ax = self.w.figmap.axes[0]
        self.assertEqual(ax.name, 'polar')
        rings = [line for line in ax.lines if line.get_gid()=='nozzle-row-ring']
        self.assertEqual({float(line.get_ydata()[0]) for line in rings},{n.radius_mm for n in self.w.nozzles})
        marks = next(c for c in ax.collections if c.get_gid()=='comb-pass-positions')
        np.testing.assert_allclose(np.rad2deg(marks.get_offsets()[:,0]),self.w.quasi_result.settings.global_angles(self.w.cfg))
        self.assertIsNone(getattr(self.w,'air_results_dialog',None))
        self.w.plots.setCurrentIndex(3);self.app.processEvents()
        self.assertTrue(all(ax.get_ylabel()=='1' for ax in self.w.figcurves.axes))

    def test_existing_time_slider_changes_alpha_pixels_and_keeps_composite(self):
        # All nozzle pressures in this fixture scale together. An automatic
        # color range scales with alpha and can legitimately show identical
        # colors, so pin the range to test the actual changing pixels.
        self.w.scale_mode.setCurrentIndex(self.w.scale_mode.findData('manual'))
        self.w.scale_min.setValue(0.); self.w.scale_max.setValue(500.)
        self.w.metric.setCurrentIndex(self.w.metric.findData('alpha_total'))
        self.w.plots.setCurrentIndex(1);self.app.processEvents()
        canvas = self.w.smoothcanvas;mesh=canvas._air_mesh
        canvas.draw()
        before_pixels=np.asarray(canvas.buffer_rgba()).copy()
        values = np.ma.array(mesh.get_array(),copy=True)
        fuel = [p.cell_mass_kg_s.copy() for p in self.w.results]
        self.w.time_slider.setValue(7);self.w._update_time_field();self.app.processEvents()
        self.assertIs(mesh,canvas._air_mesh)
        self.assertFalse(np.ma.allequal(values,mesh.get_array()))
        canvas.draw()
        after_pixels=np.asarray(canvas.buffer_rgba())
        x0,y0,x1,y1=map(int,canvas._air_axes.bbox.extents)
        height=before_pixels.shape[0]
        self.assertGreater(np.count_nonzero(before_pixels[height-y1:height-y0,x0:x1]!=after_pixels[height-y1:height-y0,x0:x1]),25)
        self.assertIn('t=',canvas._air_axes.get_title())
        for saved,p in zip(fuel,self.w.results):np.testing.assert_array_equal(saved,p.cell_mass_kg_s)

    def test_air_only_edit_preserves_fuel_but_requires_joint_f5_and_exports_actual_time(self):
        self.w.metric.setCurrentIndex(self.w.metric.findData('alpha_total'))
        self.w.plots.setCurrentIndex(1);self.w.time_slider.setValue(7)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'air.csv'
            with patch.object(QFileDialog,'getSaveFileName',return_value=(str(path),'')):
                self.w.export_results()
            import csv,json
            with path.open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream,delimiter=';'))
            self.assertEqual(float(rows[0]['time_s']),self.w.quasi_result.settings.times[7])
            self.assertEqual(json.loads(rows[0]['pressures_kgf_cm2_json']),{k:float(v[7]) for k,v in self.w.quasi_result.pressures.items()})
        fuel=self.w.results
        raw=deepcopy(self.w.cfg.air_model);raw['sources'][0]['flow_kg_s']=.9
        self.w.apply_air_model(raw)
        self.assertIs(self.w.results,fuel);self.assertIsNone(self.w.air_results)
        self.assertIn('F5',self.w.balance_label.text())

    def test_air_template_edit_keeps_completed_fuel_and_invalidates_only_air(self):
        fuel=self.w.results
        presets=deepcopy(self.w.cfg.nozzle_presets)
        presets['uncooled']['air']=dict(method='fixed',flow_kg_s=.002)
        self.w._presets_changed(presets)
        self.assertIs(self.w.results,fuel)
        self.assertIsNone(self.w.air_results)
        self.assertEqual(self.w.cfg.nozzle_presets['uncooled']['air']['flow_kg_s'],.002)

    def test_air_mode_freeze_survives_other_tabs_and_direct_geometry_edits(self):
        state=project_to_json(self.w.cfg,self.w.nozzles)
        state['settings']['r_out_mm']+=1.
        self.w.apply_project_state(state,'SYNTHETIC geometry edit')
        self.w.geometry_editor.mode_selector.setCurrentIndex(1)
        self.assertTrue(self.w.geometry_editor.is_air_mode)
        frozen=project_to_json(self.w.cfg,self.w.nozzles)
        index=self.w.undo_stack.index()
        for tab in (0,1,2):
            self.w.tabs.setCurrentIndex(tab);self.app.processEvents()
            self.assertFalse(self.w.undo_action.isEnabled())
            self.w._undo_action();self.w._redo_action()
            self.assertEqual(self.w.undo_stack.index(),index)
            self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),frozen)
        candidate=deepcopy(frozen);candidate['nozzles'][0]['radius_mm']+=1.
        self.w.apply_project_state(candidate,'SYNTHETIC forbidden coordinate')
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),frozen)
        candidate=deepcopy(frozen);candidate['settings']['r_out_mm']+=1.
        self.w.apply_geometry_state(candidate,'SYNTHETIC forbidden geometry')
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),frozen)


if __name__ == '__main__':
    unittest.main()

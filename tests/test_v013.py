"""v0.1.3 display/serialization and real Qt editing regressions."""
import json
import os
import sys
import tempfile
import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/injector-mpl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from fuelfield.core import (Settings, default_nozzles, project_to_json, project_from_json,
                            calculate_plane, evaluate_continuous_field, source_flow_kg_s)
from fuelfield.visualization import color_scale, draw_detailed_map
from fuelfield.data_io import write_result_csv
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class TestDisplay(unittest.TestCase):
    def test_old_json_defaults_and_new_preferences_roundtrip(self):
        cfg=Settings();ns=default_nozzles(cfg)
        old=project_to_json(cfg,ns)
        old['settings']={k:v for k,v in old['settings'].items() if not k.startswith('color_')}
        restored,ns2=project_from_json(old)
        self.assertEqual(restored.color_scale_mode,'auto')
        self.assertEqual(project_to_json(restored,ns2)['nozzles'],old['nozzles'])
        restored.color_scale_mode='manual';restored.color_vmin=.2;restored.color_vmax=.8
        restored.color_out_of_range='clip';restored.color_shared_z=True
        new=project_to_json(restored,ns2)
        self.assertEqual(project_to_json(*project_from_json(json.loads(json.dumps(new)))),new)

    def test_saturate_and_clip_preserve_values_and_existing_masks(self):
        values=np.ma.array([-1.,0.,1.,2.,3.],mask=[False,False,True,False,False])
        cfg=Settings(color_scale_mode='manual',color_vmin=0,color_vmax=2)
        data,norm,cmap=color_scale(values,cfg)
        self.assertEqual(cmap(norm(-1)),cmap(norm(0)))
        self.assertEqual(cmap(norm(3)),cmap(norm(2)))
        self.assertEqual(data.mask.tolist(),values.mask.tolist())
        cfg.color_out_of_range='clip'
        data,norm,cmap=color_scale(values,cfg)
        self.assertEqual(data.mask.tolist(),[True,False,True,False,True])
        self.assertEqual(cmap(norm(data))[0,3],0)
        np.testing.assert_array_equal(values.data,[-1.,0.,1.,2.,3.])
        self.assertEqual(values.mask.tolist(),[False,False,True,False,False])

    def test_shared_auto_scale_ignores_masked_wall_values(self):
        cfg=Settings(color_shared_z=True)
        arrays=[np.ma.array([1.,2.,100.],mask=[False,False,True]),np.array([3.,4.])]
        _,norm,_=color_scale(arrays[0],cfg,arrays)
        self.assertEqual((norm.vmin,norm.vmax),(1,4))
        cfg.color_shared_z=False
        _,norm,_=color_scale(arrays[0],cfg,arrays)
        self.assertEqual((norm.vmin,norm.vmax),(1,2))

    def test_scale_invalid_bounds_rejected(self):
        for low,high in ((1,1),(2,1),(float('nan'),1),(0,float('inf'))):
            with self.assertRaises(ValueError):Settings(color_vmin=low,color_vmax=high).validate()

    def test_all_scales_leave_physics_and_csv_identical(self):
        cfg=Settings(n_sample_radial=16,n_sample_angular=72)
        ns=default_nozzles(cfg);ns[0].flow_kg_h=10
        baseline=[calculate_plane(ns,cfg,z) for z in cfg.z_mm]
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a.csv';b=Path(d)/'b.csv'
            write_result_csv(a,baseline)
            for mode in ('saturate','clip'):
                cfg.color_scale_mode='manual';cfg.color_out_of_range=mode
                for z,original in zip(cfg.z_mm,baseline):
                    p=calculate_plane(ns,cfg,z)
                    np.testing.assert_array_equal(p.flux_kg_m2_s,original.flux_kg_m2_s)
                    self.assertLess(abs(p.rel_balance_error),1e-12)
                write_result_csv(b,[calculate_plane(ns,cfg,z) for z in cfg.z_mm])
                self.assertEqual(a.read_bytes(),b.read_bytes())

    def test_clipped_map_retains_walls_and_nozzle_labels(self):
        cfg=Settings(n_sample_radial=8,n_sample_angular=72,preview_resolution=200,
                     color_scale_mode='manual',color_vmin=100,color_vmax=101,
                     color_out_of_range='clip')
        ns=default_nozzles(cfg);ns[0].flow_kg_h=10
        plane=calculate_plane(ns,cfg,0);field=evaluate_continuous_field(ns,cfg,0)
        fig=Figure();FigureCanvasAgg(fig)
        ax=draw_detailed_map(fig,field,plane,cfg,ns[:1])
        fig.canvas.draw()
        self.assertEqual(len(ax.patches),2)
        self.assertIn('1',[text.get_text() for text in ax.texts])
        self.assertTrue(np.ma.getmaskarray(ax.images[0].get_array()).all())


try:
    from PyQt5.QtWidgets import QApplication, QTableWidgetSelectionRange, QMessageBox
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    HAS_QT=True
except ImportError:
    HAS_QT=False
if HAS_QT:
    from fuelfield.gui import MainWindow, INTCOL


@unittest.skipUnless(HAS_QT,'PyQt5 not installed; GUI checks require development dependencies')
class TestQtEditing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.w=MainWindow()
        self.w.preview_timer.stop()

    def tearDown(self):
        self.w.preview_timer.stop();self.w.close();self.w.deleteLater()
        self.app.processEvents()

    def select(self,table,r0,c0,r1=None,c1=None):
        table.clearSelection();table.setCurrentCell(r0,c0)
        table.setRangeSelected(QTableWidgetSelectionRange(r0,c0,r0 if r1 is None else r1,
                                                       c0 if c1 is None else c1),True)

    def results(self):
        self.w._restoring=True
        self.w.cfg=Settings(n_sample_radial=8,n_sample_angular=72,preview_resolution=200)
        for n in self.w.nozzles:n.flow_kg_h=10
        self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history()
        cfg=self.w.cfg;ns=self.w.nozzles
        self.w._done([(calculate_plane(ns,cfg,z),evaluate_continuous_field(ns,cfg,z)) for z in cfg.z_mm])

    def test_single_value_fills_selection_one_undo_and_redo(self):
        tab=self.w.nozzle_table;c=INTCOL['flow_kg_h']
        self.select(tab,0,c,2,c)
        tab.paste_text('12,5')
        self.assertEqual([n.flow_kg_h for n in self.w.nozzles[:3]],[12.5]*3)
        self.assertEqual(self.w.undo_stack.count(),1)
        self.w.undo_stack.undo()
        self.assertEqual([n.flow_kg_h for n in self.w.nozzles[:3]],[0]*3)
        self.w.undo_stack.redo()
        self.assertEqual([n.flow_kg_h for n in self.w.nozzles[:3]],[12.5]*3)

    def test_tsv_paste_is_atomic_and_skips_readonly_ids(self):
        tab=self.w.sector_table
        self.select(tab,0,0)
        tab.paste_text('99\t2\r\n99\t3\r\n')
        self.assertEqual(tab.item(0,0).text(),'1')
        self.assertEqual(self.w.nozzles[0].sector_ml[:2],[2,3])
        self.assertEqual(self.w.nozzles[0].flow_kg_h,0)
        self.assertEqual(self.w.undo_stack.count(),1)
        self.w.undo_stack.undo()
        self.assertEqual(self.w.nozzles[0].sector_ml[:2],[1,1])
        self.assertEqual(self.w.nozzles[0].sector_source,'шаблон: равномерно')
        self.w.undo_stack.redo()
        self.assertEqual(self.w.nozzles[0].sector_ml[:2],[2,3])

    def test_invalid_nozzle_paste_rolls_back_whole_operation(self):
        tab=self.w.nozzle_table;c=INTCOL['flow_kg_h']
        self.select(tab,0,c)
        tab.paste_text('10\nbad')
        self.assertEqual([n.flow_kg_h for n in self.w.nozzles[:2]],[0,0])
        self.assertEqual(tab.item(0,c).text(),'0')
        self.assertEqual(self.w.undo_stack.count(),0)

    def test_invalid_sector_does_not_change_values(self):
        self.w.sector_table.item(0,1).setText('nan')
        self.assertEqual(self.w.nozzles[0].sector_ml[0],1)
        self.assertEqual(self.w.undo_stack.count(),0)

    def test_invalid_sector_paste_rolls_back_valid_cells_too(self):
        self.select(self.w.sector_table,0,1)
        self.w.sector_table.paste_text('2\n-1')
        self.assertEqual(self.w.nozzles[0].sector_ml[:2],[1,1])
        self.assertEqual(self.w.nozzles[0].sector_source,'шаблон: равномерно')
        self.assertEqual(self.w.undo_stack.count(),0)

    def test_checkbox_undo_preserves_coordinates(self):
        positions=[(n.theta_deg,n.radius_mm) for n in self.w.nozzles]
        self.w.nozzle_table.item(0,INTCOL['enabled']).setCheckState(Qt.Unchecked)
        self.assertFalse(self.w.nozzles[0].enabled)
        self.w.undo_stack.undo()
        self.assertTrue(self.w.nozzles[0].enabled)
        self.assertEqual(positions,[(n.theta_deg,n.radius_mm) for n in self.w.nozzles])

    def test_group_demo_flow_is_one_command(self):
        with patch.object(QMessageBox,'question',return_value=QMessageBox.Yes):self.w.assign_demo_flow()
        self.assertEqual(self.w.undo_stack.count(),1)
        self.assertTrue(all(n.flow_kg_h==10 for n in self.w.nozzles))
        self.w.undo_stack.undo()
        self.assertTrue(all(n.flow_kg_h==0 for n in self.w.nozzles))

    def test_group_copy_and_sector_flow_undo_keep_manual_measurement(self):
        from PyQt5.QtWidgets import QInputDialog
        self.w.sector_table.item(0,1).setText('10')
        self.w.nozzle_table.item(1,INTCOL['flow_kg_h']).setText('30')
        self.select(self.w.nozzle_table,1,INTCOL['flow_kg_h'])
        with patch.object(QInputDialog,'getInt',return_value=(1,True)):self.w.copy_sectors()
        self.assertEqual(self.w.nozzles[1].sector_ml[0],10)
        self.assertEqual(self.w.nozzles[1].flow_kg_h,30)
        self.w.sum_sectors_to_flow()
        self.assertEqual(self.w.nozzles[1].flow_mode,'sectors')
        self.w.undo_stack.undo()
        self.assertEqual(self.w.nozzles[1].flow_mode,'manual')
        self.assertEqual(self.w.nozzles[1].flow_kg_h,30)
        self.w.undo_stack.undo()
        self.assertEqual(self.w.nozzles[1].sector_ml[0],1)

    def test_positions_and_import_are_undoable_groups(self):
        from PyQt5.QtWidgets import QFileDialog
        from fuelfield.data_io import write_sector_csv
        self.w.nozzle_table.item(0,INTCOL['theta_deg']).setText('7')
        self.w.update_positions()
        self.assertEqual(self.w.nozzles[0].theta_deg,0)
        self.w.undo_stack.undo()
        self.assertEqual(self.w.nozzles[0].theta_deg,7)
        self.w.sector_table.item(0,1).setText('2')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sectors.csv'
            ns=default_nozzles(self.w.cfg);ns[0].sector_ml=[4]*36
            write_sector_csv(path,ns)
            with patch.object(QFileDialog,'getOpenFileName',return_value=(str(path),'')), \
                 patch.object(QMessageBox,'information'):
                self.w.import_sectors()
        self.assertEqual(self.w.nozzles[0].sector_ml[0],4)
        self.w.undo_stack.undo()
        self.assertEqual(self.w.nozzles[0].sector_ml[0],2)

    def test_untouched_json_numbers_are_not_rounded_by_display_edits(self):
        self.w._restoring=True
        self.w.cfg.r_in_mm=200.123456789
        self.w.cfg.z_mm=[0,20.123456789,40,60]
        self.w.nozzles[0].flow_kg_h=10.123456789123
        self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history()
        before=deepcopy(self.w._edit_state)
        self.w.scale_shared.setChecked(True)
        after=self.w._edit_state
        self.assertEqual(before['nozzles'],after['nozzles'])
        for key,value in before['settings'].items():
            if not key.startswith('color_'):self.assertEqual(value,after['settings'][key],key)

    def test_worker_startup_and_mass_balance(self):
        self.w._restoring=True
        self.w.cfg=Settings(n_sample_radial=8,n_sample_angular=72,preview_resolution=200)
        self.w.nozzles[0].flow_kg_h=10
        self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history()
        errors=[]
        with patch.object(self.w,'_error',side_effect=errors.append):
            self.w.run_calc()
            deadline=time.monotonic()+10
            while self.w.thread is not None and time.monotonic()<deadline:
                self.app.processEvents();QTest.qWait(10)
        self.assertIsNone(self.w.thread)
        self.assertEqual(errors,[])
        self.assertEqual(len(self.w.results),4)
        for p in self.w.results:self.assertLess(abs(p.rel_balance_error),1e-12)
        self.assertTrue(self.w.calc_btn.isEnabled())

    def test_png_export_renders_inactive_views_for_each_plane(self):
        from PyQt5.QtWidgets import QFileDialog
        self.results()
        errors=[]
        with tempfile.TemporaryDirectory() as d:
            with patch.object(QFileDialog,'getExistingDirectory',return_value=d), \
                 patch.object(self.w,'_error',side_effect=errors.append), \
                 patch.object(Figure,'savefig') as save:
                self.w.export_all_png()
            self.assertEqual(errors,[])
            self.assertEqual(save.call_count,16)
            self.assertEqual(len(self.w._render_cache),16)
            for i in range(4):
                fig=self.w._render_view(1,i)
                self.assertIn(f'z={self.w.results[i].z_mm:g}',fig.axes[0].get_title())
            # Also exercise real PNG serialization of a cached inactive view.
            fig.savefig(Path(d)/'preview.png')
            self.assertGreater((Path(d)/'preview.png').stat().st_size,1000)

    def test_preview_evaluates_only_selected_source_at_zero(self):
        self.w._show_sectors(71)
        from fuelfield import gui
        with patch.object(gui,'evaluate_continuous_field',wraps=evaluate_continuous_field) as evaluate:
            self.w._draw_single_preview()
            args=evaluate.call_args.args
            self.assertEqual(len(args[0]),1)
            self.assertEqual(args[0][0].id,71)
            self.assertEqual(args[2],0)
            self.assertEqual(evaluate.call_args.kwargs['resolution'],400)
        self.w.previewcanvas.draw()
        self.assertEqual(len(self.w.figpreview.axes),3)
        image=self.w.figpreview.axes[1].images[0]
        self.assertEqual(image.get_array().shape,(400,400))
        self.assertEqual(image.get_interpolation(),'bilinear')

    def test_higher_resolution_preview_preserves_strict_clip_and_source_data(self):
        self.w.nozzle_table.item(0,INTCOL['flow_kg_h']).setText('10')
        original=deepcopy(self.w._edit_state)
        self.w.scale_mode.setCurrentIndex(1)
        self.w.scale_range.setCurrentIndex(1)
        self.w._draw_single_preview()
        image=self.w.figpreview.axes[1].images[0]
        self.assertEqual(image.get_array().shape,(400,400))
        self.assertEqual(image.get_interpolation(),'nearest')
        self.assertEqual(self.w._edit_state['nozzles'],original['nozzles'])
        self.assertEqual(len(self.w.figpreview.axes[0].lines[0].get_xdata()),289)

    def test_lazy_render_and_cache_reuse(self):
        self.results()
        self.assertEqual(len(self.w._render_cache),1)
        with patch.object(self.w,'_draw_smooth',wraps=self.w._draw_smooth) as smooth:
            self.w.plane_select.setCurrentIndex(1)
            self.assertEqual(smooth.call_count,0)
            self.w.plots.setCurrentIndex(1)
            self.assertEqual(smooth.call_count,1)
            cached=self.w.smoothcanvas
            self.w.plots.setCurrentIndex(0);self.w.plots.setCurrentIndex(1)
            self.assertIs(self.w.smoothcanvas,cached)
            self.assertEqual(smooth.call_count,1)
        self.w.draw_labels.setChecked(False)
        self.assertEqual(len(self.w._render_cache),1)

    def test_scale_undo_keeps_results_physics_edit_invalidates(self):
        self.results();original=self.w.results
        self.w.scale_mode.setCurrentIndex(1)
        self.assertIs(self.w.results,original)
        self.w.scale_range.setCurrentIndex(1)
        self.assertEqual(self.w.cfg.color_out_of_range,'clip')
        self.w.undo_stack.undo()
        self.assertEqual(self.w.cfg.color_out_of_range,'saturate')
        self.assertIs(self.w.results,original)
        self.w.nozzle_table.item(0,INTCOL['flow_kg_h']).setText('20')
        self.assertEqual(self.w.results,[])
        self.assertEqual(self.w._render_cache,{})

    def test_stale_worker_results_rejected(self):
        self.w._calc_revision=self.w._revision
        self.w.nozzle_table.item(0,INTCOL['flow_kg_h']).setText('20')
        self.w._done([])
        self.assertIn('Повторите F5',self.w.statusBar().currentMessage())

    def test_mathtext_and_keyboard_shortcuts(self):
        self.w.show();self.w.tabs.setCurrentIndex(1);self.app.processEvents()
        tab=self.w.nozzle_table
        self.select(tab,0,INTCOL['flow_kg_h'])
        tab.setFocus()
        QApplication.clipboard().setText('5')
        QTest.keyClick(tab,Qt.Key_V,Qt.ControlModifier);self.app.processEvents()
        self.assertEqual(self.w.nozzles[0].flow_kg_h,5)
        QTest.keyClick(tab,Qt.Key_Z,Qt.ControlModifier);self.app.processEvents()
        self.assertEqual(self.w.nozzles[0].flow_kg_h,0)
        QTest.keyClick(tab,Qt.Key_Y,Qt.ControlModifier);self.app.processEvents()
        self.assertEqual(self.w.nozzles[0].flow_kg_h,5)
        self.w.tabs.setCurrentIndex(2);self.app.processEvents()
        # Render the formula canvas, exercising MathText rather than only parsing Python.
        canvases=self.w.tabs.widget(2).findChildren(type(self.w.previewcanvas))
        self.assertTrue(canvases)
        for canvas in canvases:canvas.draw()


if __name__=='__main__':unittest.main()

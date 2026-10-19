"""Acceptance of the shipped demo, physical CSV balances and atomic paste."""
import csv,json,sys,tempfile,unittest
from pathlib import Path
from copy import deepcopy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import calculate_plane,project_from_json,project_to_json
from fuelfield.chamber import Chamber,legacy_geometry
from fuelfield.data_io import write_result_csv
ROOT=Path(__file__).resolve().parents[1]


class TestV06Delivery(unittest.TestCase):
    def project(self):return project_from_json(json.loads((ROOT/'examples/demo_project_v06_geometry.json').read_text()))
    def test_demo_is_synthetic_complete_and_conservative_at_all_manual_sections(self):
        cfg,ns=self.project();model=Chamber(cfg.geometry)
        self.assertEqual(cfg._project_extra['provenance']['source_kind'],'synthetic')
        self.assertEqual({p.kind for p in model.primitives},{'point','line','arc'})
        self.assertEqual(len(ns),140);self.assertEqual(sum(n.enabled for n in ns),139)
        self.assertTrue(any(g['mode']=='steps' for g in model.groups))
        self.assertEqual(cfg.z_mm,[s['z_mm'] for s in model.sections]);self.assertNotEqual(cfg.z_mm,sorted(cfg.z_mm))
        self.assertEqual(model.sections[-1]['z_mm'],model.outlet)
        cfg.n_sample_radial=8;cfg.n_sample_angular=72
        seen_wall=False
        for z in cfg.z_mm:
            p=calculate_plane(ns,cfg,z);self.assertLess(abs(p.rel_balance_error),1e-12)
            self.assertTrue(np.all(p.cell_area_m2>0));np.testing.assert_allclose(np.diff(p.radial_edges_mm),np.diff(p.radial_edges_mm)[0],rtol=0,atol=1e-12)
            self.assertAlmostEqual(p.cell_area_m2.sum(),np.pi*1e-6*(p.radial_edges_mm[-1]**2-p.radial_edges_mm[0]**2),places=13)
            seen_wall|=p.wall_bound_kg_s>0
        self.assertTrue(seen_wall)
    def test_stationary_csv_separates_wall_bound_and_subtotals(self):
        cfg,ns=self.project();cfg.n_sample_radial=8;cfg.n_sample_angular=72;planes=[calculate_plane(ns,cfg,z) for z in (50.,180.)]
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'matrix.csv';write_result_csv(str(target),planes)
            with target.open(encoding='utf-8-sig') as file:cells=list(csv.DictReader(file,delimiter=';'))
            with target.with_name('matrix_balances.csv').open(encoding='utf-8-sig') as file:balances=list(csv.DictReader(file))
            self.assertEqual(len(balances),2);self.assertEqual(len(cells),2*cfg.n_radial*cfg.n_angular)
            for row,p in zip(balances,planes):
                total=sum(float(row[key]) for key in ('inside_kg_s','inner_kg_s','outer_kg_s','wall_bound_kg_s'))
                self.assertAlmostEqual(total,float(row['injected_kg_s']),delta=1e-13)
                self.assertAlmostEqual(float(row['wall_bound_kg_s']),float(row['wall_inner_kg_s'])+float(row['wall_outer_kg_s']),delta=1e-13)
                self.assertIn('не расчёт',row['geometry_notice']);self.assertEqual(float(row['z_mm']),p.z_mm)
    def test_two_row_paste_is_one_validated_undo(self):
        from PyQt5.QtWidgets import QApplication
        from fuelfield.gui import MainWindow
        app=QApplication.instance() or QApplication([]);w=MainWindow();w.show()
        try:
            cfg,ns=self.project();raw=legacy_geometry(cfg,ns);raw['enabled']=True
            raw['primitives'] += [dict(id=f'point-{i}',wall='reference',kind='point',start=[10.+i,250.],unit='mm') for i in (0,1)]
            cfg.geometry=raw;w.apply_project_state(project_to_json(cfg,ns),'Fixture');w.geometry_pages.setCurrentIndex(1);app.processEvents();w.undo_stack.clear()
            editor=w.geometry_editor;table=editor.primitive_table;row=next(i for i,p in enumerate(editor.raw['primitives']) if p['id']=='point-0');before=project_to_json(w.cfg,w.nozzles)
            table.setCurrentCell(row,3);table.paste_text('12\t251.125\n22\t252.5');app.processEvents()
            points={p['id']:p for p in w.cfg.geometry['primitives']}
            self.assertEqual(points['point-0']['start'],[12.,251.125]);self.assertEqual(points['point-1']['start'],[22.,252.5]);self.assertEqual(w.undo_stack.count(),1)
            w.undo_stack.undo();self.assertEqual(project_to_json(w.cfg,w.nozzles),before)
        finally:w.close();w.deleteLater();app.processEvents()

    def test_wrapped_balances_do_not_resize_cached_section_canvases(self):
        from PyQt5.QtWidgets import QApplication
        from fuelfield.gui import MainWindow
        app=QApplication.instance() or QApplication([]);w=MainWindow();w.resize(1480,930);w.show()
        try:
            cfg,ns=self.project();cfg.n_sample_radial=8;cfg.n_sample_angular=72
            w.apply_project_state(project_to_json(cfg,ns),'Variable walls');w.tabs.setCurrentIndex(3);app.processEvents()
            planes=[calculate_plane(ns,cfg,z) for z in cfg.z_mm]
            w._done([(p,None) for p in planes]);app.processEvents()
            sizes=[]
            for _ in range(2):
                for i in range(len(planes)):
                    w.plane_select.setCurrentIndex(i);app.processEvents();sizes.append((w.mapcanvas.width(),w.mapcanvas.height()))
            self.assertEqual(len(set(sizes)),1,'Balance text wrapping resized a cached plot')
            cache=list(w._render_cache.values());draws=[]
            from unittest.mock import patch
            from contextlib import ExitStack
            with ExitStack() as stack:
                for canvas in cache:stack.enter_context(patch.object(canvas,'draw',side_effect=lambda:draws.append(1)))
                for i in range(len(planes)):
                    w.plane_select.setCurrentIndex(i);app.processEvents()
            self.assertEqual(draws,[],'Revisiting a cached section repainted Matplotlib')
            self.assertEqual(len(w._render_cache),len(planes))
        finally:w.close();w.deleteLater();app.processEvents()

if __name__=='__main__':unittest.main()

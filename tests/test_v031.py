"""Gesture, annotation, redraw and UI geometry regressions for v0.3.1."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backend_bases import MouseEvent,MouseButton
from PyQt5.QtWidgets import QApplication,QLabel
from PyQt5.QtCore import Qt,QPoint
from PyQt5.QtTest import QTest
from fuelfield.plot_interaction import attach_interaction
from fuelfield.gui import MainWindow
from fuelfield.analysis import EQUATIONS
from fuelfield.core import project_from_json,calculate_plane,evaluate_continuous_field

ROOT=Path(__file__).resolve().parents[1]
class TestPlotGestures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.fig=Figure(figsize=(6,4));self.canvas=FigureCanvasQTAgg(self.fig);self.canvas.resize(600,400)
        self.ax=self.fig.add_subplot(111);self.x=np.arange(11,dtype=float);self.y=2*self.x
        self.ax.plot(self.x,self.y);self.canvas.draw();self.interaction=attach_interaction(self.canvas);self.canvas.draw()
    def tearDown(self):self.canvas.close();self.canvas.deleteLater();self.app.processEvents()
    def event(self,name,x,y,button=None,double=False,ax=None):
        pixel=(ax or self.ax).transData.transform((x,y))
        return MouseEvent(name,self.canvas,*pixel,button=button,dblclick=double)
    def pin(self,x=4,y=8,ax=None):self.interaction.press(self.event('button_press_event',x,y,MouseButton.LEFT,True,ax));self.interaction.release(self.event('button_release_event',x,y,MouseButton.LEFT,ax=ax));self.canvas.draw()
    def test_hover_blits_and_preserves_source_and_limits(self):
        limits=(self.ax.get_xlim(),self.ax.get_ylim());before=self.ax.lines[0].get_ydata().copy()
        with patch.object(self.canvas,'draw',wraps=self.canvas.draw) as draw:
            for x in np.linspace(1,9,100):self.interaction.motion(self.event('motion_notify_event',x,2*x))
            self.assertEqual(draw.call_count,0)
        v,h,text=self.interaction.hover[self.ax];self.assertTrue(v.get_visible());self.assertIn('x=',text.get_text());self.assertIn('y=',text.get_text())
        np.testing.assert_array_equal(before,self.ax.lines[0].get_ydata());self.assertEqual(limits,(self.ax.get_xlim(),self.ax.get_ylim()))
        self.interaction.motion(MouseEvent('motion_notify_event',self.canvas,-10,-10));self.assertFalse(v.get_visible())
    def test_point_on_line_interpolation_and_limit_ten(self):
        for i in range(12):self.pin(2+i*.4,4+i*.8+.5)
        self.assertEqual(len(self.interaction.points),10)
        for p in self.interaction.points:
            self.assertAlmostEqual(p.y,2*p.x);self.assertEqual(len(p.artists),4)
        np.testing.assert_array_equal(self.ax.lines[0].get_ydata(),self.y)
    def test_mouse_chord_clears_all_points(self):
        self.pin();self.pin(6,12)
        self.interaction.press(self.event('button_press_event',5,10,MouseButton.RIGHT))
        self.interaction.press(self.event('button_press_event',5,10,MouseButton.LEFT))
        self.assertEqual(self.interaction.points,[])
        self.assertEqual(len(self.ax.lines),3) # original and animated hover guides
    def test_pinning_and_clearing_do_not_redraw_chart(self):
        with patch.object(self.canvas,'draw',wraps=self.canvas.draw) as draw:
            event=self.event('button_press_event',4,8,MouseButton.LEFT,True)
            self.interaction.press(event);self.assertEqual(len(self.interaction.points),1)
            self.interaction.clear_points();self.assertEqual(self.interaction.points,[])
            self.assertEqual(draw.call_count,0)

    def test_middle_drag_zoom_and_restore(self):
        original=(self.ax.get_xlim(),self.ax.get_ylim())
        self.interaction.press(self.event('button_press_event',2,3,MouseButton.MIDDLE))
        self.interaction.motion(self.event('motion_notify_event',8,16,MouseButton.MIDDLE))
        self.assertIsNotNone(self.interaction.rectangle)
        self.interaction.release(self.event('button_release_event',8,16,MouseButton.MIDDLE));self.canvas.draw()
        np.testing.assert_allclose(self.ax.get_xlim(),[2,8],atol=.04);np.testing.assert_allclose(self.ax.get_ylim(),[3,16],atol=.08)
        self.interaction.reset_zoom();self.assertEqual(original,(self.ax.get_xlim(),self.ax.get_ylim()));self.assertIsNone(self.interaction.rectangle)
    def test_middle_click_without_drag_does_not_zoom(self):
        original=(self.ax.get_xlim(),self.ax.get_ylim());e=self.event('button_press_event',4,8,MouseButton.MIDDLE)
        self.interaction.press(e);self.interaction.release(self.event('button_release_event',4,8,MouseButton.MIDDLE))
        self.assertEqual(original,(self.ax.get_xlim(),self.ax.get_ylim()))
    def test_twin_y_hover_and_zoom_own_units(self):
        self.interaction.clear_artists();other=self.ax.twinx();other.plot([0,10],[300,400],color='red');other.set_ylabel('Температура, C')
        self.interaction.reset_axes();self.canvas.draw();original_other=other.get_ylim()
        self.interaction.motion(self.event('motion_notify_event',4,340,ax=other))
        self.assertIn('y=',self.interaction.hover[other][2].get_text());self.assertIn('Y=',self.interaction.hover[other][2].get_text())
        self.interaction.press(self.event('button_press_event',2,320,MouseButton.MIDDLE,ax=other))
        self.interaction.release(self.event('button_release_event',8,380,MouseButton.MIDDLE,ax=other));self.canvas.draw()
        np.testing.assert_allclose(other.get_ylim(),[320,380],atol=.4)
        np.testing.assert_allclose(self.ax.get_xlim(),[2,8],atol=.04)
        self.interaction.reset_zoom();self.assertEqual(other.get_ylim(),original_other)
    def test_scatter_and_bar_points(self):
        for kind in ('scatter','bar'):
            self.interaction.clear_artists();self.ax.clear();getattr(self.ax,kind)([1,2,3],[5,7,9]);self.interaction.reset_axes();self.canvas.draw()
            self.pin(2,7);p=self.interaction.points[0];self.assertEqual((p.x,p.y),(2.,7.))
    def test_nan_gap_does_not_create_interpolated_segment(self):
        self.interaction.clear_artists();self.ax.clear();self.ax.plot([0,1,2,3],[0,np.nan,np.nan,3]);self.interaction.reset_axes();self.canvas.draw()
        self.pin(1.5,1.5);p=self.interaction.points[0];self.assertIn((p.x,p.y),[(0.,0.),(3.,3.)])
    def test_real_qt_mouse_events(self):
        self.canvas.show();self.app.processEvents();self.canvas.draw()
        x,y=self.ax.transData.transform((4,8));pos=QPoint(round(x),round(self.canvas.height()-y))
        QTest.mouseMove(self.canvas,QPoint(1,1));QTest.qWait(20)
        QTest.mouseMove(self.canvas,pos);QTest.qWait(20);self.app.processEvents()
        self.assertTrue(self.interaction.hover[self.ax][0].get_visible())
        QTest.mouseDClick(self.canvas,Qt.LeftButton,pos=pos);QTest.mouseRelease(self.canvas,Qt.LeftButton,pos=pos);self.app.processEvents()
        self.assertEqual(len(self.interaction.points),1)
        QTest.mousePress(self.canvas,Qt.LeftButton,pos=pos);QTest.mousePress(self.canvas,Qt.RightButton,pos=pos)
        QTest.mouseRelease(self.canvas,Qt.RightButton,pos=pos);QTest.mouseRelease(self.canvas,Qt.LeftButton,pos=pos)
        self.assertEqual(self.interaction.points,[])
        end_x,end_y=self.ax.transData.transform((8,16));end=QPoint(round(end_x),round(self.canvas.height()-end_y))
        original=self.ax.get_xlim();QTest.mousePress(self.canvas,Qt.MiddleButton,pos=pos)
        QTest.mouseMove(self.canvas,end);QTest.mouseRelease(self.canvas,Qt.MiddleButton,pos=end);self.app.processEvents()
        self.assertLess(self.ax.get_xlim()[1]-self.ax.get_xlim()[0],original[1]-original[0])
        QTest.keyClick(self.canvas,Qt.Key_R);self.app.processEvents();self.assertEqual(self.ax.get_xlim(),original)
    def test_resize_and_export_leave_valid_hover(self):
        self.pin();self.canvas.resize(800,500);self.canvas.draw()
        with tempfile.TemporaryDirectory() as d:self.fig.savefig(Path(d)/'plot.png',dpi=150)
        self.interaction.motion(self.event('motion_notify_event',4,8));self.app.processEvents()
        self.assertTrue(self.interaction.hover[self.ax][0].get_visible())

class TestV031UX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        cls.cfg,cls.ns=project_from_json(json.loads((ROOT/'examples/demo_project_user_field.json').read_text()))
        cls.plane=calculate_plane(cls.ns,cls.cfg,30);cls.field=evaluate_continuous_field(cls.ns,cls.cfg,30)
    def setUp(self):
        self.w=MainWindow();self.w._restoring=True;self.w.cfg,self.w.nozzles=project_from_json(json.loads((ROOT/'examples/demo_project_user_field.json').read_text()))
        self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history();self.w.preview_timer.stop();self.w.selection_timer.stop()
        self.w.show();self.w._done([(self.plane,self.field)]);self.w.tabs.setCurrentIndex(3);self.app.processEvents()
    def tearDown(self):self.w.close();self.w.deleteLater();self.app.processEvents()
    def test_single_panel_default_split_toggles_no_data_loss(self):
        c=self.w.comparison;self.assertFalse(c.isVisible());self.assertTrue(all(not b.isVisible() for b in c.load_buttons));self.assertFalse(self.w.split_button.isChecked())
        QTest.mouseClick(self.w.split_button,Qt.LeftButton);self.app.processEvents();self.assertTrue(c.isVisible());self.assertTrue(all(b.isVisible() for b in c.load_buttons))
        with patch('fuelfield.comparison.QFileDialog.getOpenFileName',return_value=('','')) as picker:
            for button,file_filter in zip(c.load_buttons,('JSON (*.json)','Excel (*.xlsx)','CSV (*.csv)')):
                QTest.mouseClick(button,Qt.LeftButton);self.assertEqual(picker.call_args.args[-1],file_filter)
        c.load_path(ROOT/'examples/demo_comparison.csv');tables=c.tables
        QTest.mouseClick(self.w.split_button,Qt.LeftButton);self.app.processEvents();self.assertFalse(c.isVisible());self.assertIs(c.tables,tables)
        QTest.mouseClick(self.w.split_button,Qt.LeftButton);self.app.processEvents();self.assertIs(c.tables,tables)
    def test_unchanged_comparison_keeps_annotations_when_left_layout_changes(self):
        self.w.split_button.setChecked(True);c=self.w.comparison;c.load_path(ROOT/'examples/demo_comparison.csv')
        c.canvas.draw();ax=c.figure.axes[0];pixel=ax.transData.transform((90,11))
        c.interaction.press(MouseEvent('button_press_event',c.canvas,*pixel,button=MouseButton.LEFT,dblclick=True))
        self.assertEqual(len(c.interaction.points),1)
        self.w.plots.setCurrentIndex(3);self.w.plot_count.setValue(2)
        self.assertEqual(len(c.interaction.points),1)
        c.draw();self.assertEqual(len(c.interaction.points),1)

    def test_each_metric_equation_and_nonmutation(self):
        p=self.w.results[0];before=p.cell_mass_kg_s.copy()
        for key,equation in EQUATIONS.items():
            self.w.metric.setCurrentIndex(self.w.metric.findData(key));self.assertEqual(self.w.metric_equation.text(),equation)
        np.testing.assert_array_equal(before,p.cell_mass_kg_s)
    def test_primary_graph_count_columns_selected_radius(self):
        self.w.plots.setCurrentIndex(3);self.w.plot_count.setValue(3);self.w.plot_columns.setValue(1);self.w.compare_radius.setCurrentIndex(2)
        self.assertEqual(len(self.w.figcurves.axes),3)
        self.assertIn('R=335.00',self.w.figcurves.axes[0].get_title())
        self.assertEqual(self.w.figcurves.axes[0].get_subplotspec().get_gridspec().ncols,1)
        self.assertTrue(hasattr(self.w.curvecanvas,'interaction'))
    def test_secondary_multiple_graphs_and_columns(self):
        c=self.w.comparison;self.w.split_button.setChecked(True);c.load_path(ROOT/'examples/demo_comparison.csv')
        c.plot_count.setValue(2);c.plot_columns.setValue(2);self.assertEqual(len(c.figure.axes),2)
        self.assertEqual(c.figure.axes[0].get_subplotspec().get_gridspec().ncols,2)
        c._loading_cfg=self.w.cfg;c._project_done([self.plane]);c.plot_count.setValue(4);c.plot_columns.setValue(2)
        self.assertEqual(len(c.figure.axes),4)
    def test_table_controls_are_adjacent_to_canvas_and_labels(self):
        self.w.split_button.setChecked(True);c=self.w.comparison;c.load_path(ROOT/'examples/demo_comparison.csv');self.app.processEvents()
        controls_bottom=c.table_controls.mapTo(c,c.table_controls.rect().bottomLeft()).y()
        canvas_top=c.canvas.mapTo(c,c.canvas.rect().topLeft()).y()
        self.assertLess(canvas_top-controls_bottom,150)
        for combo in (c.x,c.y,c.y2):
            labels=[l for l in c.table_controls.findChildren(QLabel) if l.y()==combo.y() or abs(l.geometry().center().y()-combo.geometry().center().y())<8]
            preceding=[l for l in labels if l.geometry().right()<combo.geometry().left()]
            self.assertTrue(preceding);self.assertLess(combo.geometry().left()-max(l.geometry().right() for l in preceding),20)
        self.assertLessEqual(self.w.width(),1480)
    def test_primary_label_gaps_and_split_position(self):
        for width in (1280,1480,1920):
            self.w.resize(width,930);self.app.processEvents();page=self.w.tabs.widget(3)
            self.assertLessEqual(self.w.width(),width)
            for widget in (self.w.metric,self.w.volume_interval,self.w.compare_radius):
                pos=widget.mapTo(page,QPoint(0,0));labels=page.findChildren(QLabel)
                preceding=[]
                for label in labels:
                    lp=label.mapTo(page,QPoint(0,0))
                    if abs(lp.y()+label.height()/2-pos.y()-widget.height()/2)<8 and lp.x()+label.width()<=pos.x():preceding.append(lp.x()+label.width())
                self.assertTrue(preceding);self.assertLess(pos.x()-max(preceding),20)
                self.assertLessEqual(pos.x()+widget.width(),page.width())
            split=self.w.split_button.mapTo(page,QPoint(0,0));self.assertGreater(split.x(),page.width()-80)

    def test_interactions_both_panels_do_not_calculate(self):
        self.w.plots.setCurrentIndex(3);self.w.split_button.setChecked(True);c=self.w.comparison;c.load_path(ROOT/'examples/demo_comparison.csv')
        before=self.w.results[0].cell_mass_kg_s.copy()
        with patch('fuelfield.gui.calculate_plane',side_effect=AssertionError('unexpected calculation')),patch('fuelfield.comparison.calculate_plane',side_effect=AssertionError('unexpected comparison calculation')):
            for canvas in (self.w.curvecanvas,c.canvas):
                canvas.draw();ax=canvas.figure.axes[0];x=np.mean(ax.get_xlim());y=np.mean(ax.get_ylim());pixel=ax.transData.transform((x,y))
                canvas.interaction.motion(MouseEvent('motion_notify_event',canvas,*pixel))
                canvas.interaction.press(MouseEvent('button_press_event',canvas,*pixel,button=MouseButton.LEFT,dblclick=True))
                self.assertEqual(len(canvas.interaction.points),1)
        np.testing.assert_array_equal(before,self.w.results[0].cell_mass_kg_s)

if __name__=='__main__':unittest.main()

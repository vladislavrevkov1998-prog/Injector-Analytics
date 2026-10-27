"""Native Qt drags, graph Undo isolation, themes and readable export."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PyQt5.QtCore import Qt,QPoint,QPointF,QSettings,QEvent
from PyQt5.QtGui import QMouseEvent,QImage
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication,QDialog,QLabel,QVBoxLayout,QTableWidget,QTableWidgetItem
from PyQt5.QtGui import QColor
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.image import imread
from matplotlib.colors import to_rgba

from fuelfield.plot_interaction import attach_interaction
from fuelfield.gui import MainWindow,INTCOL
from fuelfield.core import project_from_json,project_to_json,calculate_plane,evaluate_continuous_field
from fuelfield.axes_settings import open_axes_settings
from fuelfield.themes import apply_theme,current_theme,PALETTES

ROOT=Path(__file__).resolve().parents[1]


def qt_position(canvas,ax,point):
    x,y=ax.transData.transform(point);ratio=canvas.device_pixel_ratio
    return QPoint(round(x/ratio),round(canvas.height()-y/ratio))


def move_held(canvas,position,button=Qt.LeftButton):
    event=QMouseEvent(QMouseEvent.MouseMove,QPointF(position),QPointF(canvas.mapToGlobal(position)),Qt.NoButton,button,Qt.NoModifier)
    QApplication.sendEvent(canvas,event);QApplication.processEvents()


def pixels(canvas):
    image=canvas.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    data=image.bits();data.setsize(image.byteCount())
    return np.frombuffer(data,np.uint8).reshape(image.height(),image.bytesPerLine()//4,4)[:,:image.width()].copy()


class ThemeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.previous_theme=current_theme();self.temp=tempfile.TemporaryDirectory()
        self.settings=QSettings(str(Path(self.temp.name)/'preferences.ini'),QSettings.IniFormat)
        self.preference_patch=patch('fuelfield.themes.preferences',return_value=self.settings);self.preference_patch.start()
        apply_theme(self.app,'light',persist=False)
    def tearDown(self):
        apply_theme(self.app,self.previous_theme,persist=False)
        self.preference_patch.stop();self.temp.cleanup();self.app.processEvents()


class TestNativeV08Drag(ThemeTestCase):
    def setUp(self):
        super().setUp();self.figure=Figure(figsize=(6,4));self.canvas=FigureCanvasQTAgg(self.figure)
        self.canvas.resize(600,400);self.ax=self.figure.add_subplot();self.ax.plot([0,10],[0,20]);self.ax.set_xlim(0,10);self.ax.set_ylim(0,20)
        self.controller=attach_interaction(self.canvas);self.canvas.show();self.app.processEvents();self.canvas.draw()
    def tearDown(self):
        self.canvas.close();self.canvas.deleteLater();self.app.processEvents();super().tearDown()

    def test_held_lmb_changes_actual_line_pixels_before_release(self):
        start=qt_position(self.canvas,self.ax,(3,6));first=qt_position(self.canvas,self.ax,(7,16));second=qt_position(self.canvas,self.ax,(7,3))
        before=pixels(self.canvas);QTest.mousePress(self.canvas,Qt.LeftButton,pos=start)
        with patch.object(self.canvas,'draw',wraps=self.canvas.draw) as draw:
            move_held(self.canvas,first);self.assertEqual(len(self.controller.oblique),1)
            line=self.controller.oblique[0].artists[0];x,y=line.get_data();first_slope=np.diff(y)[0]/np.diff(x)[0]
            first_frame=pixels(self.canvas)
            move_held(self.canvas,second);x,y=line.get_data();second_slope=np.diff(y)[0]/np.diff(x)[0]
            second_frame=pixels(self.canvas)
            self.assertEqual(draw.call_count,0,'drag should paint its overlay without rerendering source charts')
        self.assertGreater(first_slope,2.3);self.assertLess(second_slope,-.5)
        self.assertGreater(np.count_nonzero(first_frame!=before),200)
        self.assertGreater(np.count_nonzero(second_frame!=first_frame),200)
        # Green guide pixels occur at a point on the new negative-slope line,
        # well away from the original blue y=2x series and a fake horizontal.
        guide=self.controller.oblique[0];p=qt_position(self.canvas,self.ax,(5,guide.y+guide.slope*(5-guide.x)))
        ratio=self.canvas.device_pixel_ratio;cx,cy=round(p.x()*ratio),round(p.y()*ratio)
        roi=second_frame[cy-5:cy+6,cx-5:cx+6,:3].astype(int)
        self.assertTrue(np.any((roi[:,:,1]>roi[:,:,0]+15)&(roi[:,:,1]>roi[:,:,2]+4)))
        self.assertIsNotNone(self.controller.slope_drag)
        QTest.mouseRelease(self.canvas,Qt.LeftButton,pos=second);self.app.processEvents()
        self.assertTrue(self.controller.history().can_undo());self.controller.history().undo();self.assertFalse(self.controller.oblique)
        self.controller.history().redo();self.assertEqual(len(self.controller.oblique),1)

    def test_vertical_line_and_escape_cancel_without_undo_entry(self):
        start=qt_position(self.canvas,self.ax,(3,6));end=QPoint(start.x(),start.y()-80)
        QTest.mousePress(self.canvas,Qt.LeftButton,pos=start);move_held(self.canvas,end)
        # Pixel rounding can move the snapped data anchor slightly; all line
        # endpoints must nevertheless remain finite in the near-vertical case.
        line=self.controller.oblique[0].artists[0]
        self.assertTrue(np.isfinite(np.asarray(line.get_data())).all())
        QTest.keyClick(self.canvas,Qt.Key_Escape);self.app.processEvents()
        self.assertFalse(self.controller.oblique);self.assertFalse(self.controller.history().can_undo())
        QTest.mouseRelease(self.canvas,Qt.LeftButton,pos=end)

    def test_thirty_moves_are_one_command_and_leave_cancels_preview(self):
        start=qt_position(self.canvas,self.ax,(3,6));end=qt_position(self.canvas,self.ax,(7,16))
        QTest.mousePress(self.canvas,Qt.LeftButton,pos=start)
        for fraction in np.linspace(.1,1,30):
            position=QPoint(round(start.x()+(end.x()-start.x())*fraction),round(start.y()+(end.y()-start.y())*fraction));move_held(self.canvas,position)
        self.assertFalse(self.controller.history().can_undo())
        QTest.mouseRelease(self.canvas,Qt.LeftButton,pos=end);self.assertEqual(self.controller.history().index,1)
        json.dumps(self.controller.snapshot(),allow_nan=False)
        self.controller.history().undo();self.assertFalse(self.controller.oblique)
        QTest.mousePress(self.canvas,Qt.LeftButton,pos=start);move_held(self.canvas,end)
        QApplication.sendEvent(self.canvas,QEvent(QEvent.Leave));self.app.processEvents()
        self.assertFalse(self.controller.oblique);self.assertFalse(self.controller.history().can_undo())
        QTest.mouseRelease(self.canvas,Qt.LeftButton,pos=end)


class TestV08ThemeSurfaces(ThemeTestCase):
    def test_fresh_table_cells_are_readable_after_dark_theme_and_restore(self):
        table=QTableWidget(1,1);table.show();self.app.processEvents();apply_theme(self.app,'dark',persist=False)
        item=QTableWidgetItem('12.345');item.setBackground(QColor('#eee8f6'));item.setForeground(QColor('#52617c'));table.setItem(0,0,item)
        changes=[];table.itemChanged.connect(lambda changed:changes.append(changed.text()))
        self.app.processEvents();table.viewport().repaint();self.app.processEvents()
        self.assertLess(item.background().color().lightnessF(),.3);self.assertGreater(item.foreground().color().lightnessF(),.5)
        self.assertEqual(changes,[]);self.assertEqual(item.text(),'12.345')
        apply_theme(self.app,'light',persist=False)
        self.assertEqual(item.background().color().name(),'#eee8f6');self.assertEqual(item.foreground().color().name(),'#52617c')
        table.close();table.deleteLater();self.app.processEvents()


class TestV08ResultHistory(ThemeTestCase):
    def setUp(self):
        super().setUp();self.w=MainWindow();self.w._restoring=True
        self.w.cfg,self.w.nozzles=project_from_json(json.loads((ROOT/'examples/demo_project_user_field.json').read_text()))
        self.w.cfg.z_mm=[30.];self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history()
        self.w.nozzle_table.item(0,INTCOL['flow_kg_h']).setText('9.125')
        self.assertTrue(self.w.undo_stack.canUndo())
        plane=calculate_plane(self.w.nozzles,self.w.cfg,30.);field=evaluate_continuous_field(self.w.nozzles,self.w.cfg,30.)
        self.w._done([(plane,field)]);self.w.tabs.setCurrentIndex(3);self.w.plots.setCurrentIndex(3);self.w.plot_count.setValue(1)
        self.w.preview_timer.stop();self.w.show();self.app.processEvents();self.canvas=self.w.curvecanvas;self.canvas.draw();self.canvas.setFocus()
    def tearDown(self):
        self.w.close();self.w.deleteLater();self.app.processEvents();super().tearDown()

    def test_ctrl_z_and_redo_only_restore_last_graph_action(self):
        result=self.w.results;fields=self.w.smooth_fields;before=project_to_json(self.w.cfg,self.w.nozzles);index=self.w.undo_stack.index()
        ax=self.canvas.figure.axes[0];line=ax.lines[0];x=np.asarray(line.get_xdata());y=np.asarray(line.get_ydata());point=qt_position(self.canvas,ax,(float(x[len(x)//2]),float(y[len(y)//2])))
        QTest.mouseDClick(self.canvas,Qt.LeftButton,pos=point);QTest.mouseRelease(self.canvas,Qt.LeftButton,pos=point);self.app.processEvents()
        self.assertEqual(len(self.canvas.interaction.points),1)
        with patch.object(self.w,'_invalidate_results',side_effect=AssertionError('graph undo invalidated calculation')):
            QTest.keyClick(self.canvas,Qt.Key_Z,Qt.ControlModifier);self.app.processEvents();self.assertFalse(self.canvas.interaction.points)
            QTest.keyClick(self.canvas,Qt.Key_Y,Qt.ControlModifier);self.app.processEvents();self.assertEqual(len(self.canvas.interaction.points),1)
            QTest.keyClick(self.canvas,Qt.Key_Z,Qt.ControlModifier);QTest.keyClick(self.canvas,Qt.Key_Z,Qt.ControlModifier);self.app.processEvents()
        self.assertIs(self.w.results,result);self.assertIs(self.w.smooth_fields,fields);self.assertEqual(self.w.undo_stack.index(),index)
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),before)

    def test_axis_dialog_is_one_undoable_graph_action(self):
        ax=self.canvas.figure.axes[0];original=(ax.get_xlim(),ax.get_ylim());dialog=open_axes_settings(self.canvas,self.w)
        dialog.widgets['xmin'].setValue(20);dialog.widgets['xmax'].setValue(160);dialog.apply_settings()
        self.assertEqual(ax.get_xlim(),(20.,160.));self.w._undo_action();self.assertEqual((ax.get_xlim(),ax.get_ylim()),original)
        self.w._redo_action();self.assertEqual(ax.get_xlim(),(20.,160.));self.assertTrue(self.w.results);dialog.close()

    def test_theme_preserves_project_results_and_styles_later_canvas_and_export(self):
        before=project_to_json(self.w.cfg,self.w.nozzles);results=self.w.results;fields=self.w.smooth_fields
        item_changes=[];self.w.nozzle_table.itemChanged.connect(lambda item:item_changes.append(item.text()))
        with patch.object(self.w,'run_calc',side_effect=AssertionError('theme recalculated physics')),patch.object(self.w,'_invalidate_results',side_effect=AssertionError('theme invalidated calculation')):
            self.w.set_theme('dark');self.app.processEvents()
        self.assertEqual(current_theme(),'dark');self.assertEqual(self.settings.value('appearance/theme'),'dark')
        self.assertEqual(len(item_changes),0,'theme must not dispatch source-table edits')
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),before);self.assertIs(self.w.results,results);self.assertIs(self.w.smooth_fields,fields)
        self.assertLess(self.w.geometry_editor.sketch.backgroundBrush().color().lightnessF(),.25)
        figure=Figure();canvas=FigureCanvasQTAgg(figure);ax=figure.add_subplot();image=ax.imshow(np.arange(4).reshape(2,2));figure.colorbar(image,ax=ax)
        canvas.draw();self.assertEqual(figure.get_facecolor(),to_rgba(PALETTES['dark']['window']))
        self.assertTrue(all(np.mean(a.xaxis.label.get_color()[:3])>.5 if isinstance(a.xaxis.label.get_color(),tuple) else a.xaxis.label.get_color()==PALETTES['dark']['text'] for a in figure.axes))
        output=Path(self.temp.name)/'readable.png';figure.savefig(output,dpi=60)
        self.assertGreater(float(imread(output)[0,0,:3].mean()),.85)
        self.assertEqual(figure.get_facecolor(),to_rgba(PALETTES['dark']['window']))
        dialog=QDialog(self.w);layout=QVBoxLayout(dialog);label=QLabel('Новый диалог');label.setStyleSheet('color:#52617c;');layout.addWidget(label);dialog.show();self.app.processEvents()
        self.assertGreater(label.palette().windowText().color().lightnessF(),.5)
        dialog.close();canvas.close();canvas.deleteLater();self.w.set_theme('light');self.app.processEvents()
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),before)


if __name__=='__main__':unittest.main()

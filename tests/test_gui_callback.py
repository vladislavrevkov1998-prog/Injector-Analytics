"""Regression for the GUI result-handling slot, without a PyQt5 dependency.

The actual MainWindow._done function body is executed against small widget fakes.
A worker emits tuples (plane, continuous_preview); the selector must use the
plane objects, not the tuple itself.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fuelfield.core import Settings, source_flow_kg_s


class FakeSelector:
    def __init__(self):
        self.items = []
        self._blocked = False
        self.current = -1
        self.callback = None

    def blockSignals(self, enabled):
        self._blocked = bool(enabled)

    def clear(self):
        self.items = []
        self.current = -1

    def addItem(self, text, data):
        self.items.append((text, data))

    def setCurrentIndex(self, value):
        self.current = value
        if not self._blocked and self.callback is not None:
            self.callback()

    def currentIndex(self):
        return self.current


class FakeProgress:
    def __init__(self):
        self.value = None
    def setValue(self, value):
        self.value = value


class FakeStatusBar:
    def __init__(self):
        self.message = None
    def showMessage(self, message, timeout=None):
        self.message = message


class TestDoneGuiCallback(unittest.TestCase):
    def test_worker_tuple_results_populate_selector_and_switch_plane(self):
        gui = Path(__file__).resolve().parents[1] / 'src' / 'fuelfield' / 'gui.py'
        tree = ast.parse(gui.read_text(encoding='utf-8'), filename=str(gui))
        window = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MainWindow')
        fn = next(n for n in window.body if isinstance(n, ast.FunctionDef) and n.name == '_done')
        module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
        namespace = {'source_flow_kg_s': source_flow_kg_s}
        exec(compile(module, str(gui), 'exec'), namespace)

        class WindowFake:
            def __init__(self):
                self.results = []
                self.smooth_fields = []
                self.progress = FakeProgress()
                self.plane_select = FakeSelector()
                self.plane_select.callback = self._plot_result
                self.nozzles = []
                self.cfg = Settings()
                self._status = FakeStatusBar()
                self.drawn_z = None
                self.draw_count = 0
            def statusBar(self):
                return self._status
            def _plot_result(self):
                if self.results and self.plane_select.currentIndex() >= 0:
                    self.drawn_z = self.results[self.plane_select.currentIndex()].z_mm
                    self.draw_count += 1

        win = WindowFake()
        planes = [SimpleNamespace(z_mm=z) for z in (0, 20, 40, 60)]
        previews = [object() for _ in planes]
        namespace['_done'](win, list(zip(planes, previews)))

        self.assertEqual(win.results, planes)
        self.assertEqual(win.smooth_fields, previews)
        self.assertEqual(win.progress.value, 4)
        self.assertEqual([t for t, _ in win.plane_select.items],
                         ['№1: z=0 мм', '№2: z=20 мм', '№3: z=40 мм', '№4: z=60 мм'])
        self.assertEqual(win.drawn_z, 0)
        self.assertEqual(win.draw_count, 1)
        win.plane_select.setCurrentIndex(2)
        self.assertEqual(win.drawn_z, 40)
        self.assertEqual(win.draw_count, 2)
        self.assertIn('4 сечения', win._status.message)


if __name__ == '__main__':
    unittest.main()

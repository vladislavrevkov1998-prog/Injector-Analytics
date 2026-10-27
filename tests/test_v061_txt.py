"""The distributed TXT must reconstruct a clean, usable installation."""
import contextlib
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from package_txt import build_bundle, GUIDES


class TestTxtDistribution(unittest.TestCase):
    def test_archive_reconstructs_runtime_and_opens_all_guides_without_repository(self):
        with tempfile.TemporaryDirectory(prefix='Копия IDLE ') as folder:
            folder = Path(folder)
            archive = build_bundle(ROOT, folder / 'build', 'test-commit')
            with zipfile.ZipFile(archive) as bundle:
                for name in bundle.namelist():
                    self.assertFalse(any(part in name.split('/') for part in ('examples', 'docs', 'tests', '__pycache__', '.git')))
                    self.assertFalse(name.endswith(('.json', '.py', '.pyc')))
                bundle.extractall(folder / 'extract')
            base = next((folder / 'extract').iterdir())
            scope = {'__name__': 'portable_restore'}
            exec((base / 'ВОССТАНОВИТЬ_ПРОГРАММУ.txt').read_text(encoding='utf-8'), scope)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                destination = scope['restore'](base)
                self.assertEqual(scope['restore'](base), destination)
            self.assertIn(sys.executable, output.getvalue())
            for source in (ROOT / 'src/fuelfield').rglob('*.py'):
                self.assertEqual((destination / source.relative_to(ROOT)).read_bytes(), source.read_bytes())
            code = '''
import sys
sys.path.insert(0, 'src')
from PyQt5.QtWidgets import QApplication, QTextBrowser
from fuelfield.gui import MainWindow
from fuelfield.help_text import read_help,HELP_IMAGES,TRANSPORT_HELP_IMAGES
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QTextDocument
app=QApplication([]);w=MainWindow();w.show();app.processEvents()
assert len(w.nozzles)==140
for name in %r:
    assert len(read_help(name))>2000
w.geometry_editor.show_guide()
assert len(w.geometry_editor.guide_dialog.findChild(QTextBrowser).toPlainText())>2000
for relative in HELP_IMAGES:
    resource=w.geometry_editor.guide_dialog.findChild(QTextBrowser).document().resource(QTextDocument.ImageResource,QUrl(relative))
    assert resource is not None and not resource.isNull(),relative
w.geometry_editor.guide_dialog.close()
w.open_regression();w.regression_dialog.show_guide()
assert len(w.regression_dialog.guide_dialog.findChild(QTextBrowser).toPlainText())>2000
w.regression_dialog.guide_dialog.close();w.regression_dialog.close()
w.open_quasi_settings();w.quasi_dialog.show_guide()
assert len(w.quasi_dialog.guide_dialog.findChild(QTextBrowser).toPlainText())>2000
w.quasi_dialog.guide_dialog.close();w.quasi_dialog.close()
w.open_transport_settings();w.transport_dialog.show_guide()
browser=w.transport_dialog.guide_dialog.findChild(QTextBrowser)
assert 'сохранение массы' in browser.toPlainText().lower()
for relative in TRANSPORT_HELP_IMAGES:
    picture=browser.document().resource(QTextDocument.ImageResource,QUrl(relative))
    assert picture is not None and not picture.isNull(),relative
    assert 600<picture.width()<=840,relative
browser.anchorClicked.emit(QUrl('V0.7_LITERATURE_REVIEW.md'))
assert 'Дитякин' in browser.toPlainText()
browser.anchorClicked.emit(QUrl('V0.7_GUIDE.md'))
assert 'Как включить и получить результат' in browser.toPlainText()
w.transport_dialog.guide_dialog.close();w.transport_dialog.close()
# The restored distribution has no docs/ or repository. Air help and its linked
# mathematical model must therefore be read from the actual embedded payload.
w.open_air_balance();w.air_dialog.show_help()
assert not w.air_dialog.enabled.isChecked()
assert w.air_dialog.table.rowCount()==0
browser=w.air_dialog.guide_dialog.findChild(QTextBrowser)
assert len(browser.toPlainText())>2000
assert 'L0' in browser.toPlainText()
assert 'V0.8_IMPLEMENTED_MODEL.md' in browser.toPlainText()
browser.anchorClicked.emit(QUrl('V0.8_IMPLEMENTED_MODEL.md'))
assert len(browser.toPlainText())>2000
assert 'Инструкция отсутствует' not in browser.toPlainText()
browser.anchorClicked.emit(QUrl('V0.8_GUIDE.md'))
assert 'Баланс воздуха' in browser.toPlainText()
w.air_dialog.guide_dialog.close();w.air_dialog.close();w.close()
print('PORTABLE_OK')
''' % (GUIDES,)
            result = subprocess.run([sys.executable, '-B', '-c', code], cwd=destination, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PORTABLE_OK', result.stdout)

    def test_corrupt_input_and_existing_modifications_fail_before_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);build_bundle(ROOT,base,'test')
            scope={'__name__':'portable_restore'}
            exec((base/'ВОССТАНОВИТЬ_ПРОГРАММУ.txt').read_text(encoding='utf-8'),scope)
            path=base/'code/src/fuelfield/help_text.py.txt';original=path.read_bytes();path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Повреждён'):scope['restore'](base)
            self.assertFalse((base/('Fuel_Field_Pilot_'+scope['VERSION'])).exists())
            path.write_bytes(original)
            with contextlib.redirect_stdout(io.StringIO()):destination=scope['restore'](base)
            (destination/'main.py').write_text('# user edit')
            with self.assertRaisesRegex(ValueError,'уже изменён'):scope['restore'](base)
            self.assertEqual((destination/'main.py').read_text(),'# user edit')

    def test_zip_bytes_are_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            a=build_bundle(ROOT,Path(folder)/'a','same');b=build_bundle(ROOT,Path(folder)/'b','same')
            self.assertEqual(hashlib.sha256(a.read_bytes()).digest(),hashlib.sha256(b.read_bytes()).digest())

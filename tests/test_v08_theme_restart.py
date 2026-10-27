"""A separate application process restores UI preferences outside project JSON."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
PROBE='''
import json,os,sys
from PyQt5.QtWidgets import QApplication
from fuelfield.themes import apply_theme,current_theme,preferences
app=QApplication([])
if sys.argv[1]=='write':
    apply_theme(app,'dark',persist=True)
    preferences().sync()
    print(json.dumps(dict(pid=os.getpid(),theme=current_theme())))
else:
    from fuelfield.gui import MainWindow
    from fuelfield.core import project_from_json
    from pathlib import Path
    window=MainWindow();window.show();app.processEvents()
    cfg,nozzles=project_from_json(json.loads((Path(sys.argv[2])/'examples/demo_project_user_field.json').read_text()))
    print(json.dumps(dict(pid=os.getpid(),theme=current_theme(),window=app.palette().window().color().name(),
                          air_enabled=cfg.air_model.get('enabled',False),nozzles=len(nozzles))))
    window.close()
'''


class TestThemeRestart(unittest.TestCase):
    def test_dark_preference_survives_a_real_process_restart_without_legacy_activation(self):
        with tempfile.TemporaryDirectory(prefix='v08-theme-profile-') as profile:
            env=os.environ.copy();env.update(XDG_CONFIG_HOME=profile,PYTHONPATH=str(ROOT/'src'))
            outputs=[]
            for phase in ('write','read'):
                run=subprocess.run([sys.executable,'-B','-c',PROBE,phase,str(ROOT)],
                    env=env,capture_output=True,text=True,timeout=60)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                outputs.append(json.loads(run.stdout.strip().splitlines()[-1]))
            before,after=outputs
            self.assertNotEqual(before['pid'],after['pid'])
            self.assertEqual(before['theme'],'dark');self.assertEqual(after['theme'],'dark')
            self.assertEqual(after['window'],'#1b2230');self.assertFalse(after['air_enabled'])
            self.assertEqual(after['nozzles'],140)

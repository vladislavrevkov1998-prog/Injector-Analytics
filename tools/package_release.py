"""Build a source ZIP of a clean committed checkout, excluding stale bytecode."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
if args.out.exists():raise SystemExit('Output exists; choose a new archive path.')
for command in (['git','diff','--quiet'],['git','diff','--cached','--quiet']):
    if subprocess.run(command,cwd=root).returncode:raise SystemExit('Commit tracked changes before packaging.')
version=None
for node in ast.parse((root/'src/fuelfield/__init__.py').read_text()).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__version__' for t in node.targets):
        version=ast.literal_eval(node.value)
if not version:raise SystemExit('Package version not found.')
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
paths=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode('utf-8').split('\0')
args.out.parent.mkdir(parents=True,exist_ok=True)
prefix=f'Fuel_Field_Pilot_{version}'
with zipfile.ZipFile(args.out,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name in paths:
        if not name or '__pycache__' in Path(name).parts or name.endswith(('.pyc','.pyo')):continue
        archive.write(root/name,f'{prefix}/{name}')
    archive.writestr(f'{prefix}/BUILD_INFO.json',json.dumps(dict(version=version,commit=commit),indent=2))
digest=hashlib.sha256(args.out.read_bytes()).hexdigest()
args.out.with_suffix(args.out.suffix+'.sha256').write_text(f'{digest}  {args.out.name}\n')
print(json.dumps(dict(path=str(args.out.resolve()),version=version,commit=commit,sha256=digest)))

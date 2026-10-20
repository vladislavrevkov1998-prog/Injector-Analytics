"""Build the clean, verifiable IDLE TXT distribution from runtime sources only."""
import argparse
import base64
import hashlib
from pathlib import Path
import subprocess
import zipfile

GUIDES = ('V0.6_GUIDE.md', 'QUASISTATIONARY_GUIDE.md', 'V0.4.1_IDENTIFICATION_GUIDE.md')

RESTORE = '''# Скопируйте этот текст в IDLE, сохраните рядом с папкой code как restore.py и нажмите F5.
from pathlib import Path
import hashlib
import os
import shlex
import sys

FILES = __FILES__
VERSION = __VERSION__
SOURCE_COMMIT = __COMMIT__

def restore(base=None):
    base = Path(base) if base is not None else Path(__file__).resolve().parent
    destination = base / ('Fuel_Field_Pilot_' + VERSION)
    pending = []
    # Проверяем ВСЕ входные файлы и конфликты до первой записи.
    for relative, checksum in FILES.items():
        source = base / ('code/' + relative + '.txt' if relative.endswith('.py') else relative)
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError('Повреждён файл: ' + str(source))
        target = destination / relative
        if any(parent.exists() and not parent.is_dir() for parent in target.parents):
            raise ValueError('Вместо папки уже существует файл: ' + str(target))
        if target.exists() and target.read_bytes() != content:
            raise ValueError('Файл уже изменён; выберите другую папку: ' + str(target))
        pending.append((target, content))
    for target, content in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    executable = sys.executable
    if Path(executable).name.lower() == 'pythonw.exe':
        executable = str(Path(executable).with_name('python.exe'))
    command = [executable, '-m', 'pip', 'install', '-r', str(destination / 'requirements.txt')]
    rendered = ('& ' + ' '.join("'" + part.replace("'", "''") + "'" for part in command)) if os.name == 'nt' else shlex.join(command)
    print('Версия:', VERSION, 'Исходный commit:', SOURCE_COMMIT)
    print('Программа восстановлена:', destination)
    print('Установите зависимости в PowerShell / терминале:\\n' + rendered)
    print('Затем откройте в IDLE файл', destination / 'main.py', 'и нажмите F5.')
    return destination

if __name__ == '__main__':
    if sys.version_info < (3, 12):
        raise SystemExit('Нужен Python 3.12 или новее.')
    restore()
'''


def build_bundle(root, output, commit):
    root, output = Path(root), Path(output)
    namespace = {}
    exec((root / 'src/fuelfield/__init__.py').read_text(encoding='utf-8'), namespace)
    version = namespace['__version__']
    files = {'main.py': (root / 'main.py').read_bytes(),
             'requirements.txt': (root / 'requirements.txt').read_bytes()}
    for path in sorted((root / 'src/fuelfield').rglob('*.py')):
        files[path.relative_to(root).as_posix()] = path.read_bytes()
    help_text = {name: (root / 'docs' / name).read_text(encoding='utf-8') for name in GUIDES}
    images={path.relative_to(root/'docs').as_posix():base64.b64encode(path.read_bytes()).decode('ascii') for path in sorted((root/'docs/images/v0.6.3').glob('*.png')) if path.name in ('current-model.png','narrowing.png','widening.png')}
    files['src/fuelfield/runtime_help.py'] = ('# Embedded runtime instructions for the clean TXT distribution.\nHELP = ' + repr(help_text) + '\nIMAGES = '+repr(images)+'\n').encode('utf-8')
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    restore = RESTORE.replace('__FILES__', repr(hashes)).replace('__VERSION__', repr(version)).replace('__COMMIT__', repr(commit))
    payload = {('code/' + name + '.txt' if name.endswith('.py') else name): content for name, content in files.items()}
    payload['ВОССТАНОВИТЬ_ПРОГРАММУ.txt'] = restore.encode('utf-8')
    payload['ПРОЧИТАЙ_МЕНЯ.txt'] = (f'''Fuel Field Pilot {version} — чистый комплект для IDLE
Исходный commit: {commit}

1. Установите Python 3.12 или новее и распакуйте весь архив в отдельную папку.
2. Откройте IDLE → File → New File. Скопируйте ВОССТАНОВИТЬ_ПРОГРАММУ.txt.
3. Сохраните как restore.py рядом с папкой code, затем Run → Run Module (F5).
4. Выполните напечатанную команду установки зависимостей в PowerShell / терминале.
5. Откройте восстановленный main.py в IDLE и нажмите F5.

Альтернатива: перенесите вручную файлы из code, убрав последнее .txt и сохранив
структуру папок. requirements.txt нужен для установки зависимостей.
Программа запускается со штатной конфигурацией; собственные проекты загружаются
через «Открыть». Примеров, проектов, документационных папок и временных файлов
в комплекте нет. Три встроенные инструкции являются частью работающего UI.
Повторный запуск restore.py проверяет SHA-256 и не перезаписывает изменённый код.
Зависимости требуют интернета при установке; архив не содержит интерпретатор.
''').encode('utf-8')
    output.mkdir(parents=True, exist_ok=True)
    for name, content in payload.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    archive = output / f'Fuel_Field_Pilot_{version}_IDLE_TXT.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, content in sorted(payload.items()):
            info = zipfile.ZipInfo(f'Fuel_Field_Pilot_{version}_TXT/{name}', date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, content)
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'main.py', 'requirements.txt', 'docs'], cwd=root, text=True).strip():
        raise SystemExit('Commit runtime sources and guides before stamping the distribution.')
    print(build_bundle(root, args.out or root / 'txt', commit))

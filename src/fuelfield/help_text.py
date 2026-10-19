"""Read repository guides or the embedded guides in a portable TXT build."""
from pathlib import Path


def read_help(name):
    try:return (Path(__file__).resolve().parents[2]/'docs'/name).read_text(encoding='utf-8')
    except OSError:
        try:
            from .runtime_help import HELP
            return HELP[name]
        except (ImportError,KeyError):return 'Инструкция отсутствует в этой копии программы. Полная версия доступна в репозитории Injector-Analytics.'

"""Read repository guides or the embedded guides in a portable TXT build."""
from pathlib import Path
import base64

HELP_IMAGES=('images/v0.6.3/current-model.png','images/v0.6.3/narrowing.png','images/v0.6.3/widening.png')


def read_help(name):
    try:return (Path(__file__).resolve().parents[2]/'docs'/name).read_text(encoding='utf-8')
    except OSError:
        try:
            from .runtime_help import HELP
            return HELP[name]
        except (ImportError,KeyError):return 'Инструкция отсутствует в этой копии программы. Полная версия доступна в репозитории Injector-Analytics.'


def populate_help(browser,name):
    """Qt document resources also work in a clean TXT install without docs/."""
    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QImage,QTextDocument
    browser.setMarkdown(read_help(name))
    for relative in HELP_IMAGES if name=='V0.6_GUIDE.md' else ():
        try:data=(Path(__file__).resolve().parents[2]/'docs'/relative).read_bytes()
        except OSError:
            try:
                from .runtime_help import IMAGES
                data=base64.b64decode(IMAGES[relative])
            except (ImportError,KeyError):continue
        browser.document().addResource(QTextDocument.ImageResource,QUrl(relative),QImage.fromData(data))
    browser.document().markContentsDirty(0,browser.document().characterCount())

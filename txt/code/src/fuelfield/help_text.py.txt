"""Read repository guides or the embedded guides in a portable TXT build."""
from pathlib import Path
import base64

HELP_IMAGES=('images/v0.6.3/current-model.png','images/v0.6.3/narrowing.png','images/v0.6.3/widening.png')
TRANSPORT_HELP_IMAGES=('images/v0.7/narrowing-transport.png','images/v0.7/widening-transport.png')
HELP_IMAGES_BY_GUIDE={'V0.6_GUIDE.md':HELP_IMAGES,
    'V0.7_GUIDE.md':TRANSPORT_HELP_IMAGES,
    'V0.7_LITERATURE_REVIEW.md':TRANSPORT_HELP_IMAGES}
LINKED_GUIDES=('V0.7_GUIDE.md','V0.7_LITERATURE_REVIEW.md','V0.8_GUIDE.md','V0.8_IMPLEMENTED_MODEL.md')


def read_help(name):
    try:return (Path(__file__).resolve().parents[2]/'docs'/name).read_text(encoding='utf-8')
    except OSError:
        try:
            from .runtime_help import HELP
            return HELP[name]
        except (ImportError,KeyError):return 'Инструкция отсутствует в этой копии программы. Полная версия доступна в репозитории Injector-Analytics.'


def populate_help(browser,name):
    """Qt document resources also work in a clean TXT install without docs/."""
    from PyQt5.QtCore import QUrl,Qt
    from PyQt5.QtGui import QImage,QTextDocument
    browser.setMarkdown(read_help(name))
    for relative in HELP_IMAGES_BY_GUIDE.get(name,()):
        try:data=(Path(__file__).resolve().parents[2]/'docs'/relative).read_bytes()
        except OSError:
            try:
                from .runtime_help import IMAGES
                data=base64.b64decode(IMAGES[relative])
            except (ImportError,KeyError):continue
        picture=QImage.fromData(data)
        if picture.width()>840:picture=picture.scaledToWidth(840,Qt.SmoothTransformation)
        browser.document().addResource(QTextDocument.ImageResource,QUrl(relative),picture)
    browser.document().markContentsDirty(0,browser.document().characterCount())
    if name in LINKED_GUIDES and not getattr(browser,'_linked_help_navigation',False):
        browser._linked_help_navigation=True
        browser.setOpenLinks(False)
        browser.anchorClicked.connect(lambda url:_open_linked_help(browser,url))


def _open_linked_help(browser,url):
    """Keep the new guide/review usable without a repository in the TXT build."""
    if url.isRelative() and url.path() in LINKED_GUIDES:
        populate_help(browser,url.path())
        browser.moveCursor(browser.textCursor().Start)
    elif url.scheme() in ('https','http'):
        from PyQt5.QtGui import QDesktopServices
        QDesktopServices.openUrl(url)

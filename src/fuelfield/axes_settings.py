"""Per-epure axis limits and explicit major/minor intervals."""
from .numeric_input import DecimalSpinBox
from PyQt5.QtCore import Qt
from PyQt5 import sip
from PyQt5.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QComboBox,QDoubleSpinBox,QDialogButtonBox,QLabel
from matplotlib.ticker import MultipleLocator,AutoLocator,NullLocator

def open_axes_settings(canvas,parent=None):
    axes=[ax for ax in canvas.figure.axes if ax.name=='rectilinear' and ax.get_label()!='<colorbar>']
    if not axes:return None
    dialog=QDialog(parent);dialog.setWindowTitle('Оси эпюр');layout=QVBoxLayout(dialog);form=QFormLayout();layout.addLayout(form)
    selector=QComboBox()
    for i,ax in enumerate(axes):selector.addItem(f'{i+1}: {ax.get_title() or ax.get_ylabel() or "Эпюра"}')
    form.addRow('График / ось Y',selector);widgets={}
    for key,label in [('xmin','X min'),('xmax','X max'),('ymin','Y min'),('ymax','Y max'),('xmajor','X: шаг основных делений'),('xminor','X: шаг промежуточных'),('ymajor','Y: шаг основных делений'),('yminor','Y: шаг промежуточных')]:
        spin=DecimalSpinBox();spin.setDecimals(9);spin.setRange(-1e12 if key.endswith(('min','max')) else 0,1e12);spin.setSingleStep(.0001);form.addRow(label,spin);widgets[key]=spin
    note=QLabel('Шаг 0: автоматические основные / без промежуточных делений. Ctrl + средняя кнопка возвращает исходный диапазон.');note.setWordWrap(True);layout.addWidget(note)
    error=QLabel();layout.addWidget(error)
    def fill():
        ax=axes[selector.currentIndex()]
        for key,value in zip(('xmin','xmax','ymin','ymax'),(*ax.get_xlim(),*ax.get_ylim())):widgets[key].setValue(value)
        for axis,prefix in ((ax.xaxis,'x'),(ax.yaxis,'y')):
            for which in ('major','minor'):
                locator=getattr(axis,'get_'+which+'_locator')();widgets[prefix+which].setValue(getattr(locator,'_edge',None).step if isinstance(locator,MultipleLocator) else 0)
    def apply():
        values={k:w.value() for k,w in widgets.items()}
        if values['xmin']>=values['xmax'] or values['ymin']>=values['ymax']:error.setText('min должен быть меньше max');return
        # Bound tick count so an accidental tiny step cannot freeze the renderer.
        for prefix in ('x','y'):
            span=values[prefix+'max']-values[prefix+'min']
            if any(0<values[prefix+kind]<span/500 for kind in ('major','minor')):error.setText('Не более 500 делений на ось: увеличьте шаг.');return
        ax=axes[selector.currentIndex()]
        if sip.isdeleted(canvas) or ax not in canvas.figure.axes:error.setText('Данные графика изменились. Откройте настройки осей заново.');return
        controller=getattr(canvas,'interaction',None)
        before=controller.history().capture(controller.history_controllers()) if controller is not None else None
        ax.set_xlim(values['xmin'],values['xmax']);ax.set_ylim(values['ymin'],values['ymax'])
        for axis,prefix in ((ax.xaxis,'x'),(ax.yaxis,'y')):
            major,minor=values[prefix+'major'],values[prefix+'minor']
            axis.set_major_locator(MultipleLocator(major) if major else AutoLocator());axis.set_minor_locator(MultipleLocator(minor) if minor else NullLocator())
        ax.grid(True,which='major',alpha=.25);ax.grid(True,which='minor',alpha=.12);canvas.draw_idle();error.setText('Применено')
        if controller is not None:controller.history().record('Настройка осей графика',before,controller.history_controllers())
    selector.currentIndexChanged.connect(fill);fill()
    buttons=QDialogButtonBox(QDialogButtonBox.Apply|QDialogButtonBox.Close);buttons.button(QDialogButtonBox.Apply).clicked.connect(apply);buttons.rejected.connect(dialog.close);layout.addWidget(buttons)
    dialog.widgets=widgets;dialog.selector=selector;dialog.apply_settings=apply;dialog.setAttribute(Qt.WA_DeleteOnClose);dialog.show();return dialog

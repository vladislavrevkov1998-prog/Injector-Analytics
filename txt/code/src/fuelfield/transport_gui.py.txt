"""Explicit one-way fuel-transport parameters; edits apply atomically."""
from copy import deepcopy

from PyQt5.QtWidgets import (QComboBox,QDialog,QDialogButtonBox,QFormLayout,
    QGroupBox,QHBoxLayout,QLabel,QLineEdit,QPushButton,QScrollArea,QSpinBox,
    QTextBrowser,QVBoxLayout,QWidget)

from .help_text import populate_help
from .transport_settings import (INTEGER_FIELDS,LABELS,MODES,
    TransportSettings,synthetic_settings)


PHYSICAL_FIELDS=('carrier_inlet_velocity_m_s','injection_speed_m_s',
    'relaxation_time_s','film_speed_m_s','cone_half_width_deg',
    'turbulence_rms_m_s','turbulence_time_s')
NUMERICAL_FIELDS=('max_step_mm','wall_tolerance_mm','carrier_radial_cells',
    'emission_radial_samples','emission_angular_samples','seed')
TOOLTIPS={
    'carrier_inlet_velocity_m_s':'Граничная скорость условного носителя для осесимметричного идеального потока. Это не рассчитанное поле воздуха.',
    'injection_speed_m_s':'Предписанная скорость начальных капель. Она не выводится автоматически из расхода или перепада давления.',
    'relaxation_time_s':'Эффективное время в законе dv/dt=(u−v)/τ. Без данных о каплях это параметр модели, не измеренный размер капли.',
    'film_speed_m_s':'Положительная скорость плёнки по стенке в сторону выхода. Накопление и испарение плёнки этой моделью не разрешаются.',
    'cone_half_width_deg':'Полуширина распределения направления источника относительно центрального угла факела форсунки.',
    'turbulence_rms_m_s':'0 выключает модель флуктуаций. Ненулевое значение задаёт поперечную стохастическую составляющую носителя.',
    'turbulence_time_s':'Требуется для ненулевых флуктуаций. Не является временным шагом квазистационарного обхода.',
    'max_step_mm':'Независимый численный осевой шаг. Не меняет положения расчётных сечений.',
    'wall_tolerance_mm':'Допуск численной аппроксимации профилей стенок; проверяйте сходимость при его уменьшении.',
    'emission_radial_samples':'Количество узлов квадратуры по углу раскрытия источника; независимо от радиальных поясов результата.',
    'emission_angular_samples':'Кратно 36 для сохранения интегральных долей измеренных секторов; независимо от окружных ячеек результата.',
    'seed':'Повторяемые реализации стохастической модели. Seed не является физическим параметром топлива.'}


class TransportDialog(QDialog):
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self._loading=False
        self._raw=deepcopy(owner.cfg.fuel_transport);self._changed_keys=set()
        self.setWindowTitle('Перенос топлива внутри ЖТ');self.resize(920,760)
        layout=QVBoxLayout(self);top=QHBoxLayout();top.addWidget(QLabel('Расчётная модель:'))
        self.mode=QComboBox()
        for value,label in MODES.items():self.mode.addItem(label,value)
        top.addWidget(self.mode,1);guide=QPushButton('Инструкция');guide.clicked.connect(self.show_guide);top.addWidget(guide);layout.addLayout(top)
        notice=QLabel('Новая модель — параметризованный перенос топлива без калибровки: идеальный осесимметричный носитель, инерция капель и заданное граничное условие стенки. Это не решение RANS, не рассчитанное течение воздуха и не подтверждённая модель взаимодействия струй.');notice.setWordWrap(True);layout.addWidget(notice)
        scroll=QScrollArea();scroll.setWidgetResizable(True);body=QWidget();content=QVBoxLayout(body);scroll.setWidget(body);layout.addWidget(scroll,1)
        self.fields={};self.groups=[]
        for title,names in [('Физические параметры — задайте явно',PHYSICAL_FIELDS),('Численная дискретизация',NUMERICAL_FIELDS)]:
            group=QGroupBox(title);form=QFormLayout(group);form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow);form.setRowWrapPolicy(QFormLayout.WrapLongRows)
            for name in names:
                if name in INTEGER_FIELDS and name!='seed':
                    widget=QSpinBox();bounds={'carrier_radial_cells':(4,128),'emission_radial_samples':(1,32),'emission_angular_samples':(36,1440)}
                    widget.setRange(*bounds[name])
                    if name=='emission_angular_samples':widget.setSingleStep(36)
                    widget.valueChanged.connect(lambda value,key=name:self.edited(key))
                else:
                    widget=QLineEdit();widget.setPlaceholderText('Не задано — значение неизвестно')
                    widget.textEdited.connect(lambda value,key=name:self.edited(key))
                widget.setMinimumWidth(180);widget.setMaximumWidth(280);widget.setToolTip(TOOLTIPS.get(name,LABELS[name]));self.fields[name]=widget
                form.addRow(LABELS[name],widget)
            content.addWidget(group);self.groups.append(group)
        wall=QLabel('Граничное условие: при первом ударе капля полностью захватывается в стеночную плёнку. Весь расход остаётся в расчёте; плёнка учитывается отдельной фазой. Её заданная скорость направлена по стенке к выходу. Коэффициенты отскока и отрыв плёнки здесь не вычисляются.');wall.setWordWrap(True);content.addWidget(wall)
        quasi=QLabel('При пульсациях давления скорости и параметры носителя остаются предписанными. Квазистационарное приближение требует, чтобы время переноса было мало относительно изменения режима; запаздывание давления этой постановкой не рассчитывается.');quasi.setWordWrap(True);content.addWidget(quasi);content.addStretch()
        bar=QHBoxLayout();self.example_button=QPushButton('Подставить учебные параметры');self.example_button.setToolTip('Заполняет синтетические предположения только в этом окне; не применяет их к проекту.');self.example_button.clicked.connect(self.use_example);bar.addWidget(self.example_button);bar.addStretch();layout.addLayout(bar)
        self.provenance=QLabel();self.provenance.setWordWrap(True);layout.addWidget(self.provenance)
        self.status=QLabel('Пустое поле означает неизвестное значение. Сохранение проверяет параметры; закрытие без сохранения не меняет проект.');self.status.setWordWrap(True);self.status.setMinimumHeight(35);layout.addWidget(self.status)
        buttons=QDialogButtonBox(QDialogButtonBox.Apply|QDialogButtonBox.Close)
        self.apply_button=buttons.button(QDialogButtonBox.Apply);self.apply_button.setText('Применить');self.apply_button.clicked.connect(self.apply)
        buttons.button(QDialogButtonBox.Close).setText('Закрыть');buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.set_values(TransportSettings.from_dict(self._raw))

    def edited(self,key):
        if self._loading:return
        self._changed_keys.add(key)
        self.status.setStyleSheet('');self.status.setText('Есть неприменённые изменения. Введённые числа не становятся измеренными данными автоматически.')
        self.update_provenance()

    def update_provenance(self):
        provenance=self._raw.get('provenance',{}) if isinstance(self._raw,dict) else {}
        if isinstance(provenance,dict) and provenance.get('source_kind')=='synthetic':
            text='Учебные синтетические параметры; не данные испытаний и не результат калибровки.'
            if self._changed_keys:text+=' Часть значений изменена пользователем.'
        else:text='Происхождение: параметры задаются пользователем; экспериментальное подтверждение и калибровка отсутствуют.'
        self.provenance.setText(text);self.provenance.setStyleSheet('color:#98531c')

    def set_values(self,settings):
        self._loading=True
        try:
            self.mode.setCurrentIndex(self.mode.findData(settings.mode))
            for name,widget in self.fields.items():
                value=getattr(settings,name)
                if isinstance(widget,QSpinBox):
                    widget.setValue(value)
                else:widget.setText('' if value is None else str(value))
            self.mode_changed()
        finally:self._loading=False
        self.update_provenance()

    def mode_changed(self):
        active=self.mode.currentData()=='confined_inertial'
        for group in self.groups:group.setEnabled(active)
        if not self._loading:self.edited('mode')

    def use_example(self):
        settings=synthetic_settings();previous=TransportSettings.from_dict(self._raw).to_dict()
        self._raw={**previous,**settings.to_dict()};self._changed_keys=set()
        if isinstance(previous.get('provenance'),dict):self._raw['provenance']={**previous['provenance'],**self._raw['provenance']}
        self.set_values(settings);self._changed_keys=set()
        self.status.setStyleSheet('color:#98531c');self.status.setText('Подставлены учебные предположения. Проект пока не изменён; для сохранения нажмите «Применить».');self.update_provenance()

    def collect(self):
        raw=deepcopy(self._raw) if isinstance(self._raw,dict) else {}
        raw['mode']=self.mode.currentData()
        for name,widget in self.fields.items():
            if isinstance(widget,QSpinBox):raw[name]=widget.value()
            elif name in INTEGER_FIELDS:
                try:raw[name]=int(widget.text().strip())
                except ValueError:raise ValueError(LABELS[name]+': требуется целое число') from None
            else:
                text=widget.text().strip()
                raw[name]=None if not text else text
        settings=TransportSettings.from_dict(raw).validate()
        return settings.to_dict()

    def apply(self):
        try:
            raw=self.collect();before=TransportSettings.from_dict(self.owner.cfg.fuel_transport).to_dict()
            if raw!=before:self.owner.apply_transport_settings(raw)
            self._raw=deepcopy(raw);self._changed_keys=set();self.update_provenance()
            self.status.setStyleSheet('color:#256144');self.status.setText('Настройки применены. Выполните «Рассчитать сечения» на вкладке 04.')
            return True
        except (ValueError,TypeError,KeyError) as exc:
            self.status.setStyleSheet('color:#a32138');self.status.setText('Настройки не применены: '+str(exc));return False

    def show_guide(self):
        self.guide_dialog=QDialog(self);self.guide_dialog.setWindowTitle('Перенос топлива v0.7 — инструкция');self.guide_dialog.resize(920,760)
        layout=QVBoxLayout(self.guide_dialog);browser=QTextBrowser();populate_help(browser,'V0.7_GUIDE.md');layout.addWidget(browser)
        buttons=QDialogButtonBox(QDialogButtonBox.Close);buttons.button(QDialogButtonBox.Close).setText('Закрыть');buttons.rejected.connect(self.guide_dialog.close);layout.addWidget(buttons);self.guide_dialog.show()

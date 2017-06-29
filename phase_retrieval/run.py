# -*- coding: utf-8 -*-

try:
    from PyQt4 import QtCore, QtGui, QtSvg
    from PyQt4.QtGui import QMovie
    style = 'Plastique'
except:
    from PyQt5 import QtCore, QtSvg
    from PyQt5 import QtWidgets as QtGui
    from PyQt5.QtGui import QMovie
    style = 'Fusion'
    
import subprocess
from subprocess import PIPE
from concurrent.futures import ProcessPoolExecutor as Pool
import sys
import os
python_bin = sys.executable 

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)
    
try:
    from reikna import cluda
    cluda.any_api()
    from phase_retrieval import gpu
    import inspect
    base = os.path.dirname(inspect.getfile(gpu))
    CGStr = os.path.join(base,'conjugate_gradient_algorithm_2D.py')
    HIOStr =os.path.join(base,'HIO_With_ER_algorithm_2D.py')
    ERStr = os.path.join(base,'ER_algorithm_2D.py')
except Exception as e:
    from phase_retrieval import cpu
    import inspect
    base = os.path.dirname(inspect.getfile(cpu))
    CGStr = os.path.join(base,'conjugate_gradient_algorithm_2D.py')
    HIOStr = os.path.join(base,'HIO_With_ER_algorithm_2D.py')
    ERStr = os.path.join(base,'ER_algorithm_2D.py')


class Ui_form(object):    
    def setupUi(self, form):
        self.options = {'algorithm':'Error Reduction','points':64,'signal':'3','param':'1.1'}
        form.setObjectName(_fromUtf8("form"))
        form.resize(730,479)
        self.gridLayout_2 = QtGui.QGridLayout(form)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.main_GLayout = QtGui.QGridLayout()
        self.main_GLayout.setObjectName(_fromUtf8("main_GLayout"))
        self.image_View = QtSvg.QSvgWidget(form)
        self.image_View.setObjectName(_fromUtf8("image_View"))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.image_View.setSizePolicy(sizePolicy)
        self.main_GLayout.addWidget(self.image_View, 0, 3, 1, 1)
        
        self.background = QtSvg.QSvgWidget(form)
        self.background.setObjectName(_fromUtf8("background"))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.background.setSizePolicy(sizePolicy)
        self.main_GLayout.addWidget(self.background, 0, 3, 1, 1)
        self.background.load(b'''<svg width="100" height="100"> <rect width="100" height="100" style="fill:rgb(255,255,255);stroke-width:3;stroke:rgb(255,255,255)"/></svg>''')
        
        self.gif_View = gif(parent=form)
        self.gif_View.setObjectName(_fromUtf8('gif_View'))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.gif_View.setSizePolicy(sizePolicy)
        self.main_GLayout.addWidget(self.gif_View, 0, 3, 1, 1)
        self.gif_View.hide()
        
        self.options_VLayout = QtGui.QVBoxLayout()
        self.options_VLayout.setObjectName(_fromUtf8("options_VLayout"))
        self.algorithm_VLayout = QtGui.QGridLayout()
        self.algorithm_VLayout.setObjectName(_fromUtf8("algorithm_VLayout"))
        self.points = QtGui.QComboBox(form)
        self.points.setObjectName(_fromUtf8("points"))
        self.algorithm_VLayout.addWidget(self.points, 2, 1, 1, 1)
        self.signal = QtGui.QComboBox(form)
        self.signal.setObjectName(_fromUtf8("signal"))
        self.algorithm_VLayout.addWidget(self.signal, 1, 1, 1, 1)
        self.algorithm = QtGui.QComboBox(form)
        self.algorithm.setObjectName(_fromUtf8("algorithm"))
        self.algorithm_VLayout.addWidget(self.algorithm, 0, 1, 1, 1)
        self.points_lab = QtGui.QLabel(form)
        self.points_lab.setObjectName(_fromUtf8("points_lab"))
        self.algorithm_VLayout.addWidget(self.points_lab, 2, 0, 1, 1)
        self.signal_lab = QtGui.QLabel(form)
        self.signal_lab.setObjectName(_fromUtf8("signal_lab"))
        self.algorithm_VLayout.addWidget(self.signal_lab, 1, 0, 1, 1)
        self.algorithm_lab = QtGui.QLabel(form)
        self.algorithm_lab.setObjectName(_fromUtf8("algorithm_lab"))
        self.algorithm_VLayout.addWidget(self.algorithm_lab, 0, 0, 1, 1)
        self.options_VLayout.addLayout(self.algorithm_VLayout)
        self.range_VLayout = QtGui.QGridLayout()
        self.range_VLayout.setObjectName(_fromUtf8("range_VLayout"))
        self.xrange_up = QtGui.QDoubleSpinBox(form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.xrange_up.sizePolicy().hasHeightForWidth())
        self.xrange_up.setSizePolicy(sizePolicy)
        self.xrange_up.setRange(1,3)
        self.xrange_up.setSingleStep(.01)
        self.xrange_up.setValue(1.6)
        self.xrange_up.setObjectName(_fromUtf8("xrange_up"))
        self.range_VLayout.addWidget(self.xrange_up, 0, 4, 1, 1)
        self.xrange_low = QtGui.QDoubleSpinBox(form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.xrange_low.sizePolicy().hasHeightForWidth())
        self.xrange_low.setSizePolicy(sizePolicy)
        self.xrange_low.setRange(-3,-1)
        self.xrange_low.setSingleStep(.01)
        self.xrange_low.setValue(-1.6)
        self.xrange_low.setObjectName(_fromUtf8("xrange_low"))
        self.range_VLayout.addWidget(self.xrange_low, 0, 2, 1, 1)
        self.yrange_lab = QtGui.QLabel(form)
        self.yrange_lab.setObjectName(_fromUtf8("yrange_lab"))
        self.range_VLayout.addWidget(self.yrange_lab, 1, 1, 1, 1)
        self.label_5 = QtGui.QLabel(form)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.range_VLayout.addWidget(self.label_5, 0, 3, 1, 1)
        self.label_7 = QtGui.QLabel(form)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.range_VLayout.addWidget(self.label_7, 1, 3, 1, 1)
        self.yrange_low = QtGui.QDoubleSpinBox(form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.yrange_low.sizePolicy().hasHeightForWidth())
        self.yrange_low.setSizePolicy(sizePolicy)
        self.yrange_low.setRange(-3,-1)
        self.yrange_low.setSingleStep(.01)
        self.yrange_low.setValue(-1.6)
        self.yrange_low.setObjectName(_fromUtf8("yrange_low"))
        self.range_VLayout.addWidget(self.yrange_low, 1, 2, 1, 1)
        self.yrange_up = QtGui.QDoubleSpinBox(form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.yrange_up.sizePolicy().hasHeightForWidth())
        self.yrange_up.setSizePolicy(sizePolicy)
        self.yrange_up.setRange(1,3)
        self.yrange_up.setSingleStep(.01)
        self.yrange_up.setValue(1.6)
        self.yrange_up.setObjectName(_fromUtf8("yrange_up"))
        self.range_VLayout.addWidget(self.yrange_up, 1, 4, 1, 1)
        self.xrange_lab = QtGui.QLabel(form)
        self.xrange_lab.setObjectName(_fromUtf8("xrange_lab"))
        self.range_VLayout.addWidget(self.xrange_lab, 0, 1, 1, 1)
        self.options_VLayout.addLayout(self.range_VLayout)
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.param_VLayout_2 = QtGui.QWidget(form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.param_VLayout_2.sizePolicy().hasHeightForWidth())
        self.param_VLayout_2.setSizePolicy(sizePolicy)
        self.param_VLayout_2.setObjectName(_fromUtf8("param_VLayout_2"))
        self.param_VLayout = QtGui.QGridLayout(self.param_VLayout_2)
        self.param_VLayout.setObjectName(_fromUtf8("param_VLayout"))
        self.param_slide_lab = QtGui.QLabel(form)
        self.param_slide_lab.setObjectName(_fromUtf8('param_slider_lab'))
        self.param_VLayout.addWidget(self.param_slide_lab, 0, 2, 1, 1)
        self.param_lab = QtGui.QLabel(self.param_VLayout_2)
        self.param_lab.setObjectName(_fromUtf8("param_lab"))
        self.param_VLayout.addWidget(self.param_lab, 0, 0, 1, 1)
        self.param_slider = QtGui.QSlider(self.param_VLayout_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.param_slider.sizePolicy().hasHeightForWidth())
        self.param_slider.setSizePolicy(sizePolicy)
        self.param_slider.setMinimum(5)
        self.param_slider.setMaximum(15)
        self.param_slider.setProperty("value", 11)
        self.param_slider.setOrientation(QtCore.Qt.Horizontal)
        self.param_slider.setObjectName(_fromUtf8("param_slider"))
        self.param_VLayout.addWidget(self.param_slider, 0, 1, 1, 1)
        self.gridLayout_3.addWidget(self.param_VLayout_2, 0, 0, 1, 1)
        self.options_VLayout.addLayout(self.gridLayout_3)
        self.submit_btn = QtGui.QPushButton(form)
        self.submit_btn.setObjectName(_fromUtf8("submit_btn"))
        self.options_VLayout.addWidget(self.submit_btn)        
        self.main_GLayout.addLayout(self.options_VLayout, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.main_GLayout, 0, 1, 1, 1)

        self.retranslateUi(form)
        QtCore.QMetaObject.connectSlotsByName(form)
        self.setconnections()
        self.setAlgorithm('Error Reduction')

    def retranslateUi(self, form):
        form.setWindowTitle(_translate("form", "Phase Retrieval", None))
        self.points_lab.setText(_translate("form", "# of points:", None))
        self.signal_lab.setText(_translate("form", "Original Signal:", None))
        self.algorithm_lab.setText(_translate("form", "Algorithm:", None))
        self.yrange_lab.setText(_translate("form", "y range:", None))
        self.label_5.setText(_translate("form", "to ", None))
        self.label_7.setText(_translate("form", "to ", None))
        self.xrange_lab.setText(_translate("form", "x range:", None))
        self.param_lab.setText(_translate("form", "h", None))
        self.param_slide_lab.setText(_translate('form','1.1',None))
        self.submit_btn.setText(_translate("form", "Submit", None))
        
        self.algorithm.addItem(_translate('form','Error Reduction',None))
        self.algorithm.addItem(_translate('form','HIO-ER',None))
        self.algorithm.addItem(_translate('form','Conjugate Gradient',None))
        
        self.points.addItem(_translate('form','64',None))
        self.points.addItem(_translate('form','1024',None))
        self.points.addItem(_translate('form','4096',None))
        self.points.addItem(_translate('form','65536',None))
        
        self.signal.addItem(_translate('form','Paraboloid',None))
        self.signal.addItem(_translate('form','Inverted Paraboloid',None))
        self.signal.addItem(_translate('form','3D Wave',None))
        
    def setconnections(self):
        self.algorithm.activated[str].connect(self.setAlgorithm)
        self.points.activated[str].connect(self.setPoints)
        self.signal.activated[str].connect(self.setSignal)        
        self.submit_btn.clicked.connect(lambda: submit(self))        
        self.param_slider.sliderReleased.connect(self.setSlider)
        self.param_slider.sliderMoved.connect(self.setSlider_lab)
    
    def setAlgorithm(self,text):
        self.options['algorithm'] = text
        if text == 'Error Reduction':
            self.param_VLayout_2.hide()
        elif text == 'HIO-ER':
            self.param_lab.setText(_translate('form',u'\u03B2',None))
            self.param_slider.setValue(11)
            self.param_slide_lab.setText('1.1')
            self.param_VLayout_2.show()
        elif text == 'Conjugate Gradient':
            self.param_lab.setText(_translate('form','h',None))
            self.param_slider.setValue(11)
            self.param_slide_lab.setText('1.1')
            self.param_VLayout_2.show()
    def setPoints(self,num):
        self.options['points'] = int(num)
    def setSignal(self,text):
        if text == 'Paraboloid':
            self.options['signal'] = 3
        elif text == 'Inverted Paraboloid':
            self.options['signal'] = 1
        else:
            self.options['signal'] = 4
    def setSlider(self):
        num = self.param_slider.value()/10
        self.param_slide_lab.setText(str(round(num,1)))
        self.options['param'] = self.param_slide_lab.text()
    def setSlider_lab(self):
        num = self.param_slider.value()/10
        self.param_slide_lab.setText(str(round(num,1)))
        
def submit(widget):
    widget.submit_btn.setEnabled(False)
    signal = widget.options['signal']
    numPoints = int(widget.options['points']**.5)
    xLower = float(widget.xrange_low.value())
    xUpper = float(widget.xrange_up.value())
    yLower = float(widget.yrange_low.value())
    yUpper = float(widget.yrange_up.value())
    param = widget.options['param']
    algorithm = widget.options['algorithm']
    if algorithm == 'Error Reduction':
        args = '{0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(python_bin,ERStr,signal,numPoints,xLower,xUpper,yLower,yUpper,param).split()
    elif algorithm == 'HIO-ER':
        args = '{0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(python_bin,HIOStr,signal,numPoints,xLower,xUpper,yLower,yUpper,param).split()
    else:   
        args = '{0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(python_bin,CGStr,signal,numPoints,xLower,xUpper,yLower,yUpper,param).split()
    future = Pool().submit(make_call,args)
    future.image_View = widget.image_View
    future.gif_View = widget.gif_View
    future.background = widget.background
    future.submit_btn = widget.submit_btn
    future.add_done_callback(image_load)
    widget.image_View.hide()
    widget.background.show()
    widget.gif_View.show()
    return None
        
        
def make_call(args):
    c = subprocess.Popen(args,stdin=PIPE,stdout=PIPE)
    return c.communicate()[0]
        
def image_load(future):
    result = future.result().decode()
    result = result.replace('b\'<?xml','<?xml')
    result = result.replace('b\"<?xml','<?xml')
    result = result.replace('svg>\'','svg>')
    result = result.replace('svg>"','svg>')
    result = result.encode()
    future.image_View.load(result)
    future.image_View.show()
    future.background.hide()
    future.gif_View.hide()
    future.submit_btn.setEnabled(True)


class gif(QtGui.QWidget):
    def __init__(self,*args,parent=None,**kwargs):
        super(gif,self).__init__(parent,*args,**kwargs)
        lab = QtGui.QLabel(self)
        import phase_retrieval.data as data
        import inspect
        base = os.path.dirname(inspect.getfile(data))
        self.movie = QMovie(os.path.join(base,'squares.gif'))
        lab.setMovie(self.movie)
        lab.setAlignment(QtCore.Qt.AlignCenter)
        self.movie.start()
		
def start(*args):
    import sys
    if args:
        app = QtGui.QApplication(*args)
    else:
        app = QtGui.QApplication(['run.py'])
    app.setStyle(style)
    form = QtGui.QWidget()
    ui = Ui_form()
    ui.setupUi(form)
    form.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    import sys
    start(sys.argv)

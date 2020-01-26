import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QLabel
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import random


class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Szacowanie zainteresowania'
        self.filename = ''
        self.layout = QVBoxLayout()
        self.file_label = QLabel()

        self.initUI()

    def initUI(self):
        self.setLayout(self.layout)
        self.setWindowTitle(self.title)
        self.setMinimumSize(800, 600)
        self.filename = self.openFileNameDialog()
        self.show()

        self.add_file_label()
        self.add_plot()

    def add_plot(self):
        plot_canvas = PlotCanvas(width=5, height=4, data=[1, 2, 0, 1, 3, 5])
        self.layout.addWidget(plot_canvas)

    def add_file_label(self):
        file_label = QLabel()
        file_label.setText(self.filename)
        self.layout.addWidget(file_label)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Wszystkie pliki (*);;Plik wideo (*.mp4)", options=options)
        if filename:
            return filename

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100, data=None):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.data = data

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()


    def plot(self):
        #data = [random.random() for i in range(25)]
        ax = self.figure.add_subplot(111)
        ax.plot(self.data, 'r-')
        ax.set_title('Zainteresowanie')
        self.draw()
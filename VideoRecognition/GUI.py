import numpy as np
from PyQt5.QtWidgets import QFileDialog, QLabel
import matplotlib.pyplot as plt

from main import analyze_Video_without_displaying
from PyQt5.QtWidgets import  QVBoxLayout, QSizePolicy, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Szacowanie zainteresowania'
        self.filename = ''
        self.layout = QVBoxLayout()
        self.status_label = QLabel()

        self.initUI()

    def initUI(self):
        self.setLayout(self.layout)
        self.setWindowTitle(self.title)
        self.setMinimumSize(800, 600)
        self.filename = self.openFileNameDialog()
        self.show()

        self.file_label = self.add_file_label()
        self.add_status_label()
        self.run_video_analysis()


    def run_video_analysis(self):
        result = analyze_Video_without_displaying(self.filename)
        self.add_plot(result)
        self.status_label.setText('Analiza ukończona. Wyniki na wykresie poniżej.')

    def add_status_label(self):
        self.status_label.setText("Trwa analiza pliku. To może chwilę zająć.")
        self.layout.addWidget(self.status_label)

    def add_plot(self, data):
        plot_canvas = PlotCanvas(width=50, height=5, data=data)
        self.layout.addWidget(plot_canvas)

    def add_file_label(self):
        file_label = QLabel()
        file_label.setText(self.filename)
        self.layout.addWidget(file_label)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Wybierz plik do analizy", "",
                                                  "Wszystkie pliki (*);;Plik wideo (*.mp4)", options=options)
        if filename:
            return filename

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=1, height=1, dpi=100, data=None):
        fig = Figure(figsize=(width, height), dpi=dpi)
        #self.axes = fig.add_subplot(111)
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
        ax.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        ax.set_ylim([-5, 5])
        ax.grid(True)
        ax.set_title('Zainteresowanie')
        ax.plot(self.data)
        ax.autoscale(axis='x', tight=True)

        self.draw()


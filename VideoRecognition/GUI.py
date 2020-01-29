from PyQt5.QtCore import QRunnable, QThreadPool
from PyQt5.QtWidgets import QFileDialog, QLabel, QHBoxLayout

from main import analyze_Video_without_displaying
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Szacowanie zainteresowania'
        self.filename = ''
        self.layout = QVBoxLayout()
        self.status_label = QLabel()
        self.file_info_layout = QHBoxLayout()
        self.status_info_layout = QHBoxLayout()
        self.result_plot = PlotCanvas(width=50, height=5)
        self.init_UI()

    def init_UI(self):
        self.setLayout(self.layout)
        self.setWindowTitle(self.title)
        self.setMinimumSize(800, 600)
        self.filename = self.open_file_name_dialog()
        self.show()

        self.layout.addLayout(self.file_info_layout)
        self.layout.addLayout(self.status_info_layout)
        self.result_plot.plot(title="Zainteresowanie", data=[0])
        self.layout.addWidget(self.result_plot)

        self.add_file_label()
        self.add_status_label()

        analysis = Runnable(self.filename, self)
        QThreadPool.globalInstance().start(analysis)  # Run analysis in a separate thread

    def analysis_finished(self, result):
        self.status_label.setText('Analiza ukończona. Wyniki na wykresie poniżej.')
        self.result_plot.plot(result, 'Zainteresowanie')  # Update plot with result

    def add_status_label(self):
        self.status_label.setText("Trwa analiza pliku. To może chwilę zająć.")
        self.status_info_layout.addWidget(self.status_label)

    def add_file_label(self):
        file_label = QLabel()
        file_label.setText("Plik: ")
        self.file_info_layout.addWidget(file_label)

        file_label = QLabel()
        file_label.setText(self.filename)
        self.file_info_layout.addWidget(file_label)

    def open_file_name_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Wybierz plik do analizy", "",
                                                  "Wszystkie pliki (*);;Plik wideo (*.mp4)", options=options)
        if filename:
            return filename


class Runnable(QRunnable):
    def __init__(self, filename, result_destination):
        super().__init__()
        self.filename = filename
        self.result_destination = result_destination

    def run(self):
        result = analyze_Video_without_displaying(self.filename)
        self.result_destination.analysis_finished(result)


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=1, height=1, dpi=100, data=None, title='Title'):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot(data, title)

    def plot(self, data, title):
        ax = self.figure.add_subplot(111)
        ax.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        ax.set_ylim([-5, 5])
        ax.grid(True)
        ax.set_title(title)
        if data is None:
            data = [0]
        ax.plot(data)
        ax.autoscale(axis='x', tight=True)

        self.draw()

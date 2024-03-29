from PyQt5.QtCore import QRunnable, QThreadPool
from PyQt5.QtWidgets import QFileDialog, QLabel, QHBoxLayout

from VideoRecognition.main import *

from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from NLP.main import NLP_analysis
# import matplotlib.pyplot as plt

import numpy as np
import pandas as pd



# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.keras.backend.set_session(tf.Session(config=config))


class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Szacowanie zainteresowania'
        self.filename = ''
        self.layout = QVBoxLayout()
        self.status_label = QLabel()
        self.file_info_layout = QHBoxLayout()
        self.status_info_layout = QHBoxLayout()
        self.result_plot = PlotCanvas()
        self.NLP_plot = PlotCanvas()
        self.init_UI()

    def init_UI(self):
        self.setLayout(self.layout)
        self.setWindowTitle(self.title)
        self.setMinimumSize(800, 1000)
        self.filename = self.open_file_name_dialog()
        self.show()

        self.layout.addLayout(self.file_info_layout)
        self.layout.addLayout(self.status_info_layout)
        self.result_plot.plot(title="Zainteresowanie na podstawie analizy video", data=[0])
        self.NLP_plot.plot(title="Zainteresowanie na podstawie analizy audio (NLP)", data=[0])
        self.layout.addWidget(self.result_plot)
        self.layout.addWidget(self.NLP_plot)

        self.add_file_label()
        self.add_status_label()

        analysis = RunnableVideoRecognition(self.filename, self)
        nlp_analysis = RunnableNLP(self.filename, self)
        QThreadPool.globalInstance().start(nlp_analysis)  # Run analysis in a separate thread
        QThreadPool.globalInstance().start(analysis)  # Run analysis in a separate thread

    def analysis_finished(self, result):
        self.status_label.setText('Analiza ukończona. Wyniki na wykresie poniżej.')
        self.result_plot.plot(result, 'Zainteresowanie na podstawie analizy video')  # Update plot with result

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


class RunnableVideoRecognition(QRunnable):
    def __init__(self, filename, result_destination):
        super().__init__()
        self.filename = filename
        self.result_destination = result_destination

    def run(self):
        # result = analyze_video_with_displaying(self.filename)
        result = analyze_Video_without_displaying(self.filename)
        pd.DataFrame(result).to_csv("results/interest_from_video.csv")
        self.result_destination.analysis_finished(result)


class RunnableNLP(QRunnable):
    def __init__(self, filename, result_destination):
        super().__init__()
        self.filename = filename
        self.result_destination = result_destination

    def run(self):
        timestamps, valance_smoothed, arousal_smoothed, dominance_smoothed = NLP_analysis(self.filename)
        # result = (valance_smoothed + arousal_smoothed) / 2
        result = arousal_smoothed
        pd.DataFrame(result).to_csv("results/interest_from_audio.csv")
        self.result_destination.NLP_plot.plot_NLP(timestamps, result)


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=1, height=1, dpi=100, data=None, title='Title'):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        if data is not None:
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

    def plot_NLP(self, timestamps, data):
        if data is None:
            data = [0]
            timestamps = [0]
        ax = self.figure.add_subplot(111)
        ax.set_ylim([0, 1])
        # ax.set_xlim([timestamps[1], timestamps[-1]])
        ax.grid(True)
        # ax.set_title("arousal smoothed")
        ax.set_ylabel('score')
        ax.set_xlabel('time')
        # ax.plot(timestamps, data)
        ax.plot(data)
        ax.autoscale(axis='x', tight=True)
        self.draw()

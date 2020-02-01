from PyQt5.QtWidgets import QApplication
from VideoRecognition import GUI
import sys

# import matplotlib.pyplot as plt
# from NLP.main import NLP_analysis


def main():
    # timestamps, valance_smoothed, arousal_smoothed, dominance_smoothed = NLP_analysis("data/vid2_short.mp4")
    # plt.figure(figsize=(15, 8))
    # plt.plot(timestamps, dominance_smoothed)
    # plt.title("dominance smoothed")
    # plt.xlabel("time (minutes)")
    # plt.ylabel("score")
    # plt.show()

    app = QApplication(sys.argv)
    ex = GUI.Window()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

from NLP.speech_recognizer import silence_based_conversion
from NLP.vad_tools import get_vad_scores, get_VAD_dictionary
from NLP.graph_tools import moving_average, assign_pseudo_timestamps
import numpy as np
import matplotlib.pyplot as plt


def NLP_analysis(path):
    """
    NLP analysis.
    :return timestamps, valence, arousal, dominance
    """

    # recognize speech and save it to txt file
    timestamps = silence_based_conversion(path)

    # for word, time in timestamps:
    #     print(word, ": ",  time)

    # load speech text
    with open("data/recognized_text.txt", "r") as file:
        text = file.read()

    print('\n' + text + '\n')

    # load VAD dictionary from csv
    vad_dict = get_VAD_dictionary()

    # find words from dictionary in speech text
    vad_scores = get_vad_scores(text, vad_dict)

    vac_scores_with_timestamps = assign_pseudo_timestamps(vad_scores, timestamps)

    # convert to numpy array
    data = np.array(vac_scores_with_timestamps)

    # plot graphs
    timestamps = data[:, 0]
    timestamps = moving_average(timestamps)

    valance = data[:, 1]
    # plt.figure(figsize=(15, 8))
    # plt.plot(valance)
    # plt.title("real valance")
    # plt.show()

    valance_smoothed = moving_average(valance)
    # plt.locator_params(axis="x", tight=False, nbins=20)
    # plt.figure(figsize=(15, 8))
    # plt.plot(timestamps, valance_smoothed)
    # plt.title("valance smoothed")
    # plt.xlabel("time (minutes)")
    # plt.ylabel("score")
    # plt.show()

    arousal = data[:, 2]
    arousal_smoothed = moving_average(arousal)
    # plt.figure(figsize=(15, 8))
    # plt.locator_params(axis="x", tight=False, nbins=20)
    # plt.plot(timestamps, arousal_smoothed)
    # plt.title("arousal smoothed")
    # plt.xlabel("time (minutes)")
    # plt.ylabel("score")
    # plt.show()

    dominance = data[:, 3]
    dominance_smoothed = moving_average(dominance)
    # plt.figure(figsize=(15, 8))
    # plt.plot(timestamps, dominance_smoothed)
    # plt.title("dominance smoothed")
    # plt.xlabel("time (minutes)")
    # plt.ylabel("score")
    # plt.show()

    return timestamps, valance_smoothed, arousal_smoothed, dominance_smoothed

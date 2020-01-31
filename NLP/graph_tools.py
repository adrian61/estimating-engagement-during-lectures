import numpy as np


def moving_average(a, n=10):
    """
    This is just to smooth/average the graph.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def assign_pseudo_timestamps(data, timestamps_list):
    data_with_time = []
    last_word_index = 0
    for i in range(len(timestamps_list)):
        current_word_index = timestamps_list[i][0]
        current_time = timestamps_list[i][1]

        time_diff = current_time - timestamps_list[i - 1][1] if i > 0 else current_time

        num_of_words_in_interval = (current_word_index - last_word_index)
        for j in range(last_word_index, current_word_index):
            timestamp = current_time + (time_diff / num_of_words_in_interval * (j - last_word_index))
            timestamp /= 60
            timestamp = round(timestamp, 2)
            data_with_time.append([timestamp, data[j][0], data[j][1], data[j][2]])

        last_word_index = current_word_index

    return data_with_time


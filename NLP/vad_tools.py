import pandas as pd


def get_VAD_dictionary():
    """
    This function loads dictionary of 20_000 words, each labeled for arousal, valence and dominance
    :return: Pandas dataframe containing dictionary
    """
    filepath = "data/NRC-VAD-Lexicon.txt"
    vad_dict = pd.read_csv(filepath,  names=["valence", "arousal", "dominance"], skiprows=45, sep='\t')
    # print(vad_dict.head(5))
    print("Number of words in dictionary: ", len(vad_dict))
    return vad_dict


def get_vad_scores(text, vad_dict):
    """
    Tries to find words from VAD dictionary in text String.
    :param text: Text.
    :param vad_dict: Pandas dataframe.
    :return: Dictionary of words from text file, containing VAD scores (valance, arousal, dominance) for each of them
    """
    scores = []
    text = text.replace('\n', ' ')
    text = text.lower()
    words = text.split(' ')
    found_counter = 0
    for w in words:
        try:
            value = vad_dict.loc[w]
            scores.append([w, round(value[0], 2), round(value[1], 2), round(value[2], 2)])
            found_counter += 1
        except KeyError:
            pass
            scores.append([w, 0.5, 0.5, 0.5])

    # print('\n'.join(str(element) for element in scores))
    vad = [score[1:] for score in scores]
    print("number of words in recognised text: ", len(words))
    print("number of words found: ", found_counter)
    return vad

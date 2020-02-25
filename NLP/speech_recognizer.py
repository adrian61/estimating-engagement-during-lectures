import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent


def silence_based_conversion(path):
    """
    A function that splits the audio file into chunks on each silence interval and applies speech recognition.
    :return Timestamps containing word numbers and the time each of them appeared.
    """

    # open the audio file stored in the local system as a wav file.
    print("openning audio file...")
    # audio = AudioSegment.from_file(path, format="mp4")
    audio = AudioSegment.from_file(path)
    # audio = AudioSegment.from_wav(path)
    print("audio file opened")

    # open a file where we will concatenate and store the recognized text
    fh = open("data/recognized_text.txt", "w")

    print("splitting audio file...")

    # split track where silence is 1 seconds or more and get chunks

    min_silence_len = 1000
    silence_thresh = -45

    not_silence_ranges = detect_nonsilent(audio, min_silence_len, silence_thresh, 1)

    chunks = []
    for i in range(len(not_silence_ranges)):
        start_i, end_i = not_silence_ranges[i][0], not_silence_ranges[i][1]

        if i < len(not_silence_ranges) - 1:
            next_start_i = not_silence_ranges[i + 1][0]
            end_i = next_start_i - 100

        if start_i > 100:
            start_i -= 100

        chunks.append(audio[start_i:end_i])

    print("audio file splitted, ", end='')
    print("number of chunks: ", len(chunks), "\n")

    # create a directory to store the audio chunks.
    try:
        os.mkdir('audio_chunks')
    except(FileExistsError):
        pass

    word_count = 0
    current_time = 0
    timestamps = []  # contains word_count : current time

    # process each chunk
    i = 0
    for chunk in chunks:

        # Create 0.5 seconds silence chunk
        chunk_silent = AudioSegment.silent(duration=500)

        # add 0.5 sec silence to beginning and
        # end of audio chunk. This is done so that
        # it doesn't seem abruptly sliced.
        audio_chunk = chunk_silent + chunk + chunk_silent

        # export audio chunk and save it in
        # the current directory.
        # print("saving chunk{0}.wav".format(i))

        # the name of the newly created chunk
        filename = "data/audio_chunks/chunk{0}.wav".format(i)

        # specify the bitrate to be 192 k
        audio_chunk.export(filename, bitrate='192k', format="wav")

        print("Processing chunk " + str(i) + ", length: " + str(round(chunk.duration_seconds, 2)) + " (s)")

        # create a speech recognition object
        r = sr.Recognizer()

        current_time += chunk.duration_seconds

        # recognize the chunk
        with sr.AudioFile(filename) as source:
            audio_listened = r.listen(source)

        try:
            # try converting it to text
            rec = r.recognize_google(audio_listened)

            num_of_words = len(rec.split(' '))
            word_count += num_of_words
            timestamps.append((word_count, current_time))

            # write the output to the file.
            fh.write(rec + '\n')

            # catch any errors.
        except sr.UnknownValueError:
            print("Could not understand audio")

        except sr.RequestError:
            print("Could not request results. check your internet connection")

        i += 1

    fh.close()
    print("\n" + "all chunks processed" + "\n")

    return timestamps

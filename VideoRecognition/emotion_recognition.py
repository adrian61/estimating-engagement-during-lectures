import dlib
import numpy as np
from imutils import face_utils
from scipy.ndimage import zoom


def emotion_recognition(rects, gray, model, predictor_landmarks, shape_x=48, shape_y=48):
    for (i, rect) in enumerate(rects):

        shape = predictor_landmarks(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Identify face coordinates
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face = gray[y:y + h, x:x + w]

        # Zoom on extracted face
        if face.shape[0] == 0 or face.shape[1] == 0:
            continue
        face = zoom(face, (shape_x / face.shape[0], shape_y / face.shape[1]))

        # Cast type float
        face = face.astype(np.float32)

        # Scale
        face /= float(face.max())
        face = np.reshape(face.flatten(), (1, 48, 48, 1))

        # Make Prediction
        prediction = model.predict(face)
        prediction_result = np.argmax(prediction)
        # 0 - Angry
        # 1 - Disgust
        # 2 - Fear
        # 3 - Happy
        # 4 - Sad
        # 5 - Surprise
        # else Neutral
        print("Total score for {n} is {s}".format(n=prediction_result, s=valueOfEmotion(prediction_result)))
    print("Next frame")


def valueOfEmotion(number):
    SumOfValue = 0
    # 3d affect space
    # first value - Arousal
    # second value - Valence
    # third value - Stance
    Angry = {5, -5, -4}
    Disgust = {2.5, -4.5, 0}
    Fear = {4, -5, 3}
    Happy = {4, 5, 5}
    Sad = {-4, -5, -2}
    Surprise = {5, 0, 0}
    Neutral = {0, 0, 0}
    if number == 0:
        for x in Angry:
            SumOfValue += x
        return SumOfValue / 3
    if number == 1:
        for x in Disgust:
            SumOfValue += x
        return SumOfValue / 3
    if number == 2:
        for x in Fear:
            SumOfValue += x
        return SumOfValue / 3
    if number == 3:
        for x in Happy:
            SumOfValue += x
        return SumOfValue / 3
    if number == 4:
        for x in Sad:
            SumOfValue += x
        return SumOfValue / 3
    if number == 5:
        for x in Surprise:
            SumOfValue += x
        return SumOfValue / 3
    else:
        for x in Neutral:
            SumOfValue += x
        return SumOfValue / 3

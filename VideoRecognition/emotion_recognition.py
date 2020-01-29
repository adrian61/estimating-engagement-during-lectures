import dlib
import numpy as np
from imutils import face_utils
from scipy.ndimage import zoom


def emotion_recognition_f(rects, gray, model, predictor_landmarks, shape_x=48, shape_y=48):
    frame_values= []
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
        frame_values.append(valueOfEmotion(prediction[0]))
        #print("Total score for {n} is {s}".format(n=prediction_result, s=valueOfEmotion(prediction[0])))
    #print("Next frame")
    return sum(frame_values) / len(frame_values)


def valueOfEmotion(prediction):
    SumOfValue = 0
    temp = 0
    # 3d affect space
    # first value - Arousal
    # second value - Valence
    # third value - Stance
    Emotions = [[5, -5, -4], [2.5, -4.5, 0], [4, -5, 3], [4, 5, 5], [-4, -5, -2], [5, 0, 0], [0, 0, 0]]
    # 0 - Angry
    # 1 - Disgust
    # 2 - Fear
    # 3 - Happy
    # 4 - Sad
    # 5 - Surprise
    # 6 - Neutral
    for i in range(0, len(Emotions)):
        for x in Emotions[i]:
            temp += x
        SumOfValue += temp / 3 * prediction[i]
    return round(SumOfValue, 3)


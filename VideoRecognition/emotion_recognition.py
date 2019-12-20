import numpy as np
from imutils import face_utils
from scipy.ndimage import zoom
import cv2

from VideoRecognition.face_recognition import eye_aspect_ratio


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
        # print(prediction_result)
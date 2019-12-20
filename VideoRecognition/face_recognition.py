import dlib
from scipy.spatial import distance
from tensorflow.keras.models import load_model as lm


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def load_model():
    model = lm('Models/video.h5')
    return model


def load_face_detect():
    face_detect = dlib.get_frontal_face_detector()
    return face_detect


def load_prediction():
    predictor_landmarks = dlib.shape_predictor("Models/face_landmarks.dat")
    return predictor_landmarks


def load_utilities_to_face_recognition():
    model = load_model()
    face_detect = load_face_detect()
    prediction = load_prediction()
    return model, face_detect, prediction

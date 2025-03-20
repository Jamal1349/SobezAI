import cv2
from best_exp import emotion_pred
from best_ges import gesture_score
from best_nitec import nitec_model
import torch
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python import vision

def video_proc(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    results_nitec = []
    results_emotion = []
    results_gesture = []
    model_path = "gesture_recognizer.task"
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = vision.GestureRecognizer
    GestureRecognizerOptions = vision.GestureRecognizerOptions
    VisionRunningMode = vision.RunningMode
    options = GestureRecognizerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )
    recognizer = GestureRecognizer.create_from_options(options)
    while cap.isOpened:
        ret, frame = cap.read()
        if ret:
            result_emotion = emotion_pred(frame)
            results_emotion.append(result_emotion)
            result_nitec = nitec_model(frame)
            results_nitec.append(result_nitec)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mpframe = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            result = recognizer.recognize(mpframe)
            if len(result.handedness) > 0:
                result_gesture = gesture_score(frame)
                results_gesture.append(result_gesture)
        else:
            break
    cap.release()
    return results_gesture, results_emotion, results_nitec
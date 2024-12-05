import torch
from nitec import NITEC_Classifier, visualize
import cv2
import pathlib
def nitec_model(input_video_path, output_video_path):
    CWD = pathlib.Path.cwd()
    nitec_pipeline = NITEC_Classifier(
        weights=CWD / 'nitec_rs18_e20.pth',
        device=torch.device('cpu')
    )
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    confidence = 0.5
    results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with torch.no_grad():
            frame_results = nitec_pipeline.predict(frame)
            results.append(frame_results)
    cap.release()
    return results


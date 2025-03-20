import torch
from nitec import NITEC_Classifier, visualize
import cv2
import pathlib


def nitec_model(frame):
    CWD = pathlib.Path.cwd()
    nitec_pipeline = NITEC_Classifier(
        weights=CWD / 'nitec_rs18_e20.pth',
        device=torch.device('cpu')
    )
    with torch.no_grad():
        frame_results = nitec_pipeline.predict(frame)
    #return frame_results
    return frame_results.results[0]

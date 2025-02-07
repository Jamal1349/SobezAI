import cv2 as cv
from torchvision import transforms
from PIL import Image
import torch
import mediapipe as mp
import torch.nn as nn
import torchvision.models as models
from mediapipe.tasks.python import vision

def gesture_score(video_path_):
    gesture_score = []
    model = models.resnet50(pretrained=True)
    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('resnet50_gesture_model_2.pth'))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = vision.GestureRecognizer
    GestureRecognizerOptions = vision.GestureRecognizerOptions
    VisionRunningMode = vision.RunningMode

    model_path = "gesture_recognizer.task"
    video_path = video_path_
    cap = cv.VideoCapture(video_path)

    # Создаем экземпляр распознавателя жестов
    options = GestureRecognizerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )

    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mpframe = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            result = recognizer.recognize(mpframe)

            if len(result.handedness) > 0:
                # Применяем трансформации к кадру (например, изменение размера)
                img_pil = Image.fromarray(frame)
                transformed_frame = transform(img_pil)
                batch_t = torch.unsqueeze(transformed_frame, 0)
                with torch.no_grad():
                    out = model(batch_t)
                    score = torch.argmax(out, dim=1)
                    score_number = int(score)
                    gesture_score.append(score_number)

    cap.release()
    return gesture_score
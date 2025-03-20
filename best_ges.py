import cv2
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models

def gesture_score(frame):
    gesture_score = []
    model = models.resnet50(pretrained=True)
    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('resnet50_gesture_model_2.pth', weights_only=True))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_pil = Image.fromarray(frame)
    transformed_frame = transform(img_pil)
    batch_t = torch.unsqueeze(transformed_frame, 0)
    with torch.no_grad():
        out = model(batch_t)
        score = torch.argmax(out, dim=1)
        score_number = int(score)
        gesture_score.append(score_number)

    return gesture_score

# frame = cv2.imread('D:/sobez/video_for_data/images_1/frame_video_3.mkv_1.png')
# print(gesture_score(frame))
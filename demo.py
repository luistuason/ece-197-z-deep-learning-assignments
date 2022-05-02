import os
import time
import numpy as np
import torch
import cv2
import imutils

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import *
import lib.transforms as T

def load_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def infer_frame(frame, og_frame, model):
    classes = { 0: 'background', 1: 'Summit Drinking Water', 2: 'Coca-Cola', 3: 'Pineapple Juice' }

    with torch.inference_mode():
        detected = model(frame)[0]
        
        for idx, box in enumerate(detected['boxes']):
            confidence = detected['scores'][idx]

            if confidence > 0.8:
                label_idx = int(detected['labels'][idx])
                box = box.detach().cpu().numpy()
                xmin, ymin, xmax, ymax = box.astype('int')

                label = classes[label_idx]
                if label_idx == 1:
                    color = (255, 0, 0)
                elif label_idx == 2:
                    color = (0, 0 ,255)
                elif label_idx == 3:
                    color = (0, 255, 0)
                cv2.rectangle(og_frame, (xmin, ymin), (xmax,ymax), color, 2)
                cv2.putText(og_frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return og_frame

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cwd = os.path.abspath(os.path.dirname(__file__))
    trained_model_path = os.path.abspath(os.path.join(cwd, 'export/trained_model.pth'))

    num_classes = 4

    #Load model
    model = load_model(num_classes)
    ckpt = torch.load(trained_model_path, map_location=device)
    model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    
    #Start videeo capture
    if (os.name == 'nt'):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    elif (os.name == 'posix'):
        cap = cv2.VideoCapture(0)

    # Get height and width
    cap_width = cap.get(3)
    cap_height = cap.get(4)

    while (True):
        ret, frame = cap.read()

        frame = imutils.resize(frame, width=640, height=480)
        og = frame.copy()

        # Convert frame to tensor
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # cv2 reads images as BGR, so convert to RGB first
        frame = frame.transpose((2, 0, 1))                      # H W C -> C H W
        frame = np.expand_dims(frame, axis=0)                   # add dim for batch size
        frame = frame / 255                                     # normalize [0,1]
        frame = torch.tensor(frame, dtype=torch.float32, device=device)

        frame_show = infer_frame(frame, og, model)

        # frame_show = cv2.flip(frame_show, 1)
        cv2.imshow('frame', frame_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()

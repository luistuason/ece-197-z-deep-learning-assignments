import os
import torch
from PIL import Image
import einops
import torchvision

from model import *
from dataset import *


def infer():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device selected: ', device)

    cwd = os.path.abspath(os.path.dirname(__file__))
    trained_model_path = os.path.abspath(os.path.join(cwd, './export/trained_model.pth'))

    num_classes = 4
    classes = {
        0: 'background',
        1: 'Summit Drinking Water',
        2: 'Coca-Cola',
        3: 'Pineapple Juice'
    }
    
    # filename = '0010010.jpg'
    filename = 'test.jpg'
    image_filepath = os.path.abspath(os.path.join(cwd, './data/drinks/' + filename))

    image = Image.open(image_filepath)
    image = T.ToTensor()(image)[0]
    image = einops.rearrange(image, 'c h w -> () c h w')
    image = image.cuda()


    # get the model using our helper function
    model = load_model(num_classes)
    ckpt = torch.load(trained_model_path)
    model.load_state_dict(ckpt)

    # move model to the right device
    model.to(device)
    model.eval()

    with torch.inference_mode():
        detected = model(image)[0]
        detected= {'boxes': detected['boxes'][detected['scores'] > 0.8],
                     'labels': detected['labels'][detected['scores'] > 0.8],
                     'scores': detected['scores'][detected['scores'] > 0.8]}

        colors = [(0, 255, 0), (0, 0, 255), (255, 00, 0), (0, 255, 0)]

        image = einops.rearrange(image, '() c h w -> c h w')

        labels = detected['labels'].detach().cpu().numpy()
        labels = [classes[label] for label in labels]

        # draw bounding boxes
        image = (image * 255).type(torch.uint8)
        image = torchvision.utils.draw_bounding_boxes(image, detected['boxes'], width=3, colors=colors, labels=labels)
        image = torchvision.transforms.ToPILImage()(image)
        image.show()
        image.save('output_'+filename)

    
if __name__ == "__main__":
    infer()

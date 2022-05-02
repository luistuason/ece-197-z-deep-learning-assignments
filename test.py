import os
import torch

from model import *
from dataset import *
from lib.engine import evaluate
import lib.utils


def test():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device selected: ', device)

    cwd = os.path.abspath(os.path.dirname(__file__))
    labels_test_path = os.path.abspath(os.path.join(cwd, 'data/drinks/labels_test.csv'))
    trained_model_path = os.path.abspath(os.path.join(cwd, 'export/trained_model.pth'))

    # declare the test dataset and dataloader
    num_classes = 4
    dataset_test = DrinksDataset(root='data/drinks/', csv_file=labels_test_path, transforms=get_transform(train=False))

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=lib.utils.collate_fn)

    # declare the model using helper function
    model = load_model(num_classes)
    ckpt = torch.load(trained_model_path, map_location=device)
    model.load_state_dict(ckpt)

    # move model to the right device
    model.to(device)

    evaluate(model, data_loader_test, device=device)
    
if __name__ == "__main__":
    test()

import os
from datetime import datetime
import torch
from model import *
from dataset import *
from lib.engine import evaluate
import lib.utils


def test():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cwd = os.path.abspath(os.path.dirname(__file__))
    labels_test_path = os.path.abspath(os.path.join(cwd, 'data/drinks/labels_test.csv'))
    trained_model_path = os.path.abspath(os.path.join(cwd, 'export/trained_model.pth'))

    num_classes = 4
    # use our dataset and defined transformations
    dataset_test = DrinksDataset(root='data/drinks/', csv_file=labels_test_path, transforms=get_transform(train=False))

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=lib.utils.collate_fn)

    # get the model using our helper function
    model = load_model(num_classes)
    ckpt = torch.load(trained_model_path, map_location=device)
    model.load_state_dict(ckpt)

    # move model to the right device
    model.to(device)

    start_time = datetime.now()
    evaluate(model, data_loader_test, device=device)

    elapsed_time = datetime.now() - start_time
    
if __name__ == "__main__":
    test()

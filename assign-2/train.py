import os
from datetime import datetime
import torch

from model import *
from dataset import *
from utility import load_dataset
from lib.engine import train_one_epoch, evaluate
import lib.utils

cwd = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.abspath(os.path.join(cwd, './data'))
labels_train_path = os.path.abspath(os.path.join(cwd, './data/drinks/labels_train.csv'))
labels_test_path = os.path.abspath(os.path.join(cwd, './data/drinks/labels_test.csv'))


def train():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device selected: ', device)

    # declare the dataset
    num_classes = 4
    dataset = DrinksDataset(root='./data/drinks/', csv_file=labels_train_path, transforms=get_transform(train=True))
    dataset_test = DrinksDataset(root='./data/drinks/', csv_file=labels_test_path, transforms=get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=lib.utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=lib.utils.collate_fn)

    # declare the model using helper function
    model = load_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0025,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 10
    epoch_number = 0

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = './export/model_{}_{}.pth'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)
        epoch_number+=1
    
if __name__ == "__main__":
    load_dataset(data_path)
    train()

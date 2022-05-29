import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import KWSDataModule
from model import KWSModel

CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    
# make a dictionary from CLASSES to integers
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def get_args():
    parser = ArgumentParser()
    # model training hyperparameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--max-epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    # where dataset will be stored
    parser.add_argument("--path", type=str, default="data/speech_commands/")

    # 35 keywords + silence + unknown
    parser.add_argument("--num-classes", type=int, default=37)
   
    # mel spectrogram parameters
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)

    # 16-bit fp model to reduce the size
    parser.add_argument("--precision", default=16)
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1)
    parser.add_argument("--num-workers", type=int, default=48)

    args = parser.parse_args("")
    return args

def train():
    args = get_args()
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    model = KWSModel(num_classes=args.num_classes, epochs=args.max_epochs, lr=args.lr)
    print(model)
    datamodule = KWSDataModule(batch_size=args.batch_size, num_workers=args.num_workers,
                               path=args.path, n_fft=args.n_fft, n_mels=args.n_mels,
                               win_length=args.win_length, hop_length=args.hop_length,
                               class_dict=CLASS_TO_IDX)
    datamodule.setup()

    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.path, "checkpoints"),
        filename="resnet18-kws-best-acc",
        save_top_k=1,
        verbose=True,
        monitor='test_acc',
        mode='max',
    )
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
    trainer = Trainer(accelerator=args.accelerator,
                      devices=args.devices,
                      precision=args.precision,
                      max_epochs=args.max_epochs,
                      logger=wandb_logger if not args.no_wandb else None,
                      callbacks=[model_checkpoint, WandbCallback() if not args.no_wandb else None])
    model.hparams.sample_rate = datamodule.sample_rate
    model.hparams.idx_to_class = idx_to_class
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    #trainer.save_checkpoint('../mnist/checkpoint.ckpt')

if __name__ == "__main__":
    train()
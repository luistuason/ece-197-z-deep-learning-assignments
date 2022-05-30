# ECE 197 Z Deep Learning - Assignment 3: Keyword Spotting using Transformer
### Build a transformer-based keyword spotting (KWS) system
--------------------------------------------------------------------------------

**Philip Luis D. Tuason III**

**2018-08149**

**BS Electronics Engineering**

*Electrical and Electronics Engineering Institute (EEEI),*

*College of Engineering,*

*University of the Philippines Diliman*

--------------------------------------------------------------------------------

## How to run

It is assumed that CUDA-enabled PyTorch is installed in your environment via conda. Other prerequisites can be found in [requirements.txt](requirements.txt)
```
pip install -r requirements.txt
```

### Training

After satisfying all the prerequisites, you may train the model through
```
python train.py
```
The dataset should automatically be downloaded as you run the script.

The default patch size for the Transformer is set to 16. This resulted in a test accuracy of **92.87%**. You may download the checkpoint for the pre-trained model [here](https://github.com/luistuason/ece-197-z-deep-learning-assignments/releases/download/v2.00/transformer-kws-best-acc.pt)

### GUI-based Demo

You may try out the model through the GUI-based application that recognizes keywords from your PC's microphone. Running the script also automatically downloads the pre-trained model attached above.
```
python kws-infer.py
```

## Video Demo

You may watch a video demo of the GUI-based application being used [here]().
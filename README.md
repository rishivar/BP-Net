# BP-Net
The repository contains code for the paper titled, "BP-Net: Efficient Deep Learning for Continuous Arterial Blood Pressure Estimation using Photoplethysmogram", which has been **accepted** at the ([20th IEEE INTERNATIONAL CONFERENCE ON MACHINE LEARNING AND APPLICATIONS](https://www.icmla-conference.org/icmla21/)). 

## Files description
- [dataloader.py](preprocess.py) <br>
    Contains required dataloader files <br>
- [model.py](resnetv2.py) <br>
    Contains model architecture <br>
- [SSL.py](resnetv2.py) <br>
    Training script for Self-Supervised Learning <br>
- [train.py](train.py) <br>
    Driver code for training the model <br>
- [test.py](eval.py) <br>
    Test script for model inference <br>
- [eval.py](eval.py) <br>
    Evaluation script for evaluating a particular model checkpoint <br>
- [edge_port.py](eval.py) <br>
    Script for porting torch model to ONNX format <br>
- [edge_eval.py](eval.py) <br>
    Evaluation script for evaluating on device with ONNX Runtime library <br>

## Note
- Download the required data and model files from [drive](https://drive.google.com/drive/folders/1TZH-nuH9BTIav6txioBtWdh9Fl6eCVZ_?usp=sharing).
- The data files are obtained from ([PPG2ABP](https://github.com/nibtehaz/PPG2ABP)) and best fold data is available at [Data](/data).
- After 10 fold cross validation training, the best model is available at [model.pt](model/model.pt).

## Publication Link
http://arxiv.org/abs/2111.14558

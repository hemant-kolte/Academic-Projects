# Automatic Covid-19 detection and severity prediction using ensemble learning with CNN

This experiment follows the code base downloaded from referenced from the published paper [Automatic COVID-19 detection from X-ray images using ensemble learning with convolutional neural network](https://link.springer.com/article/10.1007/s10044-021-00970-4/). The report corresponding to our experimentation and project can be found [here]()

## File Structure

- The code has been implemented in the Google Colab environment. It can be accessed [here](https://drive.google.com/drive/u/2/folders/1XZBEW7OoTlb2rcvatXr1Rmq0VPL8kY22)
- The pre-trained models are available in the [models](https://drive.google.com/drive/folders/1szPez-1bPcrtKlzW2S29stjVmDaROFTb?usp=sharing) folder
- The [Dataset](https://drive.google.com/drive/folders/17zX8e_kh46sAVa_IN4nbFLRcb1HBhrSG?usp=sharing) contains the following directory structure:
--**output**: contains the transformed numpy file
--**test**: contains the test images with the split of COVID and non-COVID images, used for testing the trained model
--**train**: contains the test images with the split of COVID and non-COVID images, used for training the various models
- The [Severity](https://drive.google.com/drive/folders/1nrKVGkWeOLaRaoiQpi6ASacduTIfYUqX?usp=sharing) folder contains the following directory structure:
--**final.ipynb**: It splits the sample data into 70% test, 20% validate and 10% train
--**OPCV_Processing.ipynb**: used to threshold to get the severity of COVID-19 with 30% lower and higher 
- [Model_with_extended_dataset](https://drive.google.com/drive/folders/11nhn4ctpNEtFMo9PHpAqCziunDNCsGKs?usp=sharing) contains the newly fine-tuned models
- [DataPreparation.ipynb](https://colab.research.google.com/drive/12dTFL9QwRZU1b--068vQfYbwJFcKTsyq?usp=sharing): Prepare the data by performing a transformation into numpy arrays and segregating it into multiple classes and later augmenting it.
- [Ensembling.ipynb](https://colab.research.google.com/drive/1QBw66RU8yb47wQntMzu5unjbFLyG7j-1?usp=sharing): Contains the code implementation for performing the ensembling of combinations of models in 3
- [Model_train](https://colab.research.google.com/drive/11hTYdz1gtxFl7BLJ4hg8d9NA22473eSG?usp=sharing): Takes input from the **Models** folder and trains the models **inception**, **densenet**, **resnet** and writes to the **Model_with_extended_dataset** folder
- [Model.ipynb](https://colab.research.google.com/drive/1aqhcxI9hUK3TuQJpvfXeER2Of1FWhhg6?usp=sharing): Takes models from **Models** folder and trains the Model on the **train** dataset for all the models listed in the Models listed in the **Models** folder
- [Performance.ipynb](https://colab.research.google.com/drive/1gh4d1Yq_Yne2SARIsfst_KvnZvuh67qF?usp=sharing): Used to evaluate the model on the accuracies and the F1 score along with the confusion matrices. It also provides the weights on every model

## Execution

- All the code implementation has been done as an ipynb file
- ipynb files can be run in Google Colab environment when logged in through Google or can also be run using [jupyterlab] or notebook environments
- Google Colab comes with the pre-requisites pre-installed

## Prerequisite libraries

- opencv2 >= 3.4
- tensorflow >= 2.2.0
- pytorch >= 1.5.0    


> The goal of our project would be test the ensembling
> of the existing SOTA models in ML and Vision to 
> achieve higher accuracies and F1 scores.
> Also, the model being trained on multiple datasets
> has been generalized for various X-Ray image patterns


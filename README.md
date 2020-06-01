# crime-predictor

Bilkent University CS 559 Deep Learning Project.\
Author: Oguzhan Karaahmetoglu

## Abstract

Aim of this project is to present a spatiotemporal prediction model, which predicts criminal events in Chicago City. 
Approach is based on the Soft Decision Tree mechanism and a set of ConvLSTM models. For crime prediction, Chicago
City crime records were used, which are available as records of criminal activities occurred between 2001-2020. This
raw format is converted to a spatiotemporal tensor and used in the prediction along with the weather data for the 
same time and space.

## Directory Structure

* Main flow is initiated from "main.py" script. This script does not take any arguments, all arguments are read
  from "config.py". After the parameters are read, data will be preprocessed and model will be initialized. Finally,
  model is trained for several epochs on the training set.
  
* All model hyperparameters and data preprocessing arguments are contained in "config.py". These parameters will
  be read by "main.py".
  
* "transformer.py" contains the implementation for the Crime and Weather data transformers, which are used to convert
  the raw crime and weather records to data tensors.
  
* "validation.py" has a single evaluation function, which is the AP score computation. Crime labels are converted to 
  binary values for evaluation.
  
* "batch_generator.py" has the batch generation object implementation, which returns the training, validation and 
  test samples.
  
* Approach is compared with two models (a baseline and the SOTA), which are implemented in:
    * ConvLSTM: Standard ConvLSTM model, which was used in other spatiotemporal prediction problems such as weather
    prediction. This is the baseline model used in the comparison. ("./models/convlstm.py")
    
    * GraphLSTM: SOTA model. Previously applied on the Chicago City Crime data (though not as in our setup). Graph-based
    model that generates predictions for each spatial cell by processing inputs from other cells connected by the graph.
    Each edge in the graph is modeled with an LSTM model. ("./models/graphlstm.py"")
    
    * TreeConvLSTM: Our approach. Based on a soft decision tree that partitions the spatial region and trains an
    individual ConvLSTM model in each one. ("./models/tree_convlstm.py")
    
    
## How to use?

First the configuration file should be tuned. Configuration file consists of different sections of hyperparameters
for different models and data preprocessing methods. After setting up the configuration file with the desired parameters,
the model could be trained with the following command:

```
python main.py
```

After the above command is executed, model loss will be reported after each epoch. Moreover, PR curve on the validation
set will also be plotted along with the AP score. When the training is finished, test score will be computed along with
the PR curve.
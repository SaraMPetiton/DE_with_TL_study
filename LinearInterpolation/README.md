# LinearInterpolation

In this repository, you'll find the scripts necessary to run the linear interpolations of our paper "how and why does deep ensemble coupled with transfer learning increase performance in bipolar disorder and schizophrenia classification?".

This folder contains the code for the linear interpolations we used to better understand how and why transfer learning improves performance in bipolar disorder (BD) and schizophrenia (SCZ) classification tasks.

The model weights used as pre-training for the transfer learning model can be downloaded from here : https://github.com/Duplums/yAwareContrastiveLearning/tree/main 

We are using the linear interpolation method proposed in https://arxiv.org/abs/2008.11687, and are applying it to our 3D anatomical MRI BD and SCZ datasets.

In these scripts you will find the same notations as in our paper :
RI-DL corresponds to models trained from randomly-initialized weights.
TL corresponds to models trained using pre-trained weights as initialization.
The type of model used for the classification tasks is densenet121.

In these scripts, we :
Linearly interpolate weights from one model to another.
In the folders of this repo and in the scripts, "checkpoint0" and "checkpoint1" refer to the model from which we are interpolating to the one to which we are interpolating. The reasoning behind this is that the weights of model at "checkpoint0" are equivalent to having the linear interpolation coefficient lambda equal to 0, whereas the weights of model at "checkpoint1" are the same for lambda equal to 1.
The values of the linear interpolation coefficients along the linear interpolation path take values between 0 and 1.
You can choose how many uniformly distributed values (between 0 and 1) you want in main_interpolation.py with variable 'nb_lambda'.

The linear interpolations we run here are:
- 2 different TL models at their last epoch of training.
- 2 different RI-DL models at their last epoch of training.
- a TL model at its last epoch of training and the same one at the epoch of training for which its ROC-AUC evaluated on the test set is highest.
- the RI-DL model R1 at its last epoch of training and the same one at the epoch of training for which its ROC-AUC evaluated on the test set is highest.

These last two require having saved your models saved under pth format every n number of epochs during training, and having tested these models on your testing set for each epoch checkpoint at which they have been saved.

The same pre-trained weights are used for all TL models.

The plots created in the interpolation_plots.py script plot the ROC-AUC values against the linear interpolation coefficients.
This can be easily modified to use the Balanced Accuracy instead.

## pre-saved results

You can find the plots we generated in the folder plots_interpolation, and you can find dictionaries containing the metrics (ROC-AUC and Balanced Accuracy) saved for 30 linear interpolation coefficients using our models.
The models we used can be found in saved_models.

## how to run the code
We provide the models we used for this study in the folder 'saved_models'.
However, we used data that can be fetched using the scz_bd_datasets.py script, which is necessary to run the testing along the linear interpolationg path.
If you do not have access to the same datasets as the ones we used here (the BD and SCZ datasets we use here are described at https://github.com/Duplums/SMLvsDL), you might have to modify the get_dataloader function of the LinearInterp.py script to fit your data.
The 'root' argument of main_interpolation.py is only necessary to access our dataset, so you might need to remove it or change it to the path where you can find your data on your local device.
Once you apply these slight modifications to account for your own datasets, you can use the main_interpolation.py script as it is to perform the linear interpolations using our TL and RI-DL models.

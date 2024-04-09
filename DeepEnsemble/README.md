# DeepEnsemble

In this repository, you'll find the scripts necessary to run the deep ensembling of our paper "how and why does deep ensemble coupled with transfer learning increase performance in bipolar disorder and schizophrenia classification?" (ISBI 2024).

This folder contains the code performing the deep ensemble method through averaging of the multiple models' outputs.

In "models_metrics", you will find pickle files containing the true and predicted probabilities outputed from the 90 deep learning models we have used in this study. 

To achieve the deep ensembling, a bootstrapping method is used to sample the predicted labels before computing the roc auc or the balanced accuracy using the scripts in DeepEnsemble.py. 

To run deep ensembling, the main script 'main_DE.py' only requires a flag for "--pb", equal to "bipolar" or "scz", and by default computes the ensemble roc auc using an external testing set containing only sites that have not been seen during the training of the models, and displays the values in a plot with a curve for both RI-DL and TL models separately. 
You can also choose to save the plot using the flag "--save_or_show_plot *save*". The plot will be saved in the "saved_plots" folder of the "DeepEnsemble" folder. 

## how to run the code

This whole folder can run using the metrics we provide in "models_metrics" :
- RIDL_BD : metrics for the 90 models classifying bipolar disorder using randomly initialized weights, for both internal and external test sets (respectively "Test_Intra_{densenet121}_{vbm}_{bipolar}_fold0_epoch199.pkl" and "Test_{densenet121}_{vbm}_{bipolar}_fold0_epoch199.pkl"). 
- RIDL_SCZ : same thing for schizophrenia classification.
- TL_BD : metrics for the 90 models classifying bipolar disorder using randomly initialized weights, for both internal and external test sets (respectively "Test_Intra_{densenet121}_{vbm}_{bipolar}_finetuned_fold0_epoch199.pkl" and "Test_{densenet121}_{vbm}_{bipolar}_finetuned_fold0_epoch199.pkl")
- TL_SCZ : same thing for schizophrenia classification.

If you choose to run these scripts using your own data, you can modify the files contained in RIDL_BD, RIDL_SCZ, TL_BD and TL_SCZ to contain your own metrics. You just need to make sure that they are saved in a dictionary with the same format as in any of our *.pkl files. 

**saved_plots** 
This folder contains our plots for bipolar disorder and schizophrenia classification.
The values displayed are the standard deviations for each grouping size.
If the values *slightly* vary from the plot in our paper, it is because we re-ran the script for 1e5 bootstrapping samplings, and as there is still some stochasticity to it, the std and mean values can change by about 0.1%.
The trend of the plot will always stay the same, as long as the number of samplings isn't too low (the variability of outputs increases inversely proportionally from the number of samplings).

**DeepEnsemble_scz_100000_samplings_04_04_2024_roc_auc_external_test.pkl** and **DeepEnsemble_bipolar_100000_samplings_04_04_2024_roc_auc_external_test.pkl** 
These pickle files contain the computed deep ensembling values resulting in the plots in "saved_plots".
They contain a dictionary of the results of the *runDeepEnsemble* function of main_DE.py and can be saved by adding the flag "--save_metrics" when running the main_DE.py script.
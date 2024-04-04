from DeepEnsemble import get_ytrue, create_list_metric , get_metric_bootstrap, plot
import numpy as np
import argparse, pickle, os
from datetime import datetime

def get_current_date_string():
    """ 
    Output : 
        formatted_date : (str) returns the current date in format dd_mm_YYYY
        used to prevent overwriting files created on different days
        by including this string in the name of said files before saving them
    """
    now = datetime.now()
    formatted_date = now.strftime("%d_%m_%Y")
    return formatted_date


def runDeepEnsemble(args):
    """ 
    Parameters :
        args : dictionary of keyboard arguments such as the classification type (args.pb = "bipolar" or "scz"),
        whether to save the results in a pkl file, the number of samplings (args.nb_samplings), etc.
    Aim : 
        return the metric (roc auc or balanced accuracy) value for RI-DL and TL for either bipolar disorder or schizophrenia classification,
        as well as the uncertainty metrics (standard deviations)
    Output : 
        params_plot : (dict) containing the metrics and standard deviations for RI-DL and TL
        for the classification task. This dictionary serves as the input of the function plotting the results (in plot_DE.py). 
    """
    list_models = ["model" + str(i) for i in range(1, args.nb_models+1)]

    # 1) DEFINE PATHS ######################################################################################################################################

    if args.pb == "bipolar":
        path_RIDL = os.path.join(os.getcwd(),"models_metrics/RIDL_BD") 
        path_TL = os.path.join(os.getcwd(),"models_metrics/TL_BD") 

    if args.pb == "scz":
        path_RIDL = os.path.join(os.getcwd(),"models_metrics/RIDL_SCZ")
        path_TL = os.path.join(os.getcwd(),"models_metrics/TL_SCZ") 


    # 2) GET TRUE LABELS ####################################################################################################################################
    if args.internal :
        ytrue_RIDL = get_ytrue(os.path.join(path_RIDL, "model1"), False, args.pb, True)
    else :
        ytrue_RIDL = get_ytrue(os.path.join(path_RIDL, "model1"), False, args.pb,  False)

    if args.internal:
        ytrue_TL = get_ytrue(os.path.join(path_TL, "model1"), True, args.pb, True)
    else : 
        ytrue_TL = get_ytrue(os.path.join(path_TL, "model1"), True, args.pb, False)

    # 3) NO ENSEMBLING FIRST : GET LIST OF METRICS AND PREDICTED LABELS FOR ALL MODELS FOR EXTERNAL TEST SET ################################################

    if args.bacc : 
        metric = "b_acc"
    else : 
        metric = "roc_auc"

    # metric values for all 90 runs : lists of 90 ROC-AUC or Balanced Accuracy values
    metric_RIDL = create_list_metric(metric, path_RIDL, False, list_models, args.pb, args.internal)
    metric_TL = create_list_metric(metric, path_TL, True, list_models, args.pb, args.internal)

    ypred_RIDL = create_list_metric("y_pred", path_RIDL, False, list_models,  args.pb, args.internal)
    ypred_TL = create_list_metric("y_pred", path_TL, True, list_models, args.pb, args.internal)

    # std between the metrics of the 90 runs
    std_metric_RIDL = np.std(metric_RIDL, axis = 0, ddof =1)
    std_metric_TL = np.std(metric_TL, axis = 0, ddof =1) 
    # ddof = 1 because we use the sample standard deviation


    # 3) AVERAGE ENSEMBLING : BY BOOTSTRAPPING, GET LIST OF METRICS AND PREDICTED LABELS FOR ALL MODELS FOR EXTERNAL TEST SET FOR GROUPINGS OF MODELS ######

    # we shuffle the predicted labels before grouping them and computing the metric (roc auc or balanced accuracy)
    # to get a less biased result that doesn't depend on the sampling of the predicted labels as much
    # that way, we try different groupings of models for each group size 

    print( "number of samplings (with replacement) for bootstrapping :  ", args.nb_samplings)

    list_random = [_ for _ in range(1, int(args.nb_samplings)+1)]
    grouping_sizes = args.grouping_sizes
    print("list of sizes of model output groupings:", grouping_sizes,"\n")
    print("ypred_TL", np.shape(ypred_TL), type(ypred_TL[0]))
   
    print('TL metric values')
    mean_metric_bootstrapTL, std_metric_bootstrapTL = get_metric_bootstrap(list_random, ypred_TL, \
                                                                           ytrue_TL, grouping_sizes, metric="roc_auc")
    print(mean_metric_bootstrapTL)

    print('RIDL metric values')
    mean_metric_bootstrapRIDL, std_metric_bootstrapRIDL = get_metric_bootstrap(list_random, ypred_RIDL, \
                                                                               ytrue_RIDL, grouping_sizes, metric="roc_auc")
    print(mean_metric_bootstrapRIDL)

    # save les params plot quand j'aurai les bons à un fichier pkl
    params_plot = {
        "pb": args.pb, "metric_data_TL":mean_metric_bootstrapTL , "metric_data_RIDL":mean_metric_bootstrapRIDL,\
            "list_std_RIDL":std_metric_bootstrapRIDL, "list_std_TL": std_metric_bootstrapTL,\
                "list_std_noDE":[std_metric_RIDL, std_metric_TL], \
                "list_metric_noDE": [np.mean(metric_RIDL),np.mean(metric_TL)], \
                    "grouping_sizes":grouping_sizes
        }

    if args.save_metrics:
        current_date = datetime.now()

        # Format the date as "dd_mm_yyyy"
        date_string = current_date.strftime("%d_%m_%Y")

        print("Current date as string:", date_string)
        if args.bacc : 
            metricname = "_balanced_accuracy"
        else : metricname = "_roc_auc"
        if args.internal :
            test_set = "_internal_test"
        else : test_set = "_external_test"

        file_path = os.getcwd()+"/DeepEnsemble_"+args.pb+"_"+str(int(args.nb_samplings))+"_samplings_"+date_string+metricname+test_set+".pkl"

        # Open the file in binary write mode and save the dictionary using pickle.dump
        with open(file_path, 'wb') as f:
            pickle.dump(params_plot, f)

        print("Dictionary saved to : ", file_path)
    
    return params_plot


def main():
    """ 
    Parameters :
        args : dictionary of keyboard arguments such as the classification type (args.pb = "bipolar" or "scz"),
        whether to save the results in a pkl file, the number of samplings (args.nb_samplings), etc.
    Aim : 
        run deep ensembling using the predicted labels from the 90 models we ran for RI-DL and DL (90 models for each), 
        either classifying bipolar disorder or schizophrenia.
        Saves the computed metrics from deep ensembling if using the flag --save_metrics and saves or shows the results' plot 
        depending on the parameter 'save_or_show_plot'. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--pb", type=str, required=True,  default='bipolar', choices=['scz', 'bipolar'])
    parser.add_argument("--nb_models", type=int, default= 90 )
    parser.add_argument("--internal", action='store_true', help="flag to use internal test set : unlike the external test set we used in the published study,\
                        this test set contains sites that have been seen during training. it is therefore natural that the roc auc values\
                        are higher when using this test set rather than the external one. the external test set enables a truer interpretation\
                        of the results as the sites we evaluate the models on have not been seen during training.")
    parser.add_argument("--nb_samplings", type=int, default= 1e5)
    parser.add_argument("--nb_epochs", type=int,  default= 200 )
    parser.add_argument("--bacc", type=bool,  default= 0, help = "whether or not we perform the computations \
                         for balanced accuracy instead of the roc auc")
    parser.add_argument("--save_or_show_plot", type=str,  default='show', choices=['save', 'show'])
    parser.add_argument("--save_metrics", action='store_true')
    parser.add_argument("--grouping_sizes", type=list,  default=[2,5,10,15,20,30,40,50,60], \
                        help="the size of the groups of predictions (one prediction per model) we average for ensembling")


    keyboard_args = parser.parse_args()
    parameters_plot = runDeepEnsemble(keyboard_args)
    plot(show_or_save=keyboard_args.save_or_show_plot,**parameters_plot)

if __name__ == '__main__':
    main()



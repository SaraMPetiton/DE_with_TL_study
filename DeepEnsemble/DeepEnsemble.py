import pickle, os, random
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def get_ytrue(modelpath, TL, pb, internal=False):
    """ 
    Parameters :
        modelpath (str) : path to model 
        TL (bool) : whether it's a transfer learning model or not
        pb (str) : the name of the classification task ("bipolar" or "scz")
        internal (bool) : whether we're using the internal or external test datasets
    Aim : 
        return the true labels for the testing set at hand
    Output : 
        list of true labels for internal or external testing set for chosen classification task
    """
    if TL : finetuned = "finetuned_"
    else : finetuned = ""
    if internal : 
        picklefile = "Test_Intra_{densenet121}_{vbm}_{"+pb+"}_"+finetuned+"fold0_epoch199.pkl"
    else :
        picklefile = "Test_{densenet121}_{vbm}_{"+pb+"}_"+finetuned+"fold0_epoch199.pkl"
    
    return read_pkl(os.path.join(modelpath, picklefile))["y_true"]


def create_list_metric(metricname, modelpath, TL, list_models_, pb, internal=False):
    """ 
    Parameters :
        metricname (str) : name of chosen metric
        modelpath (str) : path to model 
        TL (bool) : whether it's a transfer learning model or not
        pb (str) : the name of the classification task ("bipolar" or "scz")
        internal (bool) : whether we're using the internal or external test datasets
    Aim : 
        return the list of length (number of models) of evaluated metrics
        for all models on chosen test set
    Output : 
        (list) of all metric values for each model
    """
    if TL : finetuned = "finetuned_"
    else : finetuned = ""
    if internal : 
        picklefile = "Test_Intra_{densenet121}_{vbm}_{"+pb+"}_"+finetuned+"fold0_epoch199.pkl"
    else :
        picklefile = "Test_{densenet121}_{vbm}_{"+pb+"}_"+finetuned+"fold0_epoch199.pkl"

    assert metricname in ["roc_auc","b_acc", "y_pred"], "Wrong metric name '{}'!".format(metricname)
    if metricname == "y_pred": 
        metric = "y_pred"
    if metricname == "roc_auc":
        metric = 'roc_auc on validation set'
    if metricname == "b_acc":
        metric = 'balanced_accuracy on validation set'
    list_metrics = []
    for model in list_models_:
        if metric != "y_pred":
            list_metrics.append(read_pkl(os.path.join(modelpath, model+"/"+picklefile))["metrics"][metric])
        else :
            list_metrics.append(read_pkl(os.path.join(modelpath, model+"/"+picklefile))[metric])

    return list_metrics




def get_metric_bootstrap(list_random, ypred_TL, ytrue_TL, grouping_sizes, metric="roc_auc"):
    """ 
    Parameters :
        list_random (list) : list of random seeds to use for bootstrapping
        ypred_TL (list) : list of predicted labels of shape (number of models, number of test dataset subjects)
        ytrue_TL (list) : list of true labels
        grouping_sizes (list) : list of grouping sizes for deep ensemble (groups of size 2, 5, 10, etc.)
        metric (str) : the name of the chosen metric for evaluation on the test set
    Aim : 
        return the mean and std of the list of shape (number of random samplings, number of groups) along axis 0,
        which is the mean values over all samplings of each metric for each group size
    Output : 
        (list) of all the mean and std of metric values for each group size 
    """
    list_of_metrics_by_seed = []
    for seed in list_random :
        random.seed(seed)
        list_of_metrics_by_group = []
        for group_size in grouping_sizes:
            group = random.sample(ypred_TL, group_size)
            group_mean = np.mean(group, axis =0)
            if metric=="roc_auc" :
                group_metric = roc_auc_score(ytrue_TL, group_mean)
            if metric == "b_acc":
                # threshold predictions for binary classification
                predictions = [1 if score >= 0.5 else 0 for score in ytrue_TL]
                group_metric = balanced_accuracy_score(predictions, group_mean)
            list_of_metrics_by_group.append(group_metric)
        list_of_metrics_by_seed.append(list_of_metrics_by_group)
    
    return np.mean(list_of_metrics_by_seed, axis = 0), np.std(list_of_metrics_by_seed, axis = 0)
            



def plot(show_or_save= "show" , **parameters_plot):
    group_sizes = ["no-DE"]+[str(i)+"-DE" for i in parameters_plot["grouping_sizes"]]
    pb = parameters_plot["pb"]

    metric_TL =  [round(100*parameters_plot["list_metric_noDE"][1],2)] + [round(100*float(i),2) for i in parameters_plot["metric_data_TL"]]
    print( " transfer auc to plot ", metric_TL)
    print("len auc t", len(metric_TL))

    metric_RIDL = [round(100*parameters_plot["list_metric_noDE"][0],2)]  + [round(100*float(i),2) for i in parameters_plot["metric_data_RIDL"]]
    print( " baseline auc to plot ", metric_RIDL)

    std_TL = [round(100*parameters_plot["list_std_noDE"][1],2)]+[round(100*float(i),2) for i in parameters_plot["list_std_TL"]]
    print( " transfer std to plot ", std_TL)

    std_RIDL = [round(100*parameters_plot["list_std_noDE"][0],2)]+[round(100*float(i),2) for i in parameters_plot["list_std_RIDL"]]
    print( " baseline std to plot ", std_RIDL)
    
    x = np.array(range(len(metric_RIDL)))

    sns.set_palette("deep")
    plt.plot(x, metric_RIDL, label = 'RI-DL', color = sns.color_palette()[0])
    plt.fill_between(x, np.array(metric_RIDL)-np.array(std_RIDL), np.array(metric_RIDL)+np.array(std_RIDL), alpha=0.3, color=sns.color_palette()[0])
    plt.errorbar(x , metric_RIDL, yerr = std_RIDL ,fmt='o', color=sns.color_palette()[0], ecolor=sns.color_palette()[0], capsize=4)


    plt.plot(x , metric_TL, label = 'TL', color=sns.color_palette()[1])
    plt.fill_between(x, np.array(metric_TL)-np.array(std_TL), np.array(metric_TL)+np.array(std_TL), alpha=0.3, color=sns.color_palette()[1])
    plt.errorbar(x , metric_TL, yerr = std_TL ,fmt='o', color=sns.color_palette()[1], ecolor=sns.color_palette()[1], capsize=4)

    if show_or_save == "show":
        fontsizetext = 30
        fontsizelabel = 50
        ticks_size = 29
    else : 
        fontsizetext = 9
        fontsizelabel = 15
        ticks_size = 9

    plt.xticks(x, group_sizes, fontsize = ticks_size)

    cpt = 0
    for i, value in enumerate(std_TL):
        plt.text(i+0.02, metric_TL[cpt]+value-0.01, str(value), ha='center', va='bottom' , fontsize = fontsizetext)
        cpt = cpt+1
    cpt = 0
    for i, value in enumerate(std_RIDL):
        plt.text(i+0.02, metric_RIDL[cpt]+value+0.01, str(value), ha='center', va='bottom' , fontsize = fontsizetext)
        cpt = cpt+1

    plt.grid(axis='y')

    plt.xlabel("T",fontsize= fontsizelabel)
    plt.ylabel("roc auc %",fontsize= fontsizelabel) # at last epoch of training
    legend = plt.legend(loc = "lower right", fontsize = fontsizelabel)
    for line in legend.get_lines():
        line.set_linewidth(8)
    
    if show_or_save == "show":
        plt.show()

    else :
        current_date = datetime.now()
        formatted_date = current_date.strftime("%d_%m_%Y")
        path_save = os.path.join(os.getcwd(),"saved_plots/DeepEnsemble_"+pb+"_"+formatted_date+"_"+str(len(parameters_plot["grouping_sizes"]))+"groupsizes.png")
        plt.savefig(path_save)
        



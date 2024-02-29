import pickle
import os
import torch
from collections import OrderedDict

def get_epoch_max_auc(path_testing_results, pkl_name):
    """
    Parameters :
        path_testing_results: 
            str : name of path where the pickle files containing the metrics for each tested epoch are saved.
        pkl_name: 
            str : name of pickle file containing the metrics.
    Aim : 
        Find the epoch at which the model at hand has the highest ROC-AUC value.
    Note : this script can be easily modified to use Balanced Accuracy instead of ROC-AUC.
    Reminder : 
        The models we ran for this study were trained for 200 epochs.
        We saved the models every 5 epochs from epoch 5 to epoch 199 (indexing starts at 0). 
        We evaluated the models saved at different epochs on the testing set using both the ROC-AUC and Balanced Accuracy metrics.
        We saved the values of these metrics in pickle files, in folders named "epochsave_<epoch>".
        We did not consider the last epoch of training in this function since the goal here is to interpolate 
        between the best and last epochs of training (we don't want them to be the same).
    """

    list_auc = []
    tested_model_epochs = list(range(5, 196, 5)) 
    path_testing_results = os.path.join(os.getcwd(),path_testing_results)
    assert len(tested_model_epochs) == sum(os.path.isdir(os.path.join(path_testing_results, item)) 
                                          and item.startswith("epochsave_") for item in os.listdir(path_testing_results))

    for epoch in tested_model_epochs:
        path_to_read = os.path.join(path_testing_results,"epochsave_"+str(epoch)+"/"+pkl_name)
        pickle_read = open(path_to_read, "rb")
        dict_ = pickle.load(pickle_read)
        list_auc.append(dict_["metrics"]["roc_auc on test set"])
        
    return tested_model_epochs[list_auc.index(max(list_auc))]



def get_best_auc_model(path, pb, pkl_name):
    """ 
    Parameters :
        path : (str) name of path where the pickle files containing the metrics for each tested epoch of a model are saved.
        pb : (str) "scz" or "bipolar" : classification task
        pkl_name : name of pickle file containing the metrics.
    Aim : 
        Find the epoch for which the model has the highest ROC-AUC value and return the weights of that model.
    Output :
        state_dict : the state dictionary of the model saved after 'best_epoch' epochs of training.
    Note : this function takes into account the fact that our models are saved in pth files named under the format
            {densenet121}_{vbm}_{"+<pb>+"}_finetuned_0_epoch_"+<epoch>+".pth 
    """

    best_epoch = get_epoch_max_auc(path, pkl_name)
    print("\nThe best epoch for models at path ",path," is : ", best_epoch,"\n")

    if "TL" in path:
        path_best_model = os.path.join(path, "{densenet121}_{vbm}_{"+str(pb)+"}_finetuned_0_epoch_"+str(best_epoch)+".pth")
    else:
        path_best_model = os.path.join(path, "{densenet121}_{vbm}_{"+str(pb)+"}_0_epoch_"+str(best_epoch)+".pth")
    if torch.cuda.is_available() and torch.cuda.device_count()>0:
        state_dict = torch.load(path_best_model)
    else :
        state_dict = torch.load(path_best_model, map_location=torch.device('cpu'))
    state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict["model"].items())

    return state_dict


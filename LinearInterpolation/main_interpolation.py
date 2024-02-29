import torch
from LinearInterp import get_state_dict, GetMetrics
import find_best_ROCAUC
import argparse
import  numpy as np
import pickle, os
from datetime import datetime
from interpolation_plots import plot_vertical

# paths to where our models (.pth format) are saved.
# checkpoint 0 ("CHKPT_0") and checkpoint 1 ("CHKPT_1") refer to interpolation coefficient 0 and 1
# i.e. the models between which we perform the linear interpolation (from chkpt 0 to chkpt 1)

MODELPATH_CHKPT_0_BD_TL = "saved_models/BD/TL/chkpt1/{densenet121}_{vbm}_{bipolar}_finetuned_0_epoch_199.pth"
MODELPATH_CHKPT_0_BD_RIDL = "saved_models/BD/RIDL/chkpt1/{densenet121}_{vbm}_{bipolar}_0_epoch_199.pth" 
MODELPATH_CHKPT_1_BD_TL = "saved_models/BD/TL/chkpt2/{densenet121}_{vbm}_{bipolar}_finetuned_0_epoch_199.pth"
MODELPATH_CHKPT_1_BD_RIDL = "saved_models/BD/RIDL/chkpt2/{densenet121}_{vbm}_{bipolar}_0_epoch_199.pth"

MODELPATH_CHKPT_0_SCZ_TL = 'saved_models/SCZ/TL/chkpt1/{densenet121}_{vbm}_{scz}_finetuned_0_epoch_199.pth'
MODELPATH_CHKPT_0_SCZ_RIDL = 'saved_models/SCZ/RIDL/chkpt1/{densenet121}_{vbm}_{scz}_0_epoch_199.pth'
MODELPATH_CHKPT_1_SCZ_TL = 'saved_models/SCZ/TL/chkpt2/{densenet121}_{vbm}_{scz}_finetuned_0_epoch_199.pth'
MODELPATH_CHKPT_1_SCZ_RIDL = 'saved_models/SCZ/RIDL/chkpt2/{densenet121}_{vbm}_{scz}_0_epoch_199.pth'


# path to models saved in pth format
CHKPT_DIR = os.path.join(os.getcwd(),"temp_results")

def save_to_pkl(dict, file_path, saveanyway = False):
    """ 
    Parameters :
        dict : python dictionary 
        file_path : (str) path where to save the dictionary to
        saveanyway : (bool) whether to save the file if it already exists
    Aim : 
        Save the dictionary "dict" to a pickle file at file path "file_path" unless the file already exists.
    """
    if os.path.exists(file_path) and saveanyway:
        print("Pickle exists at :", file_path)
        quit()
    with open(file_path, "wb") as file:
        # Dump the dictionary into the file
        pickle.dump(dict, file)
    print("Linear Interpolation data saved to : ", file_path)

def get_files(pb_):
    """ 
    Parameters :
        pb_ : (str) "scz" or "bipolar" : classification task 
    Aim : 
        to return the paths to the saved models (.pth format) used for interpolation.
    Output : 
        files : (dict)
        A python dictionary containing 2 lists: 'checkpoint_0' and 'checkpoint_1',
        each list contains 2 paths : the first is the path to the TL model, the second is the path to the RI-DL model.
        checkpoint_0 corresponds to the models at linear interpolation coefficient 0.
        checkpoint_1 corresponds to the models at linear interpolation coefficient 1.
        the format of 'files ' is therefore: 
        files = {'checkpoint_0 : [pathTL0, pathDL0], 'checkpoint_1', [pathTL1, pathDL1]}
    """
    if pb_ == "bipolar":
        files = {
            'checkpoint_0': [MODELPATH_CHKPT_0_BD_TL, MODELPATH_CHKPT_0_BD_RIDL], 
            'checkpoint_1': [MODELPATH_CHKPT_1_BD_TL, MODELPATH_CHKPT_1_BD_RIDL]
        }
          
    if pb_ == "scz":
        files = {
            'checkpoint_0': [MODELPATH_CHKPT_0_SCZ_TL, MODELPATH_CHKPT_0_SCZ_RIDL],
            'checkpoint_1': [MODELPATH_CHKPT_1_SCZ_TL, MODELPATH_CHKPT_1_SCZ_RIDL]
        }

    return files


def get_dictionary_metrics_interpolation(nb_lambda : int, pb : str ,pkl_nameTL : str, pkl_nameRIDL : str,
                                         path_saved_pkl_test_TL : str , path_saved_pkl_test_RIDL : str, root : str):
    """ 
    Parameters :
        nb_lambda : (int) number of linear interpolation coefficients.
        pb : (str) "scz" or "bipolar" : classification task 
        pkl_nameTL : (str) name of pickle files containing the metrics for TL models
        pkl_nameRIDL : (str) name of pickle files containing the metrics for RI-DL models
        path_saved_pkl_test_TL : (str) path to TL model for which you wish to interpolate between last and best-performing epochs
        path_saved_pkl_test_RIDL : (str) path to RI-DL model for which you wish to interpolate between last and best-performing epochs
        root : (str) path to datasets
    Aim : 
        Perform the linear interpolation between two RI-DL models and two TL models.
        Perform the linear interpolation between the best and last epochs of one of the two RI-DL models.
        Perform the linear interpolation between the best and last epochs of one of the two TL models.
        Save the ROC-AUC and Balanced Accuracy metrics associated with the interpolations in a pickle file in the 'saved_interpolations' folder.
    Note : Possibility to linearly interpolate a RI-DL model with a TL model. Some minor changes in the code would be needed.
            An example of a resulting plot can be found in the "plots_interpolation" folder.
    Output : 
        file_path : name of pickle file where 'dict' has been saved.
    """

    files = get_files(pb)

    
    my_dict_TL = {'root': root, 'checkpoint_dir': CHKPT_DIR, 'pb': pb, 'exp_name': "densenet121_vbm_"+pb+"_TL", \
            'sampler':'random', 'nb_epochs': 200, 'batch_size':64, 'device':('cuda' if torch.cuda.is_available() else 'cpu')}

    my_dict_RIDL = {'root': root, 'checkpoint_dir': CHKPT_DIR, 'pb': pb, 'exp_name': "densenet121_vbm_"+pb+"_RIDL", \
            'sampler':'random', 'nb_epochs': 200, 'batch_size':64, 'device':('cuda' if torch.cuda.is_available() else 'cpu')}

    
    print("\nThe classification task is : ", pb, "\n")

    state_dictest_TL = get_state_dict(files['checkpoint_0'][0])
    state_dictest_RIDL = get_state_dict(files['checkpoint_0'][1])
    state_dictest2_TL = get_state_dict(files['checkpoint_1'][0])
    state_dictest2_RIDL = get_state_dict(files['checkpoint_1'][1])

    AUC_transfer, BACC_transfer, coeffs = GetMetrics(state_dictest_TL, state_dictest2_TL, nb_lambda, **my_dict_TL)
    print("\n\nList of ROC-AUC values interpolation TL :", AUC_transfer)
    
    AUC_baseline, BACC_baseline, coeffs = GetMetrics(state_dictest_RIDL, state_dictest2_RIDL, nb_lambda, **my_dict_RIDL)
    print("\n\nList of ROC-AUC values interpolation RI-DL :", AUC_baseline)

    state_dictest_TL_BEST = find_best_ROCAUC.get_best_auc_model(path_saved_pkl_test_TL, pb,  pkl_nameTL)
    state_dictest_RIDL_BEST = find_best_ROCAUC.get_best_auc_model(path_saved_pkl_test_RIDL, pb,  pkl_nameRIDL)

    
    AUC_transfervsbest_transfer, BACC_transfervsbest_transfer, coeffs = GetMetrics(state_dictest_TL, state_dictest_TL_BEST, nb_lambda, **my_dict_TL)
    AUC_baselinevsbest_baseline, BACC_baselinevsbest_baseline, coeffs = GetMetrics(state_dictest_RIDL, state_dictest_RIDL_BEST, nb_lambda, **my_dict_RIDL)

    # dictionary to save the metrics to a pkl file : 

    dict = {"ROC-AUC TL": AUC_transfer, "ROC-AUC RIDL": AUC_baseline, 
            "Balanced Accuracy TL": BACC_transfer, "Balanced Accuracy RIDL": BACC_baseline,
            "ROC-AUC TL vs best TL": AUC_transfervsbest_transfer, 
            "Balanced Accuracy TL vs best TL": BACC_transfervsbest_transfer,
            "ROC-AUC RIDL vs best RIDL":AUC_baselinevsbest_baseline, 
            "Balanced Accuracy RIDL vs best RIDL":BACC_baselinevsbest_baseline
            }
    
    current_date = datetime.now()
    # Format the date as "day_month_twolastdigitsoftheyear"
    formatted_date = current_date.strftime("%d_%m_%y")
    file_path = "/neurospin/psy_sbox/temp_sara/LinearInterpolation/saved_interpolations/"+pb+"_"+formatted_date+".pkl"
    save_to_pkl(dict, file_path, True)

    return file_path


def main():
    parser = argparse.ArgumentParser()
    """ 
    Parameters :
        Argument Parser : controls the number of linear interpolation coefficients and 
        other parameters of the function get_dictionary_metrics_interpolation.
    Aim : 
        1. Call the get_dictionary_metrics_interpolation function for both classification tasks 
            (bipolar disorder and schizophrenia anatomical MRI classifications). 
            This performs the linear interpolations between transfer learning (TL) and randomly initialized deep learning (RI-DL) models.

        2. Plot the results using the ROC-AUC metric computed on the testing set of the bipolar disorder (BD) and schizophrenia (SCZ) datasets.
            The metric used for the plots here is ROC-AUC but it can be easily changed to Balanced Accuracy if needed.
            The plots are saved in the folder 'plots_interpolation'.
    """

    parser.add_argument('--nb_lambda', type=int, default=30, 
                        help="number of linear interpolation coefficients")
    
    parser.add_argument('--root', type=str, default = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data",
                        help="root path to datasets")
    
    parser.add_argument('--pkl_nameTL_BD', type=str, default="Test_{densenet121}_{vbm}_{bipolar}_finetuned.pkl", 
                        help="names of pickle files containing the metrics using TL for BD classification")
    
    parser.add_argument('--pkl_nameRIDL_BD', type=str, default="Test_{densenet121}_{vbm}_{bipolar}.pkl", 
                        help="names of pickle files containing the metrics using RI-DL for BD classification")
    
    parser.add_argument('--pkl_nameTL_SCZ', type=str, default="Test_{densenet121}_{vbm}_{scz}_finetuned.pkl", 
                        help="names of pickle files containing the metrics using TL for SCZ classification")
    
    parser.add_argument('--pkl_nameRIDL_SCZ', type=str, default="Test_{densenet121}_{vbm}_{scz}.pkl", 
                        help="names of pickle files containing the metrics using RI-DL for SCZ classification")

    parser.add_argument('--path_saved_pkl_test_TL_BD', type=str, default="saved_models/BD/TL/chkpt1", 
                        help="path to TL model 1 for BD classification")
    
    parser.add_argument('--path_saved_pkl_test_RIDL_BD', type=str, default="saved_models/BD/RIDL/chkpt1", 
                        help="path to RI-DL model 1 for BD classification")
    

    parser.add_argument('--path_saved_pkl_test_TL_SCZ', type=str, default="saved_models/SCZ/TL/chkpt1", 
                        help="path to TL model 1 for SCZ classification")
    
    parser.add_argument('--path_saved_pkl_test_RIDL_SCZ', type=str, default="saved_models/SCZ/RIDL/chkpt1", 
                        help="path to RI-DL model 1 for SCZ classification")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    file_pathBD = get_dictionary_metrics_interpolation(args.nb_lambda, "bipolar", 
                                                                pkl_nameTL = args.pkl_nameTL_BD,
                                                                pkl_nameRIDL = args.pkl_nameRIDL_BD,
                                                                path_saved_pkl_test_TL = args.path_saved_pkl_test_TL_BD, 
                                                                path_saved_pkl_test_RIDL = args.path_saved_pkl_test_RIDL_BD,
                                                                root=args.root)

    file_pathSCZ = get_dictionary_metrics_interpolation(args.nb_lambda, "scz", 
                                                                pkl_nameTL = args.pkl_nameTL_SCZ, 
                                                                pkl_nameRIDL = args.pkl_nameRIDL_SCZ, 
                                                                path_saved_pkl_test_TL = args.path_saved_pkl_test_TL_SCZ, 
                                                                path_saved_pkl_test_RIDL =args.path_saved_pkl_test_RIDL_SCZ,
                                                                root=args.root)

    
    plot_vertical(args.nb_lambda, file_pathBD, file_pathSCZ, plots_save_display="save")

if __name__ == '__main__':
    main()














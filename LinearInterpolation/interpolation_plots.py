import matplotlib.pyplot as plt
import pickle, os
import numpy as np
from datetime import datetime


# this function creates linear interpolation plots, one for BD, one for SCZ, where the BD plot is above the SCZ plot
def create_plot_BD_SCZ_vertical(imagename, plots_save_display, coeffs, auc_transfer_BD, auc_transfer_SCZ, auc_RIDL_BD, auc_RIDL_SCZ, 
                                auc_TBest_BD, auc_TBest_SCZ, AUC_RIDLBest_BD, AUC_RIDLBest_SCZ):
    """ 
    Parameters :
        imagename : (str) name of file to save image to.
        plots_save_display : (str) save" or "show" : whether to save or show the plot.
        coeffs : (np.array) array containing the linear interpolation coefficients.
        auc_transfer_BD : list of ROC-AUC values for TL in the case of BD classification.
        auc_transfer_SCZ : same thing for SCZ classification. 
        auc_RIDL_BD : list of ROC-AUC values for RI-DL in the case of BD classification.
        auc_RIDL_SCZ : same thing for SCZ classification. 
        auc_TBest_BD : list of ROC-AUC values for interpolation between the last and best epoch of a TL model for BD classification.
        auc_TBest_SCZ : same thing for SCZ.
        AUC_RIDLBest_BD : list of ROC-AUC values for interpolation between the last and best epoch of a RI-DL model for BD classification.
        AUC_RIDLBest_SCZ : same thing for SCZ. 
    Aim : 
        save or show the plotted ROC-AUC values against the interpolation coefficient values for both BD and SCZ tasks.
    """
    # Create a figure with subplots that share the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    tick_labels = np.array([f"{x:.2f}" for x in coeffs])
    tick_labels = [str(element) for element in tick_labels]

    if plots_save_display=="show":
        fontsize_xticks = 25
        fontsize_labels = 40
        fontsize_legend = 27
        labelsize = 20

    if plots_save_display=="save":
        fontsize_xticks = 10
        fontsize_labels = 15
        fontsize_legend = 7
        labelsize = 10

    # Plot the data on the subplots
    ax1.plot(coeffs, auc_transfer_BD, ".-", label = "TL to TL")
    ax1.plot(coeffs, auc_RIDL_BD, ".-", label = "RI-DL to RI-DL")
    ax1.plot(coeffs, auc_TBest_BD, ".-", label =  "TL to TL*")
    ax1.plot(coeffs, AUC_RIDLBest_BD, ".-", label = "RI-DL to RI-DL*")
    ax1.set_xticks(coeffs, tick_labels, rotation = 45, ha = "right", fontsize = fontsize_xticks)
    ax1.tick_params(axis='y', labelsize=labelsize)
    ax1.set_xlim(0, 1)

    ax2.plot(coeffs, auc_transfer_SCZ, ".-", label = "TL to TL")
    ax2.plot(coeffs, auc_RIDL_SCZ, ".-", label = "RI-DL to RI-DL")
    ax2.plot(coeffs, auc_TBest_SCZ, ".-", label =  "TL to TL*")
    ax2.plot(coeffs, AUC_RIDLBest_SCZ, ".-", label = "RI-DL to RI-DL*")
    ax2.set_xticks(coeffs, tick_labels, rotation = 65, ha = "right", fontsize = fontsize_xticks)
    ax2.set_xlabel("linear interpolation coefficient", fontsize = fontsize_labels, labelpad = 20)
    ax2.set_xlim(0, 1)
    ax2.tick_params(axis='y', labelsize=labelsize)


    # Set labels for the y-axis only on the left subplot
    ax1.set_ylabel('ROC-AUC', fontsize = fontsize_labels, labelpad=10)
    ax2.set_ylabel('ROC-AUC', fontsize = fontsize_labels,labelpad=10)

    ax1.set_title('BD', fontsize = fontsize_labels)
    ax2.set_title('SCZ', fontsize = fontsize_labels)

   # handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(loc='center right', fontsize = fontsize_legend)
    ax2.legend(loc='center right', fontsize = fontsize_legend)

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.15 ,bottom=0.15, top = 0.907)   
    
    if plots_save_display =="show":
        print("show vertical plot ...")
        plt.show()
    if plots_save_display=="save":
        print("save vertical plot ...")
        cwd = os.path.join(os.getcwd(),"plots_interpolation")
        plt.savefig(os.path.join(cwd,imagename))


def plot_vertical(nb_lambda, file_path_bd, file_path_scz, plots_save_display="save"):
    """ 
    Parameters :
        nb_lambda : (int) number of linear interpolation coefficients
        file_path_bd : (str) path to pickle file containing the linear interpolation metrics to plot for BD
        file_path_scz : (str) path to pickle file containing the linear interpolation metrics to plot for SCZ
        plots_save_display : (str) "save" or "show" the plot.
    Aim : 
        1. Read the files containing the metrics evaluated along the linear interpolation path for BD and SCZ classifications.
        2. Call function create_plot_BD_SCZ_vertical to plot these metrics either show or save the plot.
    """
    assert plots_save_display == "save" or plots_save_display == "show", "The variable plots_save_display should be equal to either 'save' or 'show'"

    formatted_date = datetime.now().strftime("%d_%m_%y")
    imagename = formatted_date+"_LinearInterpolation_AUC_"+str(nb_lambda)+"lambda_ROCAUC"

    # Open the pickle files in read mode
    with open(file_path_scz, 'rb') as file:
        # Load the dictionary from the pickle file
        dict_scz = pickle.load(file)

    with open(file_path_bd, 'rb') as file:
        # Load the dictionary from the pickle file
        dict_bd = pickle.load(file)

    # the linear interpolation coefficient takes nb_lambda values uniformly distributed between 0 and 1
    coeffs = np.linspace(0, 1, nb_lambda) 
    create_plot_BD_SCZ_vertical(imagename,plots_save_display, coeffs, dict_bd["ROC-AUC TL"], dict_scz["ROC-AUC TL"], 
                                dict_bd["ROC-AUC RIDL"], dict_scz["ROC-AUC RIDL"], dict_bd["ROC-AUC TL vs best TL"], 
                                dict_scz["ROC-AUC TL vs best TL"],dict_bd["ROC-AUC RIDL vs best RIDL"],dict_scz["ROC-AUC RIDL vs best RIDL"])




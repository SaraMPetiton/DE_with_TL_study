############################################################################
# Performance barrier experiments
# -------------------------------
#
# Experiments comparing the performance barrier interpolating the
# weights of two models. 
# Plot of the linear interpolation in function of the lambda coefficient of interpolation and in terms of roc auc 
# between two transfer (P-T) models, two randomly intialized (RI-T) models, one P-T model at its best and last epoch, and one RI-T model at its best and last epoch,
# the best epoch being the one at which the models have the highest roc auc on the testing dataset

import os, torch, pickle, nibabel, subprocess
import numpy as np
from collections import OrderedDict
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.nn import DataParallel
from collections import namedtuple
from torchvision.transforms.transforms import Compose
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from transforms import Crop, Padding, Normalize
from densenet import densenet121
from scz_bd_datasets import SCZDataset, BipolarDataset

DataItem = namedtuple("DataItem", ["inputs", "outputs", "labels"])
SetItem = namedtuple("SetItem", ["test", "train"])


def check_state_dicts(state_0, state_1):
    """ 
    Adapted from paper "What is being transfered in transfer learning" : https://arxiv.org/abs/2008.11687 
    Parameters :
        state_0: 
            Model* state dictionary for coefficient 0
        state_1: 
            Model* state dictionary for coefficient 1

    * = can be the same of different models
    
    Aim :
        Checks that the two state dictionaries share the same keys.
        This makes sure the format of the models are compatible to interpolate their weights.
    """
    assert sorted(state_0.keys()) == sorted(state_1.keys())


def get_state_dict(path):
    """
    Parameters : 
        path : 
            model path (we used models saved with '.pth' format)
    Aim :
        Return the state dictionary of model saved at path 'path'.
    Outputs : 
        dict :
            state dictionary of the model.
    """

    if torch.cuda.is_available() and torch.cuda.device_count()>0: 
        state_dict = torch.load(path)
    else : 
        state_dict = torch.load(path, map_location=torch.device('cpu'))

    # specific to the models we saved, if uncessesary return directly state_dict : 
    dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict["model"].items())
    return dict


def collate_fn(list_samples):
    """ 
    From https://github.com/Duplums/SMLvsDL.
    Parameters : 
        list_samples : 
            list of samples using indices from sampler.
    Aim :
        the function passed as the collate_fn argument is used to collate lists
            of samples into batches. A custom collate_fn is used here to apply the transformations.
    Outputs : 
        DataItem :
        named tuple named 'DataItem' containing 3 items named "inputs", "outputs", and "labels".            
    See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn.
    """
    data = dict(outputs=None) # compliant with DataManager <collate_fn>
    data["inputs"] = torch.stack([torch.from_numpy(np.array(sample[0])) for sample in list_samples], dim=0).float()
    data["labels"] = torch.stack([torch.tensor(np.array(sample[1],dtype=float)) for sample in list_samples], dim=0).squeeze().float()

    return DataItem(**data)

def get_dataloader(train=False, **args):
    """ 
    Adapted and simplified from https://github.com/Duplums/SMLvsDL. 
    Parameters : 
        **args : dictionary of parameters such as 'pb' (scz or bipolar), 'root' (rootpath), and 'batch_size'.
        train : not required. Change train to = True and uncomment last "if" statement if you want to retrieve the training set.
    Aim :
        Returns the DataLoader objects containing the data for training and testing for tiher the bipolar dataset or schizophrenia dataset.
    Outputs : 
        SetItem(train=trainloader, test=testloader)
        named tuple called "SetItem" containing two DataLoader objects named "test" and "train"
    """
    if args['pb'] == "scz":
        print("Loading the schizophrenia dataset ...")
        dataset_cls = SCZDataset 

    elif args['pb'] == "bipolar":
        print("Loading the bipolar dataset ...")
        dataset_cls = BipolarDataset 

    dataset = dict()

    # these transforms are important !
    input_transforms = Compose([Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'),  Normalize()])
        
    dataset["test"] = dataset_cls(args['root'], preproc="vbm", split="test",transforms=input_transforms, target=["diagnosis"])        

    testloader = DataLoader(dataset["test"], batch_size= args['batch_size'], collate_fn=collate_fn, num_workers= 3, pin_memory=True, drop_last=False)
    
    trainloader = None
    """
    if train:
        dataset["train"] = dataset_cls(args['root'], preproc="vbm", split="train", transforms=input_transforms, target=["diagnosis"])
        sampler_ = RandomSampler(dataset["train"])
        dataset = dataset["train"]
        trainloader = DataLoader(
            dataset, batch_size=args['batch_size'], sampler=sampler_,
            collate_fn=collate_fn, num_workers= 3, pin_memory=True, drop_last=False)
    """

    return SetItem(train=trainloader, test=testloader)

def test(model, loss_, loader : DataLoader, **args):
    """ 
    Adapted and simplified from https://github.com/Duplums/SMLvsDL. 
    Parameters : 
        model : here, a densenet121. Details can be found in the densenet.py script located in the current folder.
        loss_ : the type of loss used to compute the testing. Here, we use nn.BCEWithLogitsLoss from pytorch, 
                with parameters defined in the testing_interpLin function.
        loader : the DataLoader containing the data for testing ready to be fed to the model.
        **args : dictionary of parameters such as 'pb' (scz or bipolar), 'root' (rootpath), 'batch_size', 'checkpoint_dir', and 'exp_name'.
    Aim :
        Returns the metric values evaluated on the testing set using the model 'model', as well as the predicted and true data.
        Saves these values to a pickle file at folder 'checkpoint_dir' with 'exp_name' as the name of the file.
    Outputs : 
        y: array-like
            the predicted data.
        y_true: array-like
            the true data
        X: array_like
            the input data
        loss: float
            the value of the loss function.
        values: dict
            the values of the metrics : ROC-AUC and Balanced Accuracy in this case.
    """
    model.eval()
    nb_batch = len(loader)
    pbar = tqdm(total=nb_batch, desc="Mini-Batch")
    loss = 0
    values = {}

    with torch.no_grad():

        if args["device"]=='cuda' and not torch.cuda.is_available():
            raise ValueError("No GPU found.")
        
        y, y_true, X = [], [], []
        
        for dataitem in loader:
            
            pbar.update()
            inputs = dataitem.inputs
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(args['device'])

            list_targets, targets = [], []
            for item in (dataitem.outputs, dataitem.labels):
                if item is not None:
                    targets.append(item.to(args['device']))
                    y_true.extend(item.cpu().detach().numpy())

            if len(targets) == 1:
                targets = targets[0]
            elif len(targets) == 0:
                targets = None
            if targets is not None:
                list_targets.append(targets)
            outputs = model(inputs)
            
            if len(list_targets) > 0:
                batch_loss = loss_(outputs, *list_targets)
                loss += float(batch_loss) / nb_batch
                
            y.extend(outputs.cpu().detach().numpy())
            
            if isinstance(inputs, torch.Tensor):
                X.extend(inputs.cpu().detach().numpy())

            ytrue = torch.tensor(y_true).detach().cpu().numpy()
            ypred = torch.tensor(y).detach().cpu().numpy()
            values["roc_auc on test set"] = roc_auc_score(ytrue, ypred)

            assert len(ypred.shape) == 1, "The vector of predictions y does not have the right number of dimensions"
            ypred_acc = (torch.tensor(y).detach().cpu().numpy() > 0)
            values["balanced_accuracy on test set"] = balanced_accuracy_score(ytrue, ypred_acc)
        
        pbar.close()

    saving_dir = args['checkpoint_dir']
    exp_name = args["exp_name"]

    if saving_dir is not None:
        if not os.path.isdir(saving_dir):
            subprocess.check_call(['mkdir', '-p', saving_dir])
            print("Directory %s created."%saving_dir)
        with open(os.path.join(saving_dir, exp_name+'.pkl'), 'wb') as f:
            pickle.dump({'y_pred': y, 'y_true': y_true, 'loss': loss, 'metrics': values}, f)

    return y, X, y_true, loss, values  

def testing_interpLin(state_dict_, **args):
    """
    Adapted and simplified from https://github.com/Duplums/SMLvsDL. 
    Parameters :
        state_dict_ :  state dictionary of interpolated weights 
        **args : dictionary of parameters such as 'pb' (scz or bipolar), 'root' (rootpath), 
            'nb_epochs', 'exp_name', 'device' (cuda available or not)
    Aim :
        1. Loads the DataLoader containing the testing dataset
        2. Loads the state dictionary containing the interpolated weights into the model
        3. Calls the 'test' function to perform testing on the testing dataset
        4. Saves the metrics from testinf to a dictionary 
    Output :
        results_tests : dictionary containing the ROC-AUC and Ballanced Accuracy evaluated on the testing set
                        using the model with the weights of 'state_dict_'
    """
    net = densenet121()

    pos_weights = {"scz": 1.131, "bipolar": 1.584}
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights[args['pb']], dtype=torch.float32,
                                                            device=args["device"]))

    results_tests = []
    loader = get_dataloader(**args)

    if loader is not None : print("...loaded.")
    assert state_dict_!=None,"The state dictionary provided is empty!!"
    
    try :
        print("Loading the model for linear interpolation ...")
        net.load_state_dict(state_dict_)
    except BaseException as e:
        print('Error while loading the linearly interpolated weights: %s' % str(e))
    print("... state dictionary loaded.")
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    net = net.to(args['device'])

    y, X, y_true, loss, values = test(net, loss, loader.test, **args)
    results = {'y_pred': y, 'y_true': y_true, 'loss': loss, 'metrics': values}
    results_tests.append({"test type":"test", "metrics":results["metrics"]})

    return results_tests



def eval_fn(state_dict, **args):
    """
    Adapted from paper "What is being transfered in transfer learning" : https://arxiv.org/abs/2008.11687.

    Parameters :
        state_dict : state dictionary of interpolated weights 
        **args : dictionary of parameters needed in testing_interpLin
    Aim :
        Perform the evaluation of a model with state dictionary 'state_dict' with function testing_interpLin.
        Return the ROC-AUC and Balanced Accuracy evaluated on our testing dataset.
    
    Output :
        results_rocAUC, results_accuracy : 
            ROC AUC and Balanced Accuracy values for current configuration.
    """
    results_tests = testing_interpLin(state_dict, **args)
    res_inter = [d["metrics"] for d in results_tests if d["test type"] == "test"][0]
    results_rocAUC = res_inter['roc_auc on test set'] 
    results_accuracy = res_inter['balanced_accuracy on test set']

    return results_rocAUC, results_accuracy


def interpolate_state_dicts(state_0, state_1, coeff):
    """ 
    Adapted from paper "What is being transfered in transfer learning" : https://arxiv.org/abs/2008.11687.
    Parameters :
    state_0 : 
        Model state dictionary for interpolation coefficient 0.
    state_1 : 
        Model state dictionary for interpolation coefficient 1.
    coeff :
        Linear interpolation coefficient for current interpolation.

    * = can be the same of different models

    Aim ------------------------------------------------------------
        Interpolates the weights of the two models with coefficient 'coeff'.

    Output ---------------------------------------------------------
        A new dictionary where the weights are equal to those of state_0 interpolated with the weights of state_1.
    """

    return {key: (1 - coeff) * state_0[key] + coeff * state_1[key]
            for key in state_0.keys()}


def eval_interpolation(state_0, state_1, n_coeffs, **args):
    """ 
    Adapted from paper "What is being transfered in transfer learning" : https://arxiv.org/abs/2008.11687.
    Parameters :
    state_0 : 
        state dictionary at interpolation coefficient = 0, weights from which we perform the interpolation
    state_1 : 
        state dictionary at interpolation coefficient = 1, weights to which we perform the interpolation
    n_coeffs : 
        number of linear interpolation coefficients
    **args : 
        dictionary of parameters needed in testing_interpLin

    Aim :
        Generates the coefficients from 0 to 1 and the number of coefficients chosen (n_coeffs)
        Creates new interpolated state dictionary for each coefficient value using interpolate_state_dicts function.
        Creates new model using the interpolated dictionary and evaluates it with ROC-AUC and Balanced Accuracy metrics using eval_fn function.
    Output :
        coeffs : 
            linear interpolation coefficients (list)
        metrics_auc: 
            roc auc for each coefficient (list)
        metrics_bal_acc:  
            balanced accuracy for each coefficient (list)
    """

    check_state_dicts(state_0, state_1)
    coeffs = np.linspace(0, 1, n_coeffs)
    metrics_auc , metrics_bal_acc = [], []

    for idx in range(n_coeffs):
        print("coefficient index = ", idx, "/", n_coeffs)
        new_state_dict = interpolate_state_dicts(state_0, state_1, coeffs[idx])
        auc, b_acc = eval_fn(new_state_dict, **args)
        metrics_auc.append(auc)
        metrics_bal_acc.append(b_acc)

    return coeffs, metrics_auc, metrics_bal_acc

def GetMetrics(state_dictest_0, state_dictest_1, nbcoef, **args):  
    """
    Parameters :
        state_dictest_0 : 
            state dictionary at linear interp coefficient 0
        state_dictest_1 : 
            state dictionary at linear interp coefficient 1
        nbcoef : 
            number of linear interpolation coefficient values between 0 and 1 (included)
        **args : 
            dictionary of parameters needed in testing_interpLin
    Aim :
        The linear interpolation is performed from model with state_dictest_0 to model with state_dictest_1.
    Output :
        AUC, BACC :
            two lists of ROC AUC and Balanced Accuracy values for each linear interp coeff.
        coeffs :
            A list of these coefficients.
    """
    coeffs, metrics_auc, metrics_b_accuracy = eval_interpolation(state_dictest_0, state_dictest_1, nbcoef, **args)
    coeffs = coeffs.tolist()
    AUC = [item for item in metrics_auc]
    BACC = [item for item in metrics_b_accuracy]

    return AUC, BACC, coeffs






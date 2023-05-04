##################################################
# # Skip heuristics algorithm with trainable hyperparameters using validation predictions after training.
##################################################
# Author: Sanjeev Kumar
# Email: sanjeev.kumar(at)student.uni-siegen.de
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# Author: Michael Moeller
# Email: michael.moeller(at)uni-siegen.de
##################################################


from data_processing.sliding_window import apply_sliding_window
from misc.torchutils import seed_worker
from skip_heuristics_scripts.data_skipping import data_skipping
# from model.evaluation_simulated_annealing import print_saving_activity_wise
from log_data_scripts.plotting import (
    plot_stems_f1,
    plot_stems_data_saving,
    plot_stems_comp_saving,
)
from sklearn.metrics import accuracy_score, jaccard_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch

def skip_heuristics_for_val_preds_as_val_gt(data, args, threshold_value, tolerance_value, all_mod_eval_output):
    
    # config dictionary containing setting parameters
    config = vars(args)
    config["saving_threshold"] = threshold_value
    config["saving_tolerance"] = tolerance_value
    f1_scores_sbj_array = []
    data_saved_sbj_array = []
    comp_saved_sbj_array = []

    for sbj in np.unique(data[:, 0]):
        # print('-'*50)
        # loading data
        # print('\n DATA SKIPPING APPLIED ON VALIDATION DATASET: SUBJECT {0} OF {1}'.format(int(sbj) + 1, int(np.max(data[:, 0])) + 1))
        train_data = data[data[:, 0] != sbj]  # training data from all but one subject
        val_data = data[data[:, 0] == sbj]  # validaaton data from one subject

        # calculate concurrent windows
        curr_label = None
        curr_window = 0
        windows = []
        for sbj_id in np.unique(train_data[:, 0]):  # first column is subject id
            sbj_label = train_data[train_data[:, 0] == sbj_id][:, -1]  # label column for each subject in training data
            for label in sbj_label:
                if label != curr_label and curr_label is not None:
                    windows.append(
                        [curr_label, curr_window / args.sampling_rate, curr_window]
                    )  # store training duration in terms of number of windows, curr_window is actually data point here
                       # 'curr_window / args.sampling_rate' calculates number of windows without overlapping
                    curr_label = label                    
                    curr_window = 1   # reset curr_window to 1
                elif label == curr_label:
                    curr_window += 1 # curr_window is actually data point and not the window as apply_window function has not been used yet
                else:
                    curr_label = label
                    curr_window += 1
        windows = np.array(
            windows
        )  

        # calculate savings array, calculates activity duration 
        # (in terms of number of windows without overlap) for each class/activity
        saving_array = np.zeros(args.nb_classes)
        for label in range(args.nb_classes):
            label_windows = windows[
                windows[:, 0] == label
            ]  # accessing windows label wise
            if label_windows.size != 0:
                if args.saving_type == "mean":
                    saving_array[int(label)] = np.mean(
                        label_windows[:, 1].astype(float)
                    )  # mean of activity duration of each activity across all subjects
                elif args.saving_type == "median":
                    saving_array[int(label)] = np.median(
                        label_windows[:, 1].astype(float)
                    )  # median of activity duration of each activity across all subjects
                elif args.saving_type == "min":
                    saving_array[int(label)] = np.min(label_windows[:, 1].astype(float))
                elif args.saving_type == "max":
                    saving_array[int(label)] = np.max(label_windows[:, 1].astype(float))
                elif args.saving_type == "first_quartile":
                    saving_array[int(label)] = np.percentile(
                        label_windows[:, 1].astype(float), 25
                    )

        args.saving_array = saving_array
        # print(f'{args.saving_type} activity duaration for all activities: \n',
        # saving_array)

        # Sensor data is segmented using a sliding window mechanism
        X_val, y_val = apply_sliding_window(
            val_data[:, :-1],
            val_data[:, -1],
            sliding_window_size=args.sw_length,
            unit=args.sw_unit,
            sampling_rate=args.sampling_rate,
            sliding_window_overlap=args.sw_overlap,
        )

        X_val = X_val[:, :, 1:]  # removing subj_id from X_val

        val_features, val_labels = X_val, y_val

        g = torch.Generator()
        g.manual_seed(config["seed"])

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(val_features), torch.from_numpy(val_labels)
        )
        valloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
        )

        # helper objects
        val_gt = []
        with torch.no_grad():
            computations_saved = np.zeros(config["nb_classes"])
            data_saved = np.zeros(config["nb_classes"])
            # iterate over validation dataset
            for x, y in valloader:
                # send x and y to GPU
                inputs, targets = x.to(config["gpu"]), y.to(config["gpu"])
                y_true = targets.cpu().numpy().flatten()
                val_gt = np.concatenate((np.array(val_gt, int), np.array(y_true, int)))

            #! change this
            mod_val_preds = np.copy(val_gt)
            # feed validation data predictions, skip over some data
            mod_val_preds, data_saved, computations_saved = data_skipping(
                mod_val_preds, config, data_saved, computations_saved
            )

            count_array = np.zeros(config["nb_classes"])
            for k in range(config["nb_classes"]):
                count_array[k] = (val_gt == k).sum()

            comp_windows, data_windows = (
                count_array,
                count_array * (1 - config["sw_overlap"] * 0.01) * config["sw_length"], #! how did we get this
            )
            f1_scores_sbj_array.append(
                f1_score(val_gt, mod_val_preds, average="macro")
            )
            data_saved_sbj_array.append(sum(data_saved) / sum(data_windows))
            comp_saved_sbj_array.append(sum(computations_saved) / sum(comp_windows))

    avg_f1_score_sbj = round(np.mean(f1_scores_sbj_array),2)
    avg_data_saved_sbj = round(np.mean(data_saved_sbj_array) * 100, 2)
    avg_comp_saved_sbj = round(np.mean(comp_saved_sbj_array) * 100, 2)

    return avg_f1_score_sbj, avg_comp_saved_sbj, avg_data_saved_sbj

def skip_heuristics(activity, args, window_threshold, skip_windows, tolerance_value, all_eval_output):
    """
    Applies a data skipping technique over the validation predictions to get modified predictions,
    calculates and returns several evaluation metrics and saving ratios for each activity.
    
    Parameters:
    activity (int): The activity for which to calculate evaluation metrics and saving ratios.
    args (Namespace): A namespace containing the arguments.
    window_threshold (float): The threshold to apply for data skipping.
    skip_windows (int): The number of windows to skip during data skipping.
    tolerance_value (float): The tolerance value to use during data skipping.
    all_eval_output (ndarray): An ndarray containing the evaluation outputs for all activities.
    
    Returns:
    f_one_gt_mod_val (float): The f1-score for ground truth modified validation predictions.
    f_one_gt_val (float): The f1-score for ground truth validation predictions.
    f_one_val_mod_val (float): The f1-score for modified validation predictions.
    f_one_gt_mod_val_avg (float): The average f1-score for ground truth modified validation predictions.
    f_one_gt_val_avg (float): The average f1-score for ground truth validation predictions.
    comp_saved_ratio (float): The computation saving ratio for the specified activity.
    data_saved_ratio (float): The data saving ratio for the specified activity.
    computations_saved (ndarray): An ndarray containing the computation saved for each activity.
    data_saved (ndarray): An ndarray containing the data saved for each activity.
    comp_windows (ndarray): An ndarray containing the total number of windows for each activity.
    data_windows (ndarray): An ndarray containing the total amount of data for each activity.
    all_mod_val_preds (ndarray): An ndarray containing modifieed validation predictions. 
    """  
    
    config = vars(args)
    config["saving_window_threshold"] = window_threshold
    config["saving_tolerance"] = tolerance_value
    config["saving_skip_windows"] = skip_windows

    #print(config["saving_threshold"])

    computations_saved = np.zeros(config["nb_classes"])
    data_saved = np.zeros(config["nb_classes"])
    
    # defining validation predictions, validation gt and madified validation predictions
    all_val_preds = all_eval_output[:,0]
    all_mod_val_preds = np.copy(all_eval_output[:,0])
    all_val_gt = all_eval_output[:,1]

    # count total observations for each activity in validation gt 
    count_array = np.zeros(config["nb_classes"])
    for k in range(config["nb_classes"]):
        count_array[k] = (all_val_gt == k).sum()

    # apply data skip over validation predictions to get modified predictions
    all_mod_val_preds, data_saved, computations_saved = data_skipping(activity,
        all_mod_val_preds, config, data_saved, computations_saved,  apply_best=False
    )

    # total windows for each activity in validation gt 
    comp_windows, data_windows = (
        count_array,
        count_array * (1 - config["sw_overlap"] * 0.01) * config["sw_length"], #! how did we get this
    )

    # for activity wise calculations
    # percentage saving for this activity
    # saving percentage in each activity
    comp_saved_ratio_activity = (computations_saved/comp_windows*100)
    data_saved_ratio_activity = (data_saved/data_windows*100)
    # print_saving_activity_wise(activity, config["dataset"], comp_saved_ratio_activity, data_saved_ratio_activity)
    
    f_one_gt_mod_val = f1_score(all_val_gt, all_mod_val_preds, labels = np.array([activity]), average= None) 
    f_one_gt_val = f1_score(all_val_gt, all_val_preds, labels = np.array([activity]), average= None) 
    f_one_val_mod_val = f1_score(all_val_preds, all_mod_val_preds, labels = np.array([activity]), average= None) 
    f_one_gt_mod_val_avg = f1_score(all_val_gt, all_mod_val_preds, average= 'macro')
    f_one_gt_val_avg = f1_score(all_val_gt, all_val_preds, average= 'macro')

    data_saved_ratio = data_saved_ratio_activity[activity]
    comp_saved_ratio = comp_saved_ratio_activity[activity]        

    # # uncomment this for combined data: all_mod_val_preds
    # f1_scores = f1_score(all_val_gt, all_mod_val_preds, average="macro") 
    # accuracy_scores = accuracy_score(all_val_gt, all_mod_val_preds)
    # recall_scores = recall_score(all_val_gt, all_mod_val_preds, average="macro")
    # precision_scores = precision_score(all_val_gt, all_mod_val_preds,average="macro")         
    # data_saved_ratio = sum(data_saved) / sum(data_windows)
    # comp_saved_ratio = sum(computations_saved) / sum(comp_windows)

    return f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val, f_one_gt_mod_val_avg, f_one_gt_val_avg, comp_saved_ratio, data_saved_ratio, computations_saved, data_saved, comp_windows, data_windows, all_mod_val_preds
    # return f1_scores, accuracy_scores, recall_scores, precision_scores, comp_saved_ratio, data_saved_ratio, computations_saved, data_saved, comp_windows, data_windows, all_mod_val_preds
    # return f1_scores_activity, comp_saved_ratio, data_saved_ratio
    # return f1_scores, comp_saved_ratio, data_saved_ratio

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score
from skip_heuristics_scripts.skip_heuristics import skip_heuristics
import numpy as np
import matplotlib.pyplot as plt
def print_saving_activity_wise(activity, datasetname, comp_saved_ratio_activity, data_saved_ratio_activity):
    
    if datasetname == 'rwhar':
        label_name = ['climbing_down', 'climbing_up', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
    if datasetname == 'wetlab':
        label_name = ['null_class', 'cutting', 'inverting', 'peeling', 'pestling',\
                       'pipetting', 'pouring', 'pour catalysator', 'stirring', 'transfer']

    # uncomment for combined dataset   
    # for i in range(len(label_name)):
    #     print(f'Computation saved for activity {label_name[i]} is {comp_saved_ratio_activity[i]} %')
        # print(f'Data saved for activity {label_name[i]} is {data_saved_ratio_activity[i]} %', '\n*2')
    
    # comment this if not using for activity only
    print(f'Computation saved for activity {label_name[activity]} is {comp_saved_ratio_activity[activity]} %')

    return None
def evaluate_sim_ann_on_best(datasetname, activity, args, best, all_eval_output):
    config = vars(args)
    window_threshold = best[:,0]
    skip_windows = best[:,1]
    tolerance_value = best[:,2]
    
    f_one_gt_mod_val, f_one_gt_val, f_one_val_mod_val, \
        comp_saved_ratio, data_saved_ratio, computations_saved,\
            data_saved, comp_windows, data_windows, \
                all_mod_val_preds= skip_heuristics(activity, args, window_threshold, 
                                                                       skip_windows, tolerance_value, all_eval_output)
    # seperate activities   
    # print(computations_saved, "\n", comp_windows)
    # print(data_saved, "\n", data_windows)

    # saving percentage in each activity
    comp_saved_ratio_activity = np.round(computations_saved/comp_windows*100,2)
    data_saved_ratio_activity = np.round(data_saved/data_windows*100,2)
    # print_saving_activity_wise(activity, datasetname, comp_saved_ratio_activity, data_saved_ratio_activity)

    # accuracy_scores_activity = accuracy_score(all_val_gt_activity, all_mod_val_preds_activity)
    # precision_scores_activity = precision_score(all_val_gt_activity, all_mod_val_preds_activity, labels = np.array([activity]), average= None)
    # recall_scores_activity = recall_score(all_val_gt_activity, all_mod_val_preds_activity,labels = np.array([activity]), average= None )
    # accuracy_score_full_data = accuracy_score(all_val_gt, all_mod_val_preds)
    # confusion_matrix_activity = confusion_matrix(all_val_gt_activity, all_mod_val_preds_activity)
    activity_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_activity,display_labels=activity_labels)
    # disp.plot()
    # plt.show()
    # print(classification_report(all_val_gt_activity, all_mod_val_preds_activity, digits=3))
    print(f"f1_gt_mod_val and  f_one_gt_val for activity {args.class_names[activity]}: {f_one_gt_mod_val}, {f_one_gt_val}")
            # f"Accuracy for activity {activity} : {accuracy_scores_activity} \n",
            # f"Recall for activity {activity} : {recall_scores_activity}\n",
            # f"Precision for activity {activity} : {precision_scores_activity}\n")
    # defining validation predictions, validation gt and madified validation predictions
    all_val_preds = all_eval_output[:,0]
    all_mod_val_preds = np.copy(all_eval_output[:,0])
    all_val_gt = all_eval_output[:,1]

    # comp saved between val&mod_val
    # count total observations for each activity in validation gt 
    count_array = np.zeros(config["nb_classes"])
    for k in range(config["nb_classes"]):
        count_array[k] = (all_val_preds == k).sum()

    # total windows for each activity in val pred 
    comp_windows_val_pred = count_array

    comp_saved_ratio_val_pred = (computations_saved/comp_windows_val_pred*100)
    comp_saved = skip_windows/(skip_windows + window_threshold)*100     

    print(f'Comp saved for {args.class_names[activity]}: {comp_saved}')
    # # f1 score on full data 
    # print(f"f1 score full data: {f_one}")

    return all_mod_val_preds, f_one_gt_mod_val, comp_saved
    # return all_mod_val_preds
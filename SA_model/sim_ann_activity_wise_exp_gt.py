
import numpy as np
import time
import os
from sklearn.metrics import f1_score
# from Other_helpful_scripts.bar_plot_act_f1_comp import bar_plot_act_f1_comp
# from Other_helpful_scripts.graph_activities_gt_val_mod_val import apply_best_settings_get_f1_full_data, graph_gt_val_mod_val
from SA_model.generate_input_data_for_SA import ml_generate_train_data, ml_generate_train_data_exp_gt

from SA_model.simulated_annealing import simulated_annealing
from log_data_scripts.save_csv_results import activity_save_best_results_to_csv
from ml_evaluate import mod_bar_graph
from ml_validation import ml_validation

 
def window_to_time(window, config):
    # number of windows to time equation
    # n is number of windows
    # one window is 1 sec
    one_window_duration = 1 
    t = (window-1)*(1-config["sw_overlap"]/100)*one_window_duration + one_window_duration 
    return t

def sim_ann_activity_wise_gt(args, window_threshold, skip_windows, tol_value, max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val, \
                           init_temp, ann_rate,  log_date, log_timestamp, data):
    config = vars(args)
    log_dir = os.path.join('logs', log_date, log_timestamp)

    if config["dataset"] == 'rwhar':
        label_name = ['climbing_down', 'climbing_up', 'jumping', 'lying',\
                       'running', 'sitting', 'standing', 'walking']
        activity_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    if config["dataset"] == 'wetlab':
        label_name = ['null_class', 'cutting', 'inverting', 'peeling', 'pestling',\
                       'pipetting', 'pouring', 'stirring', 'transfer']
        activity_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    
    # defining range #! tol = 0
    window_threshold = np.array([1,100])
    skip_windows = np.array([1,100])
    tol_value = np.array([0, 0])

    for _, sbj in enumerate(np.unique(data[:, 0])):
    # for sbj in [2]:
        # generating training data (validation data -> leave-one-out)
        ml_train_gt_gt = ml_generate_train_data_exp_gt(data, args, sbj)
        
        # # training for tolerance zone only 
        # best_filename_gt = ''
        # best = np.loadtxt(best_filename_gt, skiprows=1, usecols=(1,2,3),delimiter=',').T
        # best_threshold = best[0]
        # best_win_skip = best[1]

        # window_threshold = np.array([best_threshold,best_threshold])
        # skip_windows = np.array([best_win_skip,best_win_skip])
        # tol_value = np.array([0, 10])


        # creating empty lists to save best settings, performance metrics 
        best_thrs_for_activity_lst = []
        best_skip_win_for_activity_lst = []
        best_tol_val_for_activity_lst = []
        best_f1_for_activity_lst = []
        f_one_target_lst = list()
        best_data_saved_for_activity_lst = []
        best_comp_saved_for_activity_lst = []
        best_loss_for_activity_lst = []
        elapsed_time_lst = []
        acitivity_name_lst = []
        f_one_gt_mod_val_lst = []
        f_one_gt_val_lst = []
        f_one_val_mod_val_lst = []
        f_one_gt_mod_val_avg_lst = []
        f_one_gt_val_avg_lst = []

        # training for each activity  
        for labels in activity_labels:
            activity = labels
            activity_name = label_name[activity]
            # get the start time
            start_time = time.time()
            # running SA metaheuristic
            best, loss, best_f1, f_one_target, best_data_saved, best_comp_saved,\
                f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val,\
                    f_one_gt_mod_val_avg, f_one_gt_val_avg =\
                simulated_annealing(activity, activity_name, args, window_threshold,\
                                    skip_windows, tol_value, max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val,\
                                        init_temp, ann_rate, log_date, log_timestamp, ml_train_gt_gt)
            # get the end time
            end_time = time.time()
            elapsed_time = end_time - start_time

            # printing best settings for each activity
            print(f"Done! for activity: {label_name[activity]}")
            print(f"Optimum window threshold is: {best[:,0]} \n", 
                        f"Optimum skip windows is: {best[:,1]} \n",
                        f"Optimum tolerance value is: {best[:,2]} \n",
                        f"F1 score (for particular activity) at optimum hyperparameter: {best_f1} target was {f_one_target} \n",
                        f"Avg. modified f1 score (for all activities) at optimum hyperparameter: {f_one_gt_mod_val_avg} target was {1} \n",
                        # f"Data saved at optimum hyperparameter: {best_data_saved} \n",
                        f"Computation saved at optimum hyperparameter: {best_comp_saved} \n",
                        f"Lowest loss value : {loss} \n",
                        f"Total time elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} \n\n")
            print(f"optimum time after which device will be switched off: {window_to_time(best[:,0],config)} seconds")
            print(f" optimum switch off duration: {window_to_time(best[:,1], config)} seconds")

            # appending best settings
            best_thrs_for_activity_lst.append(float(best[:,0]))
            best_skip_win_for_activity_lst.append(float(best[:,1]))
            best_tol_val_for_activity_lst.append(float(best[:,2]))
            best_f1_for_activity_lst.append(best_f1)
            f_one_target_lst.append(f_one_target)
            best_data_saved_for_activity_lst.append(best_data_saved)
            best_comp_saved_for_activity_lst.append(best_comp_saved)
            best_loss_for_activity_lst.append(loss)
            acitivity_name_lst.append(activity_name)
            # get the execution time
            elapsed_time_lst.append(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            f_one_gt_mod_val_lst.append(f_one_gt_mod_val)
            f_one_gt_val_lst.append(f_one_gt_val)
            f_one_val_mod_val_lst.append(f_one_val_mod_val)
            f_one_gt_mod_val_avg_lst.append(f_one_gt_mod_val_avg)
            f_one_gt_val_avg_lst.append(f_one_gt_val_avg)
        # saving best settings for each subject 
        algo_name = 'SA'
        filename_best_csv = activity_save_best_results_to_csv(best_thrs_for_activity_lst, 
                                    best_skip_win_for_activity_lst,
                                    best_tol_val_for_activity_lst,
                                    best_f1_for_activity_lst, f_one_target_lst, best_data_saved_for_activity_lst, 
                                    best_comp_saved_for_activity_lst,
                                    best_loss_for_activity_lst, elapsed_time_lst, acitivity_name_lst,f_one_gt_mod_val_lst,
                                    f_one_gt_val_lst, f_one_val_mod_val_lst,f_one_gt_mod_val_avg_lst,f_one_gt_val_avg_lst, log_dir, args, algo_name, sbj)

        # to create bar plot of f1 score after training (each activity is only trained individually and best settings are not applied to whole data set)
        # bar_plot_act_f1_comp(sbj, acitivity_name_lst, best_f1_for_activity_lst, f_one_gt_val_lst, best_comp_saved_for_activity_lst, log_dir, config, algo_name, f1_avg =False)

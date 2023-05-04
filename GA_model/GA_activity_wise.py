

import time
import os
import numpy as np
from GA_model.genetic_algo import genetic_algorithm
from Other_helpful_scripts.bar_plot_act_f1_comp import bar_plot_act_f1_comp
from SA_model.generate_input_data_for_SA import ml_generate_train_data
# from Other_helpful_scripts.graph_activities_gt_val_mod_val import apply_best_settings_get_f1_full_data, graph_gt_val_mod_val
from log_data_scripts.save_csv_results import activity_save_best_results_to_csv

def window_to_time(window, config):
    # number of windows to time equation
    # n is number of windows
    # one window is 1 sec
    one_window_duration = 1 
    t = (window-1)*(1-config["sw_overlap"]/100)*one_window_duration + one_window_duration 
    return round(t,2)

def ga_activity_wise(args, data, bounds, n_bits,  n_pop, r_cross, r_mut, termin_iter, max_iter, log_date, log_timestamp, *arg):

    config = vars(args)
    log_dir = os.path.join('logs', log_date, log_timestamp)
    if config["dataset"] == 'rwhar':
        label_name = ['climbing_down', 'climbing_up', 'jumping', 'lying',\
                       'running', 'sitting', 'standing', 'walking']
        activity_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    if config["dataset"] == 'wetlab':
        label_name = ['null_class', 'cutting', 'inverting', 'peeling', 'pestling',\
                       'pipetting', 'pouring', 'pour catalysator', 'stirring', 'transfer']
        activity_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # activity_labels = np.array([9])
    # for _, sbj in enumerate(np.unique(data[:, 0])):
    for sbj in [0]:
        # generating training data (validation data -> leave-one-out)
        ml_train_gt_pred = ml_generate_train_data(data, args, sbj)

        best_thrs_for_activity_lst = []
        best_skip_win_for_activity_lst = []
        best_tol_val_for_activity_lst = []
        best_f1_for_activity_lst = []
        f_one_target_lst = []
        best_data_saved_for_activity_lst = []
        best_comp_saved_for_activity_lst = []
        best_loss_for_activity_lst = []
        elapsed_time_lst = []
        acitivity_name_lst = []
        f_one_gt_val_lst = []
        f_one_gt_mod_val_lst = []
        f_one_val_mod_val_lst = []
        f_one_gt_val_avg_lst = []
        f_one_gt_mod_val_avg_lst = []
        
        # activity that should be used 
        for labels in activity_labels:
            activity = labels
            activity_name = label_name[activity]
            # get the start time
            start_time = time.time()
            best_h_param, loss, best_f1, f_one_target, best_comp_saved, best_data_saved,\
                f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val,\
                        f_one_gt_mod_val_avg, f_one_gt_val_avg=\
                                genetic_algorithm(activity, args, ml_train_gt_pred, bounds, n_bits,  termin_iter, max_iter, n_pop, r_cross, r_mut)
            # get the end time
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"Done! for activity: {label_name[activity]}")
            print(f"Optimum window threshold is: {best_h_param[0]} \n", 
                        f"Optimum skip windows is: {best_h_param[1]} \n",
                        f"Optimum tolerance value is: {best_h_param[2]} \n",
                        f"F1 score (for particular activity) at optimum hyperparameter: {best_f1} \n",
                        f"Avg. modified f1 score (for all activities) at optimum hyperparameter: {f_one_gt_mod_val_avg} target was {1} \n",
                        # f"Data saved at optimum hyperparameter: {best_data_saved} \n",
                        f"Computation saved at optimum hyperparameter: {best_comp_saved} \n",
                        f"Lowest loss value : {loss} \n",
                        f"Total time elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} \n\n")  
            print(f"optimum time after which device will be switched off: {window_to_time(best_h_param[0], config)} seconds")
            print(f" optimum switch off duration: {window_to_time(best_h_param[1], config)} seconds")

            # appending best settings
            best_thrs_for_activity_lst.append(float(best_h_param[0]))
            best_skip_win_for_activity_lst.append(float(best_h_param[1]))
            best_tol_val_for_activity_lst.append(float(best_h_param[2]))
            best_f1_for_activity_lst.append(np.array([best_f1]))
            f_one_target_lst.append(f_one_target)
            best_data_saved_for_activity_lst.append(best_data_saved)
            best_comp_saved_for_activity_lst.append(best_comp_saved)
            best_loss_for_activity_lst.append(loss)
            acitivity_name_lst.append(activity_name)
            # get the execution time
            elapsed_time_lst.append(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            f_one_gt_val_lst.append(f_one_gt_val)
            f_one_gt_mod_val_lst.append(f_one_gt_mod_val)
            f_one_val_mod_val_lst.append(f_one_val_mod_val)
            f_one_gt_mod_val_avg_lst.append(f_one_gt_mod_val_avg)
            f_one_gt_val_avg_lst.append(f_one_gt_val_avg)
        algo_name = 'GA'
        filename_best_csv = activity_save_best_results_to_csv(best_thrs_for_activity_lst, 
                                    best_skip_win_for_activity_lst,
                                    best_tol_val_for_activity_lst,
                                    best_f1_for_activity_lst, f_one_target_lst, best_data_saved_for_activity_lst, 
                                    best_comp_saved_for_activity_lst,
                                    best_loss_for_activity_lst, elapsed_time_lst, acitivity_name_lst,f_one_gt_mod_val_lst,
                                    f_one_gt_val_lst, f_one_val_mod_val_lst, f_one_gt_mod_val_avg_lst, f_one_gt_val_avg_lst, log_dir, args, algo_name, sbj)
        
    return None
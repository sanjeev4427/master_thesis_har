import numpy as np
from SA_model.simulated_annealing import loss_function
from skip_heuristics_scripts.skip_heuristics import skip_heuristics


def objective_ga(activity, args, h_param, all_mod_eval_output):
      
        window_threshold = h_param[0]
        skip_windows = h_param[1]
        tolerance_value = int(h_param[2])
        # tolerance_value = tol_value
        f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val, f_one_gt_mod_val_avg, f_one_gt_val_avg, \
        comp_saved_ratio, data_saved_ratio, \
        computations_saved, data_saved, \
        comp_windows, data_windows,_ = skip_heuristics(activity, args, window_threshold, skip_windows, tolerance_value, all_mod_eval_output)

        # for activity wise optimization
        # f_one, comp_saved_ratio, data_saved_ratio \
        #     = skip_heuristics(activity, args, window_threshold, skip_windows, tolerance_value, all_mod_eval_output)  

        # print(computations_saved/comp_windows*100)
        # print(data_saved, "\n", data_windows)

        #>> uncomment for combined data
        # # defining loss function
        # if comp_saved_ratio > min_comp_saving_desired_percent:
        #     if np.abs(np.var(comp_windows) - np.var(computations_saved)) > 10:
        #         loss = np.abs(1-f_one) + 2*np.abs(1-comp_saved_ratio*0.01) 
        #     else:
        #         loss = np.abs(1-f_one) + 2*np.abs(1-comp_saved_ratio*0.01)
        # elif comp_saved_ratio <= min_comp_saving_desired_percent:
        #     if np.abs(np.var(comp_windows) - np.var(computations_saved)) > 10:
        #         loss = np.abs(1-f_one) + 2*np.abs(1-comp_saved_ratio*0.01) + 5
        #     else:
        #         loss = np.abs(1-f_one) + 2*np.abs(1-comp_saved_ratio*0.01) + 5

        f_alpha = 10
        c_alpha = 1
        d_alpha = 2

        #set f1 target
        f_one_target = 1
        f_one = f_one_gt_mod_val


        #defining computation saved calculation
        lam = args.sw_overlap /100
        comp_saved_ratio = skip_windows/(window_threshold + skip_windows)*100
        data_saved_ratio = 100*(skip_windows - lam*(skip_windows+1))/(skip_windows - lam*(skip_windows+1) + window_threshold + (window_threshold-1)*(1 - lam))

        loss = loss_function(f_alpha, c_alpha, d_alpha, f_one, f_one_target, f_one_gt_val, f_one_gt_mod_val_avg, comp_saved_ratio, data_saved_ratio)

        return loss, f_one, f_one_target, float(data_saved_ratio), float(comp_saved_ratio), f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val,\
                                                                                        f_one_gt_mod_val_avg, f_one_gt_val_avg
from model.evaluation_simulated_annealing import mod_val_preds_for_plot
from model.find_initial_temperature import find_initial_temperature
import time
from model.simulated_annealing import simulated_annealing, anneal_objective
# from model.save_sim_ann_results import save_best_sim_ann_results_to_csv
from model.save_csv_results import activity_save_best_sim_ann_results_to_csv
def simulated_annealing_all_validation_data(args, window_threshold, skip_windows, max_step_size_win_thr, max_step_size_skip_win,\
                                             tol_value, init_temp, ann_rate_array, all_val_gt_two_times):
    config = vars(args)
    # mod_val_preds_for_plot(DATASET, args, best, tol_value, all_val_gt_two_times)
    # plot_sensordata_and_labels(DATASET, all_val_gt_two_times, all_val_gt_two_times)
    # initial temperature
    # init_temp = find_initial_temperature(data, args, threshold_value, skip_windows, \
    #       max_step_size_skip_win, tol_value, n_iter_init_temp, step_size_thrs, all_val_gt)

    # print(f"number of iterations at each temperature: {n_iter_init_temp}")
    # print(f"Initial temperaures are:  {init_temp}")

    best_thrs_for_ann_rate_lst = []
    best_skip_win_for_ann_rate_lst = []
    best_f1_for_ann_rate_lst = []
    best_data_saved_for_ann_rate_lst = []
    best_comp_saved_for_ann_rate_lst = []
    best_loss_for_ann_rate_lst = []
    elapsed_time_lst = []
    for i in range(len(ann_rate_array)):
          # get the start time
          start_time = time.time()
          # perform the simulated annealing search
          best, loss, best_f1, best_data_saved, best_comp_saved = \
                         simulated_annealing(args, window_threshold, skip_windows, max_step_size_win_thr, max_step_size_skip_win,\
                                             tol_value, init_temp, ann_rate_array[i], all_val_gt_two_times)
          # get the end time
          end_time = time.time()
          elapsed_time = end_time - start_time
          print('Execution time:', elapsed_time, 'seconds')
          print(f'Done! for annealing rate: {ann_rate_array[i]}')
          print(f"Optimum window threshold is: {best[:,0]} \n", 
                    f"Optimum skip windows is: {best[:,1]} \n",
                    f"F1 score at optimum hyperparameter: {best_f1} \n",
                    f"Data saved at optimum hyperparameter: {best_data_saved} \n",
                    f"Computation saved at optimum hyperparameter: {best_comp_saved} \n",
                    f"Lowest loss value : {loss} \n",
                    f"Total time elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} \n\n")

          best_thrs_for_ann_rate_lst.append(float(best[:,0]))
          best_skip_win_for_ann_rate_lst.append(float(best[:,1]))
          best_f1_for_ann_rate_lst.append(best_f1)
          best_data_saved_for_ann_rate_lst.append(best_data_saved)
          best_comp_saved_for_ann_rate_lst.append(best_comp_saved)
          best_loss_for_ann_rate_lst.append(loss)
          # get the execution time
          elapsed_time_lst.append(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        #   save_best_sim_ann_results_to_csv(ann_rate_array[0:i+1], best_thrs_for_ann_rate_lst, 
        #                                 best_skip_win_for_ann_rate_lst,
        #                                 best_f1_for_ann_rate_lst, best_data_saved_for_ann_rate_lst, 
        #                                 best_comp_saved_for_ann_rate_lst,
        #                                 best_loss_for_ann_rate_lst, elapsed_time_lst, config["dataset"])
#     best = np.array([[0.003274766, 388]])	

    def window_to_time(window):
     # number of windows to time equation
     # n is number of windows
     # one window is 1 sec
     one_window_duration = 1 
     t = (window-1)*(1-config["sw_overlap"]/100)*one_window_duration + one_window_duration 
     return t
    
    print(f"optimum time after which device will be switched off: {window_to_time(best[:,0])} seconds")
    print(f" optimum switch off duration: {window_to_time(best[:,1])} seconds")

    return None

import numpy as np
def apply_best_on_data_skip(config, current_activity, filename):
    if config['dataset'] == 'rwhar':
        # method 1: one focussed at a time with skipping applied for every activity during training
        # filename = r"C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\logs\20230326\184107\best_results_for_rwhar_SA_case1_rwhar_improved_tol_zone.csv"
        # method 2: one focussed at a time with skipping not applied for every activity during training + loss function includes f1 avg as well as f2 avg as well
        # filename = r"C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\logs\20230328\124134\best_results_for_rwhar_SA_rwhar_idea_skipping_over_all_other_act_f1_avg_added.csv"
        # method 3: one focussed at a time with skipping applied for every activity during training + loss function includes f1 avg as well as f2 avg as well
        # filename = r"C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\logs\20230328\134038\best_results_for_rwhar_SA_rwhar_case1_with_f1_avg_added_in_loss.csv"
        filename = filename
    elif config['dataset'] == 'wetlab':
        # filename = r"C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\logs\20230326\222003\best_results_for_wetlab_SA_case1_wetlab_improved_tol_zone_better_loss_func.csv"
         filename = filename
    # best = np.loadtxt(filename,skiprows=1, usecols=(1,2,3),delimiter=',').T
    # best = [[1]*8,[1]*8, [2]*8]
    # print(best)
    best = np.loadtxt(filename, skiprows=1, usecols=(1,2,3),delimiter=',').T
    best_threshold = best[0]
    best_win_skip = best[1]
    best_tolerance = best[2]
    window_threshold = best_threshold[int(current_activity)]
    skip_window = best_win_skip[int(current_activity)]
    tolerance_value = best_tolerance[int(current_activity)] 
    return int(window_threshold), int(skip_window), int(tolerance_value)
##################################################
# Skipping data function and tolerance zone function is defined.  
##################################################
# Author: Sanjeev Kumar
# Email: sanjeev.kumar(at)student.uni-siegen.de
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# Author: Michael Moeller
# Email: michael.moeller(at)uni-siegen.de
##################################################

import math
import numpy as np

from skip_heuristics_scripts.apply_best_on_data_skip import apply_best_on_data_skip
def tolerance_zone(
    mod_val_preds_current,
    config,
    window_count,
    last_window,
    tolerance=0,
    tolerance_window_count=1,
    last_tolerance_window=-1
    ):
    """
    Method called when activity changes.

    Args:
        mod_val_preds_current (int): current value of validation prediction
        config (dict): General setting dictionary
        tolerance (int): tolerance threshold used to reset values. Defaults to 0.
        tolerance_window_count (int): activity window count in tolerance zone. Defaults to 0.
        last_tolerance_window (int): stores validation prediction in last loop. Defaults to -1.
        window_count (int): count activity windows for any particular predicted activity. Defaults to 0.

    Returns:
        int: activity windows count for any particular predicted activity.
        int: tolerance threshold used to reset values.
        int: activity window count in tolerance zone.
        int: validation prediction in last loop.

    """
    # increase tolerance counter
    tolerance += 1

    # if current window same as one in previous tolerance loop
    if mod_val_preds_current == last_tolerance_window:
        # increase tolerance window counter
        tolerance_window_count += 1
    else:
        # set last tolerance window to current
        last_tolerance_window = mod_val_preds_current
        # set tolerance window counter to 1
        # tolerance_window_count = 0        
        tolerance_window_count = 1

    # if tolerance window counter is greater than 2, SAVING_TOLERANCE = 2
    # then quit counting of current activity and switch to tolerance activity
    if tolerance_window_count >= (config["saving_tolerance"]):
        # overwrite current count and last window to tolerance values
        window_count = tolerance_window_count
        last_window = last_tolerance_window
        # reset tolerance, tolerance window counter and last tolerance window
        tolerance = 0
        tolerance_window_count = 1
        last_tolerance_window = -1
        
    # else if tolerance greater than tolerance threshold,
    # # then quit counting of current activity and reset counting
    # elif tolerance > config["saving_tolerance"]:
    #     # reset window counter and last window
    #     window_count = 1
    #     last_window = -1
    #     # reset tolerance values
    #     tolerance = 0
    #     # tolerance_window_count = 0
    #     tolerance_window_count = 1
    #     last_tolerance_window = -1
        
    # if tolerance safety measures to do not trigger, treat current window as it was of the
    else:
        window_count += 1
        last_window = last_window
    return window_count, tolerance, tolerance_window_count, last_tolerance_window, last_window
    


# function for skipping validation prediction data
def data_skipping(current_activity, mod_val_preds, config, data_saved, computations_saved, apply_best=False, *args):
    """
    This function implements data skipping based on current activity and window predictions,
    and applies data saving techniques using predefined configuration parameters.

    Args:
    current_activity (int): current activity label
    mod_val_preds (array): window predictions
    config (dict): configuration parameters for data saving and skipping
    data_saved (array): array for data saving calculations for each activity
    computations_saved (array): array for computation saving calculations for each activity
    apply_best (bool, optional): Flag to apply best hyperparameters. Defaults to False.
    *args : optional arguments

    Returns:
    mod_val_preds (ndarray): modified validation prediction
    data_saved (list): data saved for each activity
    computations_saved (list): computation saved for each activity 
    """

    j = 0
    tolerance = 0
    window_count = 1
    # tolerance_window_count = 0
    tolerance_window_count = 1
    last_window = -1
    last_tolerance_window = -1
    # print(f"trained threshold value: {config['saving_threshold']}")
    # overlap adjustment parameter, i.e. how many windows are actually worth a new window worth of data
    overlap_adjustment = 1 / (1 - config["sw_overlap"] * 0.01)

    while j < (len(mod_val_preds)):        
        # if last window is same as current; increase counter
        if mod_val_preds[j] == last_window:
            window_count += 1
            #update the activities within tolerance zone to be current activity, except if tolerance lies at the intersection of two activities
            mod_val_preds[j-tolerance:j] = mod_val_preds[j]
            
            tolerance = 0
            tolerance_window_count = 1
            last_tolerance_window = -1

        elif last_window == -1:
            last_window = mod_val_preds[j]
            
        else:
            if apply_best == True:
                current_activity = mod_val_preds[j - (window_count-1)]
                filename = args[0]
                _,_,tolerance_value = apply_best_on_data_skip(config, current_activity, filename)
                config["saving_tolerance"] = tolerance_value
            (
                window_count,
                tolerance,
                tolerance_window_count,
                last_tolerance_window, 
                last_window
            ) = tolerance_zone(
                mod_val_preds[j],
                config,
                window_count,
                last_window,
                tolerance,
                tolerance_window_count,
                last_tolerance_window,
                
            )

        # obtain avg/min/max activity duration (num of windows without overlap) of current activity
        # saving_array has activity duration of all activities calculated from training data
        # [j - window_count] gives the index of first element of new activity after the tol zone
        # activity_duration = config["saving_array"][mod_val_preds[j - window_count]]

        # calculate threshold of how many consecutive windows of said activity must be seen
        # (save_v * threshold - first window) * overlap adjustment
        # ? how this is calculated?

        # window_threshold = (
        #     (activity_duration * config["saving_threshold"]) - config["sw_length"]
        # ) * overlap_adjustment
        #print(config["saving_threshold"])
        
        # apply best settings for skip
        
        if apply_best == True:
            filename = args[0]
            current_activity = mod_val_preds[j - (window_count-1)]
            window_threshold,window_skip,_ = apply_best_on_data_skip(config, current_activity,filename) 
        else:
            window_threshold = int(config["saving_window_threshold"])
            window_skip = int(config["saving_skip_windows"])

        # computation saved 
        lam = config["sw_overlap"] /100
        computations_saved[int(mod_val_preds[j - (window_count-1)])] = window_skip/(window_skip + window_threshold)*100
        data_saved[int(mod_val_preds[j - (window_count-1)])] = 100*(window_skip - lam*(window_skip+1))/(window_skip - lam*(window_skip+1) + window_threshold + (window_threshold-1)*(1 - lam))
        if window_threshold < 0:
            window_threshold = 0
            activity_windows = 2
        # else:
            # probably this is total activity windows with overlap_adjustment
            # ? how this is calculated?

            # activity_windows = int(
            #     math.ceil(
            #         (activity_duration - config["sw_length"]) * overlap_adjustment
            #     )
            # )
        # skip windows search space: current activity maximum window count minus current activity window already counted
    
       
        # calculate how many windows will be skipped if count is above threshold
        # used only for actovity wise data skipping
        # when training firs condition is checked and for getting results from trained hyparams apply_best must be true
        if mod_val_preds[j - (window_count-1)] == current_activity or apply_best:
            if window_count > window_threshold:            
                #! hypara for simulated annealing skip_windows updated here 
                # window_skip_search_space_upper_limit = count_array[mod_val_preds[j - window_count]] - window_threshold
                # config["saving_skip_windows"] = int(np.random.uniform(0, window_skip_search_space_upper_limit))
                # throw away all windows from this point till the window skip
                # window skip is learnable parameter 
                # window_skip = activity_windows - window_count

                
                if window_skip < 0:  # -1 in sbj 1 rwhr thrs val= 1
                    window_skip = 0

                if j + window_skip >= len(mod_val_preds):
                    # mod_val_preds[j - window_count :] = mod_val_preds[j - window_count]
                    mod_val_preds[j - (window_count-1) :] = mod_val_preds[j - (window_count-1)]
                    window_skip = len(mod_val_preds[j : ])
                else:
                    # from starting point to the end of window skip fill with activity at the star of current activity
                    # mod_val_preds[j - window_count : j + window_skip + 1] = mod_val_preds[j - window_count]
                    mod_val_preds[j - (window_count-1) : j + window_skip] = mod_val_preds[j - (window_count-1)]
                # data saved = num of wind to skip*length of window*non-overlap win
                # ? I think it misses to count the overlap data which is saved as well but if it window skip already
                # ? ...consider the overlap then there should be no overlap term here

                # data_saved[int(mod_val_preds[j - (window_count-1)])] += (
                #     window_skip * (1 - config["sw_overlap"] * 0.01) * config["sw_length"]
                # )
                # saved computation simply count the number of window skip
                # saved computations are saved in class order
                # computations_saved[int(mod_val_preds[j - (window_count-1)])] += window_skip
                
                # reset window counter and last window
                window_count = 1
                last_window = -1
                # reset tolerance values
                tolerance = 0
                tolerance_window_count = 1
                last_tolerance_window = -1

                # increase j + what was skipped
                j += window_skip
            else:
                # increase j
                j += 1
        else:
            # increase j
            j += 1
    return mod_val_preds, data_saved, computations_saved

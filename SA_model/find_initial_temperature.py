
import numpy as np
from SA_model.generate_input_data_for_SA import ml_generate_train_data
# from SA_model.generate_input_data_for_SA import ml_generate_train_data 
# from model.plotting import plot_finding_initial_temp, \
                    # plot_acceptance_ratio_finding_initial_temp
from SA_model.simulated_annealing import anneal_objective

def generate_acceptance_ratio(temp, activity, args, tol_value, all_mod_eval_output, window_threshold, skip_windows,\
    # sourcery skip: low-code-quality, min-max-identity
                               max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val, n_iter):

    # intialising hyperparameters
    best_window_threshold = np.random.randint(window_threshold[0], window_threshold[1]+1)

    # best_tol_value = tol_value[0] + \
    #                     np.random.rand(1)*(tol_value[1] - tol_value[0])

    best_skip_windows = np.random.randint(skip_windows[0], skip_windows[1]+1)

    best_tol_value = np.random.randint(tol_value[0], tol_value[1]+1)

    best = np.array([best_window_threshold, best_skip_windows, best_tol_value])
    best = best.reshape(1,3)
    best_eval, best_f1, _, best_data_saved, best_comp_saved,_,_,_,_,_ = anneal_objective(activity, args, best, all_mod_eval_output)

    
                             
    curr, curr_eval = np.copy(best), best_eval
    curr_f1, curr_comp_saved, curr_data_saved = best_f1, best_comp_saved, best_data_saved
    candidate = np.zeros((1,3))
    loss_array = []
    threshold_value_array = []
    skip_windows_array = []
    fscore_array = []
    data_saved_array = []
    comp_saved_array = []
    accepted_states = 0
    rejected_states = 0

    #run the algorithm
    #? may use a termination condition = f1_non-skipped - f1_skipped = 0
    for _ in range(n_iter):
        # print("-"*20)
        # print(f"iteration :   {i+1} of {n_iter}: ", '\n'*2)
        # take a step #? should I take a step in negative direction
        #? should I try other distributions: gauss 
        candidate[:,0] = curr[:,0] + np.random.randint(-max_step_size_win_thr, max_step_size_win_thr + 1)
        candidate[:,1] = curr[:,1] + np.random.randint(-max_step_size_skip_win, max_step_size_skip_win + 1)
        candidate[:,2] = curr[:,2] + np.random.randint(-max_step_size_tol_val, max_step_size_tol_val + 1)
        # candidate[:,0] = np.random.uniform(0,1)
        # candidate[:,1] = np.random.randint(-skip_windows[0], skip_windows[1]+1)
        # for negative values make it positive # +1 for zero value 
        if candidate[:,0] <= 0:
            candidate[:,0] = np.abs(candidate[:,0]) + 1
        # if crossing upper bound, make it equal to upper bound    
        if candidate[:,0] > window_threshold[1]:
            candidate[:,0] = window_threshold[1]
        # for negative values make it positive # +1 for zero value
        if candidate[:,1] <= 0:
            candidate[:,1] = np.abs(candidate[:,1]) + 1
        # if crossing upper bound, make it equal to upper bound
        if candidate[:,1] > skip_windows[1]:
            candidate[:,1] = skip_windows[1]
        # for negative values make it positive # +1 for zero value
        if candidate[:,2] <= 0:
            candidate[:,2] = np.abs(candidate[:,2]) + 1
        # if crossing upper bound, make it equal to upper bound
        if candidate[:,2] > tol_value[1]:
            candidate[:,2] = tol_value[1]
        # evaluate candidate point
        # minimize loss function returned 
        candidate_eval, candidate_f1, _, candidate_data_saved, candidate_comp_saved,_,_,_,_,_\
                                        = anneal_objective(activity, args,candidate, all_mod_eval_output)
        
        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval 

        # calculate metropolis acceptance criterion
        metropolis = np.exp(-diff / temp)
        print("diff : ", diff)
        print("metropolis: ", metropolis)

        # check if we should keep the new point
        if diff <=0:
            curr, curr_eval = candidate, candidate_eval
            accepted_states += 1
            print('accepted')
        elif np.random.rand() <= metropolis: # compared when diff is more than zero
            # store the new current point
            curr, curr_eval = candidate, candidate_eval
            accepted_states += 1
            print('accepted')
        else:
            rejected_states += 1
            print('rejected')
    print("accepted states: ", accepted_states)
    print("rejected states: ", rejected_states)
    acceptance_ratio = accepted_states / (accepted_states + rejected_states)
    print(f"acceptance ratio: {acceptance_ratio}")

    return acceptance_ratio

def find_initial_temperature(data, args, tol_value,\
                                window_threshold, skip_windows,
    # sourcery skip: use-fstring-for-formatting
                                max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val, n_iter):
    config= vars(args)
    print("Warming system... (Finding initial temperature)")
    print("initializing parameters...")
    initial_temp_activity = []
    sbj = 0
    all_mod_eval_output = ml_generate_train_data(data, args, sbj)
    for activity in [1]:
        # to ensure same starting temperature
        np.random.seed(2)
        # random initial temperature
        # divide according to see difference and equate to 0.5 for eg to find approx init temp
        temp = np.random.rand() #! divide by 1000 works
        print(f"initial temperature: {temp}")
        # for staritng up 
        acceptance_ratio = generate_acceptance_ratio(temp, activity, args, tol_value,\
                                                    all_mod_eval_output, window_threshold, skip_windows,
                                                    max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val, n_iter)
        # for plotting at each level of temperature

        acceptance_ratio_array = []
        temperature_array = []
        while acceptance_ratio < 0.80:
            # storing temperature and acceptance ratio values
            acceptance_ratio_array.append(acceptance_ratio)
            temperature_array.append(temp)
            # double the temperature
            temp = temp * 2
            acceptance_ratio = generate_acceptance_ratio(temp, activity, args, tol_value,\
                                        all_mod_eval_output, window_threshold, skip_windows,
                                            max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val, n_iter) 
            print("temperature: {}".format(temp), '\n'*2)

        # storing last temperature and acceptance ratio values
        acceptance_ratio_array.append(acceptance_ratio)
        temperature_array.append(temp)
        # store initial temperature for each activity
        initial_temp_activity.append(temp)
        print ("Initial temperature should be: {}".format(temp))
            # plot_finding_initial_temp(temperature_array, config, activity[i])
            # plot_acceptance_ratio_finding_initial_temp(acceptance_ratio_array, config, activity[i])
    return initial_temp_activity 
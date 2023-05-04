# genetic algorithm search of the one max optimization problem
from numpy.random import randint
from GA_model.crossover import crossover
from GA_model.decode import decode
from GA_model.mutation import mutation
from GA_model.objective_ga import objective_ga
from GA_model.plotting_ga import activity_plot_comp_saved_gen, activity_plot_f1_gen, activity_plot_loss_ga, activity_plot_skip_windows_gen, activity_plot_threshold_gen
from GA_model.selection import selection
from log_data_scripts.save_csv_results import activity_save_best_results_to_csv, activity_save_ga_results_to_csv


# genetic algorithm
def genetic_algorithm(activity, args, all_mod_eval_output, bounds, n_bits, termin_iter, max_iter, n_pop, r_cross, r_mut):
	config = vars(args)

	# initial population of random bitstring
	pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
	# decoding of bitstring
	h_param = decode(bounds, n_bits, pop[0])
	score, f_one, f_one_target,\
		data_saved_ratio, comp_saved_ratio,\
				f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val,\
				f_one_gt_mod_val_avg, f_one_gt_val_avg = objective_ga(activity, args, h_param, all_mod_eval_output)
	# keep track of best solution
	best, best_eval = 0, score
	win_thrs_list = list([h_param[0]])
	skip_win_list = list([h_param[1]])
	tol_value_list = list([h_param[2]])
	best_loss_list = list([best_eval])
	best_f1_list = list([f_one])
	best_comp_saved_ratio_list = list([comp_saved_ratio])
	best_gen_list = list([0])
	best_f1, best_comp_saved, best_data_saved, best_h_param = f_one, comp_saved_ratio, data_saved_ratio, h_param
	best_f_one_gt_mod_val_avg, best_f_one_gt_val_avg = f_one_gt_mod_val_avg, f_one_gt_val_avg
	# enumerate generations
	gen = 0
	termin_count = 0
	while termin_count < termin_iter:
		# counting generations
		gen += 1 
		print(f"Genration: {gen}...")
		print(f"Termination count: {termin_count} (max: {termin_iter})")
		# increase termination counter
		termin_count += 1
		
		# to make sure the algorithm is not stuck in perpetual iteration
		if gen > max_iter:
			break

		# decode population
		h_param_decoded = [decode(bounds, n_bits, p) for p in pop]
		# evaluate all candidates in the population
		scores_pop = list()
		f1_pop = list()
		comp_saved_ratio_pop = list()
		data_saved_ratio_pop = list()
		for h_param in h_param_decoded:
			score, f_one, f_one_target,\
				data_saved_ratio, comp_saved_ratio,\
					f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val,\
						f_one_gt_mod_val_avg, f_one_gt_val_avg = objective_ga(activity, args, h_param, all_mod_eval_output)
			scores_pop.append(score)	
			f1_pop.append(f_one)		
			comp_saved_ratio_pop.append(comp_saved_ratio)
			data_saved_ratio_pop.append(data_saved_ratio)
		# check for new best solution
		for i in range(n_pop):

			if scores_pop[i] < best_eval:
				#setting termin count to zero as lower loss found
				termin_count = 0 
				best, best_eval = pop[i], scores_pop[i]

				best_f1, best_comp_saved, best_data_saved, best_h_param \
					 	 = f1_pop[i], comp_saved_ratio_pop[i], data_saved_ratio_pop[i], h_param_decoded[i]
				
				best_f_one_gt_mod_val_avg, best_f_one_gt_val_avg = f_one_gt_mod_val_avg, f_one_gt_val_avg
				print("*"*10)
				print(f"Best at gen: {gen} \n",
	                    f"new best loss: {scores_pop[i]} at: {h_param_decoded[i]} \n",
			                f"new best f1: {f1_pop[i]} \n",
			                    f"new best comp saved: {comp_saved_ratio_pop[i]}")
				print("*"*10)
				
				win_thrs_list.append(h_param_decoded[i][0])
				skip_win_list.append(h_param_decoded[i][1])
				tol_value_list.append(h_param_decoded[i][2])
				best_loss_list.append(scores_pop[i])
				best_f1_list.append(f1_pop[i])
				best_comp_saved_ratio_list.append(comp_saved_ratio_pop[i])
				best_gen_list.append(gen)      
		
		# select parents
        # tournament is run n_pop times to select n_pop best
		selected = [selection(pop, scores_pop) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
		
	# activity_plot_loss_ga(best_loss_list, best_gen_list, config, activity_name)
	# activity_plot_f1_gen(best_f1_list, best_gen_list, config, activity_name)
	# activity_plot_comp_saved_gen(best_comp_saved_ratio_list, best_gen_list, config, activity_name)
	# activity_plot_threshold_gen(win_thrs_list, best_gen_list, config, activity_name)
	# activity_plot_skip_windows_gen(skip_win_list, best_gen_list, config, activity_name)	
	# activity_save_ga_results_to_csv(best_loss_list, win_thrs_list, skip_win_list, best_f1_list, 
    #                                             best_comp_saved_ratio_list, best_comp_saved_ratio_list, config, activity_name)
	
	return best_h_param, best_eval[0], best_f1[0], f_one_target, round(best_comp_saved,2), best_data_saved, f_one_gt_mod_val[0],  f_one_gt_val[0], f_one_val_mod_val[0],\
																							best_f_one_gt_mod_val_avg, best_f_one_gt_val_avg


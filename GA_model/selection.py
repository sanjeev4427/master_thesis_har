from numpy.random import randint

# tournament selection
#? selection pressure: which is a probabilistic measure of a candidate’s likelihood of participation in a tournament
# More the selection pressure more will be the Convergence rate.
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	# generate 2 random inx and compare them with third random index, lowest among three is the winner
	for ix in randint(0, len(pop), k-1): # k-1 numbers between 0 and 99
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]





# import multiprocessing

# def tournament_selection_worker(pop, scores, k, start, end, result_queue):
#     # Tournament selection on a subset of the population
#     selection_ix = randint(start, end)  
#     for ix in randint(start, end, k-1):
#         if scores[ix] < scores[selection_ix]:
#             selection_ix = ix
#     result_queue.put(pop[selection_ix])

# def selection(pop, scores, k=3, num_processes=None):
#     if num_processes is None:
#         num_processes = multiprocessing.cpu_count()

#     chunk_size = len(pop) // num_processes

#     # Create processes and run tournament selection on each chunk of the population
#     processes = []
#     result_queue = multiprocessing.Queue()
#     for i in range(num_processes):
#         start = i * chunk_size
#         end = (i+1) * chunk_size if i < num_processes - 1 else len(pop)
#         p = multiprocessing.Process(target=tournament_selection_worker,
#                                     args=(pop, scores, k, start, end, result_queue))
#         p.start()
#         processes.append(p)

#     # Collect results from processes
#     results = []
#     for i in range(num_processes):
#         results.append(result_queue.get())

#     # Wait for all processes to finish
#     for p in processes:
#         p.join()

#     # Return the best individual from the tournament
#     return min(results, key=lambda x: scores[pop.index(x)])





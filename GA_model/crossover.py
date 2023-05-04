# crossover two parents to create two children
from numpy.random import randint,rand

# two point crossover
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select two crossover points that are not on the end of the string
        pt1, pt2 = sorted(randint(1, len(p1)-2, 2))
        # perform crossover
        c1[pt1:pt2], c2[pt1:pt2] = p2[pt1:pt2], p1[pt1:pt2]
    return [c1, c2]

# #single point crossover
# def crossover(p1, p2, r_cross):
# 	# children are copies of parents by default
# 	c1, c2 = p1.copy(), p2.copy()
# 	# check for recombination
# 	if rand() < r_cross:
# 		# select crossover point that is not on the end of the string
# 		pt = randint(1, len(p1)-2)
# 		# perform crossover
# 		c1 = p1[:pt] + p2[pt:]
# 		c2 = p2[:pt] + p1[pt:]
# 	return [c1, c2]



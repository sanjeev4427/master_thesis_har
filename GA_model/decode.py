
# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits - 1 # largest number represented by n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
        # bounds[i][0]-1: since lower bound is 1, we want it to be zero in this calculation
        # for 7 bit, value is 1 to 128 as integer is 0 to 127
		value = int(bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])) 
		# store
		decoded.append(value)
	return decoded
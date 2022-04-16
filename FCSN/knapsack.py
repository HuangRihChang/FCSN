# This file was found online, but I am sorry that I don’t know who the original author is now.
# http://www.geeksforgeeks.org/knapsack-problem/

import numpy as np

def knapsack(v, w, max_weight):
    rows = len(v) + 1
    cols = max_weight + 1

    # adding dummy values as later on we consider these values as indexed from 1 for convinence
    v = np.r_[[0], v]
    w = np.r_[[0], w]
    # row : values , #col : weights
    dp_array = [[0 for i in range(cols)] for j in range(rows)]
    # 0th row and 0th column have value 0
    # values
    for i in range(1, rows):
        # weights
        for j in range(1, cols):
            # if this weight exceeds max_weight at that point
            if j - w[i] < 0:
                dp_array[i][j] = dp_array[i - 1][j]
            # max of -> last ele taken | this ele taken + max of previous values possible
            else:
                dp_array[i][j] = max(dp_array[i - 1][j], v[i] + dp_array[i - 1][j - w[i]])
    # return dp_array[rows][cols]  : will have the max value possible for given wieghts
    chosen = []
    i = rows - 1
    j = cols - 1

    # Get the items to be picked
    while i > 0 and j > 0:
        # ith element is added
        if dp_array[i][j] != dp_array[i - 1][j]:
            # add the value
            chosen.append(i-1)
            # decrease the weight possible (j)
            j = j - w[i]
            # go to previous row
            i = i - 1
        else:
            i = i - 1
    return chosen

def knapSack_improve(W, wt, val, n):
	""" Maximize the value that a knapsack of capacity W can hold. You can either put the item or discard it, there is
	no concept of putting some part of item in the knapsack.

	:param int W: Maximum capacity -in frames- of the knapsack.
	:param list[int] wt: The weights (lengths -in frames-) of each video shot.
	:param list[float] val: The values (importance scores) of each video shot.
	:param int n: The number of the shots.
	:return: A list containing the indices of the selected shots.
	"""
	K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

	# Build table K[][] in bottom up manner
	for i in range(n + 1):
		for w in range(W + 1):
			if i == 0 or w == 0:
				K[i][w] = 0
			elif wt[i - 1] <= w:
				K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
			else:
				K[i][w] = K[i - 1][w]

	selected = []
	w = W
	for i in range(n, 0, -1):
		if K[i][w] != K[i - 1][w]:
			selected.insert(0, i - 1)
			w -= wt[i - 1]
	return selected

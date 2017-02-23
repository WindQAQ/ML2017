import sys
import numpy as np

A = np.loadtxt(sys.argv[1], delimiter=',')
B = np.loadtxt(sys.argv[2], delimiter=',')

M = np.matmul(A, B)

with open('ans_one.txt', 'w') as f:
	for n in sorted(M.flatten()):
		print(int(n), file=f)
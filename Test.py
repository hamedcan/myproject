import numpy as np

np.ones
r = [[1,1],[0,0]]
p = [[1,0],[1,0]]

print(np.add(r,p))
print(np.count_nonzero(np.add(r,p) == 1))
print(np.multiply(r,p))
print(np.count_nonzero(np.multiply(r,p)))


dice = []
dice.append(2)
dice.append(1)
dice.append(6)
print(np.average(dice))


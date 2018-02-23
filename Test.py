import numpy as np

h = np.ones((2,5,6))

b = h[:,:,4:5]
print(b.shape)
import numpy as np
import matplotlib.pyplot as plt

random = np.random.normal(0,1,size=[100,100])
plt.imshow(random,aspect="auto")
plt.show()
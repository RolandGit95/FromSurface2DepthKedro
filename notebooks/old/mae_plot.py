import os
import numpy as np
import matplotlib.pyplot as plt
import glob

files = glob.glob('../data/08_validation/regimeA/*.npy')

# +
data = dict()

for file in files:
    key = os.path.splitext(os.path.basename(file))[0]
    values = np.load(file)
    data[key] = values
# -

for key,values in data.items():
    plt.plot(values, label=key)
plt.show()



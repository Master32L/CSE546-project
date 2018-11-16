import pickle
import matplotlib.pyplot as plt
import numpy as np

#%%

with open('log.pkl', mode='rb') as f:
    log = pickle.load(f)

#%%

x = np.arange(len(log['training_loss'])) + 1

#%%

plt.plot(x, log['training_loss'], c='r', linewidth=2)
plt.plot(x, log['test_loss'], c='y', linewidth=2)
plt.xlabel('epoch')
plt.ylabel('mean squared error')
plt.legend(['training loss', 'test loss'])
plt.show()

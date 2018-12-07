import pickle
import matplotlib.pyplot as plt
import numpy as np

#%%

VERSION = '1206-4'
with open(f'./result/{VERSION}/log.pkl', mode='rb') as f:
    log = pickle.load(f)
print(np.min(log['test loss']))

#%%

x = np.arange(len(log['training loss'])) + 1

#%%

plt.plot(x, log['training loss'], c='r', linewidth=2)
plt.plot(x, log['test loss'], c='y', linewidth=2)
plt.xlabel('epoch')
plt.ylabel('mean squared error')
plt.legend(['train', 'test'])
plt.title('Mean squared error vs epoch')
#plt.show()
plt.savefig(f'./result/{VERSION}/curve.png')

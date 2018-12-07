import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%%

VERSION = '1206-fix'
with open(f'search-{VERSION}.pkl', mode='rb') as f:
    log = pickle.load(f)
print(np.min([np.min(l) for l in log[1]]))

#%%

df = pd.DataFrame({'train mse': [np.min(l) for l in log[0]],
                   'test mse': [np.min(l) for l in log[1]],
                   'feature dim': log[2], 'weight decay': log[3]})
df.to_csv(f'search-{VERSION}.csv', index=False)

#%%

opt_trial = np.argmin([np.min(l) for l in log[0]])
x = np.arange(len(log[1][opt_trial])) + 1

plt.plot(x, log[0][opt_trial], c='r', linewidth=2)
plt.plot(x, log[1][opt_trial], c='y', linewidth=2)
plt.xlabel('epoch')
plt.ylabel('mean squared error')
plt.legend(['train', 'test'])
plt.title('Mean squared error vs epoch')
#plt.show()
plt.savefig(f'search-{VERSION}.png')

#%%

#VERSION = '1206'
#with open(f'search-{VERSION}.pkl', mode='rb') as f:
#    log = pickle.load(f)
#
#feature_dim = [256, 128, 64, 256, 64, 64, 128, 32, 256, 16, 64, 32]
#weight_decay = [0.00028469611926819837, 2.5320274524027227e-05,
#                0.0036202248137276794, 1.0971634242389526e-05,
#                1.9683437677534807e-05, 0.08862949571024872,
#                7.12407234761956e-05, 0.0009358367251291517,
#                4.8750772678984874e-05, 0.00011101205062393785,
#                0.01674802094329846, 0.00019625812280088626]
#
#with open(f'search-{VERSION}-fix.pkl', mode='wb') as f:
#    pickle.dump([log[0], log[1], feature_dim, weight_decay], f)

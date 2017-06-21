resting_state = labels == 'rest'
Xtask = X[~resting_state]
ytask = labels[np.logical_not(resting_state)]


import numpy as np
import pandas as pd
X = np.load('preds.npy')
img = pd.read_csv('test.csv')
img['x1'] = X[:,0]*640
img['x2'] = X[:,1]*640
img['y1'] = X[:,2]*480
img['y2'] = X[:,3]*480
""" img['x1'] = 0.05*640
img['x2'] = 0.95*640
img['y1'] = 0.05*480
img['y2'] = 0.95*480 """
img.to_csv('subbigles.csv',index = False)
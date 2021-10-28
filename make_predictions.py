import numpy as np
import pandas as pd
from joblib import load
from plot_modules import plot_predictions

""" read catalog and use features """
features_list = ['R+', 'T+', 'dm+', 'n_parent', 'n_child']
percent = 50
df = pd.read_csv('test_catalogs/mld_Hauksson_1981-2019.csv', index_col=0)
X = df[features_list]

""" prediction """
# -- load classifier
model = load('models/RF_1P_5_{}.sav'.format(percent))
# -- get class probabilities
prob = model.predict_proba(X)[:, 1]

""" plot predictions """
plot_predictions(df=df, prob=prob, figure_name='predictions.png', dt=15)

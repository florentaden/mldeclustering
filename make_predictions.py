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
model = load(f'models/RF_1P_5_{percent}.sav')

# -- get class probabilities
probs = model.predict_proba(X)[:, 1]

# -- save labelled catalog
new_df = df.copy()
new_df['probs'] = probs
new_df['label'] = np.round(probs)
new_df.to_csv('test_catalogs/labelled_mld_Hauksson_1981-2019.csv', index=True)

""" plot predictions """
plot_predictions(df=df, probs=probs, figure_name='predictions.png', dt=15)

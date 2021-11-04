import numpy as np
import pandas as pd
from time import time
from multiprocessing import cpu_count, Pool
from collections import Counter

""" read catalog """
input_name = 'Hauksson_1981-2019.csv'
output_name = 'mld_' + input_name

# -- !parameters to estimate beforehand!
fractal_dimension=1.6
w=1.3
completeness_magnitude = 2.3

df = pd.read_csv('test_catalogs/' + input_name) # read
df = df[df.magnitude >= completeness_magnitude] # keep only above mc
df.reset_index(inplace=True, drop=True)
new_df = df[1:].copy() # N-1 events will have a neighbor

""" functions needed further in the code """
def nnd(event):
    """ identify the nearest neighbor of the given event in the parent catalog
        and return the following features:
             i+: index of the nearest neighbor (nn);
             N+: nearest-neighbor distance between the event and its nn;
             T+: rescaled time;
             R+: rescaled distance (!epicentral by default!);
            dm+: difference in magnitude between the event and its nn.
    """
    i = event.name
    neighbors = df[df.time < event.time].copy() # neighbors always come before

    # -- rescaled time
    neighbors['T+'] = (event.time - neighbors.time)*np.power(
        10, -0.5*w*neighbors.magnitude)

    # -- rescaled distance (epicentral!)
    neighbors['R+'] = np.power(np.sqrt(
        np.power(event.latitude-neighbors.latitude, 2) + \
        np.power(event.longitude-neighbors.longitude, 2)),
        fractal_dimension)*np.power(10, -0.5*w*neighbors.magnitude)

    # -- magnitude difference
    neighbors['dm+'] = neighbors.magnitude - event.magnitude

    # -- nearest-neighbor distance
    neighbors['N+'] = neighbors['T+']*neighbors['R+']

    # -- find the nearest neighbor which minimize eta
    nn = neighbors.loc[neighbors['N+'].idxmin()]

    # -- update dataframe
    return nn.name, nn['N+'], nn['T+'], nn['R+'], nn['dm+']

def count_off_sib(event):
    """ return the remaining features based on the identification of the
        nearest neighbors and the first set of features:
            nc: number of children for the given event i.e. the number of time
                the event has been chosen as a nearest neighbor;
            ns: number of siblings for the given event i.e. the number of events
                sharing the same nearest neighbor.
    """
    # -- number of offspring
    if event.name not in counts.keys():
        n_children = 0
    else:
        n_children = counts[event.name]

    # -- number of siblings
    n_siblings = counts[event['i+']]
    return n_children, n_siblings

""" identify the nearest neighbor (nn) for each event """
to = time() # start the clock
ncpu = cpu_count() # feel free to change to a constant, the higher the faster
if ncpu == 1:
    output = pd.DataFrame(
        map(nnd, [child for _, child in new_df.iterrows()]),
        columns=['i+', 'N+', 'T+', 'R+', 'dm+'])
else:
    with Pool(ncpu) as pool: # running in // is necessary for large catalogs
        output = pd.DataFrame(
            pool.map(nnd, [child for _, child in new_df.iterrows()]),
            columns=['i+', 'N+', 'T+', 'R+', 'dm+']) # the feature dataframe
output.index += 1
new_df = pd.concat([new_df, output], axis=1) # update catalog

""" estimate the number of siblings and children """
counts = Counter(new_df['i+']) # count the # of time an event as been the nn
if ncpu == 1:
    output = pd.DataFrame(
        map(count_off_sib, [child for _, child in new_df.iterrows()]),
        columns=['n_child', 'n_parent']) # the dataframe containing the remaining features
else:
    with Pool(ncpu) as pool:
        output = pd.DataFrame(
            pool.map(count_off_sib, [child for _, child in new_df.iterrows()]),
            columns=['n_child', 'n_parent']) # the dataframe containing the remaining features
output.index += 1

""" write updated catalog ready to be declustered """
new_df = pd.concat([new_df, output], axis=1)
new_df['n_parent'] = new_df['n_parent']/(len(new_df)/new_df.time.max())
new_df['n_child'] = new_df['n_child']/(len(new_df)/new_df.time.max())

new_df.to_csv('test_catalogs/' + output_name, index=False)
print('done in {:.03f}min'.format((time()-to)/60.))

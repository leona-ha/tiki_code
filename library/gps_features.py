import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import statistics

def haversine(lon1, lat1, lon2, lat2):
    # Haversine formula to calculate distance
    pass

def db2(x, epsilon, min_samples):
    clustering_model = DBSCAN(eps=epsilon, min_samples=min_samples, metric="haversine")
    cluster_labels = clustering_model.fit_predict(x[['Longitude', 'Latitude']].apply(np.radians))
    return pd.Series(cluster_labels, index=x.index)

def identify_home(df):
    df['home'] = df.groupby('customer')['clusterID'].transform(lambda x: statistics.mode(x))
    return df
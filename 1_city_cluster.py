
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot

from numpy import unique
from numpy import where
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

divar_data = pd.read_csv("./divar_posts_dataset.csv")

divar_data = divar_data.dropna(subset=['cat3','cat2'])

divar_data['full cat'] = divar_data['cat1'].map(str) + ' + ' + divar_data['cat2'].map(str) + ' + ' + divar_data['cat3'].map(str)

list_full_cat_dup = divar_data['full cat'].values.tolist()
list_full_cat = list(set(list_full_cat_dup))

list_city_dup = divar_data['city'].values.tolist()
list_city = list(set(list_city_dup))

city = []
ind = 0
index_city = []

gr = divar_data.groupby(['city', 'full cat']).size().reset_index(name="Count")
df = pd.DataFrame(gr)
df['con'] = df['city'].map(str) + ' - ' + df['full cat'].map(str) + ' - ' + df['Count'].map(str)
li = df['con'].values.tolist()

for i in list_city:

    ind = ind+1
    st = f"{i} = {ind}"
    index_city.append(st)

    c = []
    for j in list_full_cat:

        c.append(0)

    for k in li:

        st = str(k)
        if (st.split(' - ')[0] == i):
            cat = st.split(' - ')[1]
            index = list_full_cat.index(cat)
            c[index] = c[index] + int(st.split(' - ')[2])

    city.append(c)


print("")
print("===================================== City = Index =====================================")
print("")

for i in index_city:

    print(i)


print("")
print("Clustering results :")
print("")

print("===================================== Agglomerative Clustering =====================================")

model = AgglomerativeClustering(n_clusters=2)
# fit model and predict clusters
yhat = model.fit_predict(city)
# retrieve unique clusters
clusters = unique(yhat)

print(model.labels_)

score = silhouette_score(city, model.labels_, metric='euclidean')

print('Silhouetter Score: %.3f' % score)

print("===================================== BIRCH =====================================")

model = Birch(threshold=0.01, n_clusters=2)
# fit the model
model.fit(city)
# assign a cluster to each example
yhat = model.predict(city)

print(model.labels_)

score = silhouette_score(city, model.labels_, metric='euclidean')

print('Silhouetter Score: %.3f' % score)

print("===================================== K-Means =====================================")

model = KMeans(n_clusters=2)
# fit the model
model.fit(city)
# assign a cluster to each example
yhat = model.predict(city)
# retrieve unique clusters
clusters = unique(yhat)

print(model.labels_)

score = silhouette_score(city, model.labels_, metric='euclidean')

print('Silhouetter Score: %.3f' % score)

print("===================================== Mini-Batch K-Means =====================================")

model = MiniBatchKMeans(n_clusters=2)
# fit the model
model.fit(city)
# assign a cluster to each example
yhat = model.predict(city)
# retrieve unique clusters
clusters = unique(yhat)

print(model.labels_)

score = silhouette_score(city, model.labels_, metric='euclidean')

print('Silhouetter Score: %.3f' % score)

print("===================================== Mean Shift =====================================")

model = MeanShift()
# fit model and predict clusters
yhat = model.fit_predict(city)
# retrieve unique clusters
clusters = unique(yhat)

print(model.labels_)

score = silhouette_score(city, model.labels_, metric='euclidean')

print('Silhouetter Score: %.3f' % score)

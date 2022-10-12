
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

index_names = divar_data[divar_data['price'] == -1].index
divar_data.drop(index_names, inplace=True)

divar_data = divar_data.dropna(subset=['cat3','cat2'])

divar_data['full cat'] = divar_data['cat1'].map(str) + ' + ' + divar_data['cat2'].map(str) + ' + ' + divar_data['cat3'].map(str)

list_full_cat_dup = divar_data['full cat'].values.tolist()
list_full_cat = list(set(list_full_cat_dup))

list_price_dup = divar_data['price'].values.tolist()
list_price_int = list(set(list_price_dup))

list_price = []
for i in list_price_int:

    st = str(i)
    list_price.append(st)


item = []
ind = 0
index_item = []

gr = divar_data.groupby(['full cat', 'price']).size().reset_index(name="Count")
df = pd.DataFrame(gr)
df['con'] = df['full cat'].map(str) + ' - ' + df['price'].map(str) + ' - ' + df['Count'].map(str)
li = df['con'].values.tolist()

for i in list_full_cat:

    ind = ind+1
    st = f"{i} = {ind}"
    index_item.append(st)

    it = []

    for j in list_price:

        it.append(0)

    for k in li:

        st = str(k)

        if (st.split(' - ')[0] == i):

            pri = st.split(' - ')[1]
            index = list_price.index(pri)
            it[index] = it[index] + int(st.split(' - ')[2])

    item.append(it)


print("")
print("===================================== Cat = Index =====================================")
print("")

for i in index_item:

    print(i)

print("")
print("Clustering results :")
print("")

print("===================================== Agglomerative Clustering =====================================")

model = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat = model.fit_predict(item)
# retrieve unique clusters
clusters = unique(yhat)

print(model.labels_)

score = silhouette_score(item, model.labels_, metric='euclidean')

print('Silhouetter Score: %.3f' % score)

print("===================================== BIRCH =====================================")

model = Birch(threshold=0.01, n_clusters=4)
# fit the model
model.fit(item)
# assign a cluster to each example
yhat = model.predict(item)

print(model.labels_)

score = silhouette_score(item, model.labels_, metric='euclidean')

print('Silhouetter Score: %.3f' % score)

print("===================================== K-Means =====================================")

model = KMeans(n_clusters=4)
# fit the model
model.fit(item)
# assign a cluster to each example
yhat = model.predict(item)
# retrieve unique clusters
clusters = unique(yhat)

print(model.labels_)

score = silhouette_score(item, model.labels_, metric='euclidean')

print('Silhouetter Score: %.3f' % score)

print("===================================== Mini-Batch K-Means =====================================")

model = MiniBatchKMeans(n_clusters=4)
# fit the model
model.fit(item)
# assign a cluster to each example
yhat = model.predict(item)
# retrieve unique clusters
clusters = unique(yhat)

print(model.labels_)

score = silhouette_score(item, model.labels_, metric='euclidean')

print('Silhouetter Score: %.3f' % score)

print("===================================== Mean Shift =====================================")

model = MeanShift()
# fit model and predict clusters
yhat = model.fit_predict(item)
# retrieve unique clusters
clusters = unique(yhat)

print(model.labels_)

score = silhouette_score(item, model.labels_, metric='euclidean')

print('Silhouetter Score: %.3f' % score)


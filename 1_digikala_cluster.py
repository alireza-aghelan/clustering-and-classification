
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

digikala_data = pd.read_csv("./orders.csv")

list_c = ['تهران', 'مشهد', 'اصفهان', 'کرج', 'شیراز', 'اهواز', 'کرمانشاه', 'تبریز', 'قم']

list_city = ['Tehran', 'Mashhad', 'Isfahan', 'Karaj', 'Shiraz', 'Ahvaz', 'Kermanshah', 'Tabriz', 'Qom']

digikala_data = digikala_data.loc[digikala_data['city_name_fa'].isin(list_c)]

digikala_data["city_name_fa"].replace({"تهران": "Tehran", "مشهد": "Mashhad", "اصفهان": "Isfahan", "کرج": "Karaj", "شیراز": "Shiraz", "اهواز": "Ahvaz", "کرمانشاه": "kermanshah", "تبریز": "Tabriz", "قم": "Qom"}, inplace=True)

list_col_id = digikala_data['ID_Item'].values.tolist()
ID_Itemstr = []
for i in list_col_id:

    st = str(i)
    ID_Itemstr.append(st)

list_item = list(set(ID_Itemstr))

gr = digikala_data.groupby(['city_name_fa', 'ID_Item']).size().reset_index(name="Count")
df = pd.DataFrame(gr)
df['con'] = df['city_name_fa'].map(str) + ' - ' + df['ID_Item'].map(str) + ' - ' + df['Count'].map(str)
li = df['con'].values.tolist()

city = []
ind = 0
index_city = []

for i in list_city:

    ind = ind+1
    st = f"{i} = {ind}"
    index_city.append(st)

    c = []

    for j in list_item:

        c.append(0)

    for k in li:

        st = str(k)
        if (st.split(' - ')[0] == i):
            ite = st.split(' - ')[1]
            index = list_item.index(ite)
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

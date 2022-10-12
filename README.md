# clustering-and-classification
 In the first part, we want to perform clustering on the Divar and Digikala datasets as follows :

1-Clustering cities based on the products that are put up for sale in the Divar dataset
2-Clustering cities based on the products sold in them in the Digikala dataset
3-Clustering products based on their price in the Divar dataset

datasets can be downloaded from the link below :

https://fumdrive.um.ac.ir/index.php/s/LonCqfKMLc55kcL

To cluster cities, we first create a vector for each city.
the number of elements of this vector is equal to the number of types of products, for each type of product, there is a corresponding number of products.

In the next step, the vectors are entered into different clustering algorithms (Agglomerative Clustering - BIRCH - K-Means - Mini-Batch K-Means - Mean Shift) and clustering of cities is done. (clustering products is similar to clustering cities)

![1_2eQuIebjtTMJot27bWXgCQ](https://user-images.githubusercontent.com/47056654/195444041-8b52b11a-c739-4933-80bd-aa7f6b979af0.png)

to evaluate our clustering algorithms, we used the silhouette score.

below are some results for clustering cities in the Divar dataset :

city = index

<img width="381" alt="image" src="https://user-images.githubusercontent.com/47056654/195443217-5eb4ade0-8d18-4b68-a9f5-99df9694fd07.png">

clustering results:

<img width="527" alt="image" src="https://user-images.githubusercontent.com/47056654/195443244-6026c5d3-45a5-4cfe-80ea-03a49030ed9f.png">

In the next part, we determine whether a given price is an outlier or not
For this, we have done the following steps:

1- first we do some pre-processing: based on the price of different products, we assign a price label to different products and add a new column (label_price) to the dataset.

2- train a DecisionTreeClassifier.

3- The desired product specifications are entered and the trained model determines a price label for it.

4- The entered price is compared with the price label suggested by the model and it is determined whether the entered price is an outlier or not.


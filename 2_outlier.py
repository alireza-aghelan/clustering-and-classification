
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder


divar_data = pd.read_csv("./divar_posts_dataset.csv")

index_names = divar_data[divar_data['price'] == -1].index
divar_data.drop(index_names, inplace=True)

divar_data = divar_data.dropna(subset=['cat3','cat2'])

list_price_dup = divar_data['price'].values.tolist()

label_price = []
for i in list_price_dup:

    st = str(i)
    digit_numb = len(st)
    label_price.append(digit_numb)

divar_data['label_price'] = label_price

le = LabelEncoder()

divar_data['cat1en'] = le.fit_transform(divar_data['cat1'])
divar_data['cat2en'] = le.fit_transform(divar_data['cat2'])
divar_data['cat3en'] = le.fit_transform(divar_data['cat3'])

divar_data['cat1 enc'] = divar_data['cat1'].map(str) + ':' + divar_data['cat1en'].map(str)
list_cat1_enc_dup = divar_data['cat1 enc'].values.tolist()
list_cat1_enc = list(set(list_cat1_enc_dup))


divar_data['cat2 enc'] = divar_data['cat2'].map(str) + ':' + divar_data['cat2en'].map(str)
list_cat2_enc_dup = divar_data['cat2 enc'].values.tolist()
list_cat2_enc = list(set(list_cat2_enc_dup))


divar_data['cat3 enc'] = divar_data['cat3'].map(str) + ':' + divar_data['cat3en'].map(str)
list_cat3_enc_dup = divar_data['cat3 enc'].values.tolist()
list_cat3_enc = list(set(list_cat3_enc_dup))

print("")
print("===================================== Cat1 list =====================================")

for i in list_cat1_enc:

    print(i)

print("")
print("enter number of cat :")
c1en = input()

print("===================================== Cat2 list =====================================")

for i in list_cat2_enc:

    print(i)

print("")
print("enter number of cat :")
c2en = input()

print("===================================== Cat3 list =====================================")

for i in list_cat3_enc:

    print(i)

print("")
print("enter number of cat :")
c3en = input()

print("")
print("enter price :")
price_in = input()

print("")
print("predicting...")
pri_len = len(str(price_in))

# Separating the target variable
X = divar_data[['cat1en', 'cat2en', 'cat3en']]
Y = divar_data['label_price']

# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Creating the classifier object
clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

# Performing training
clf.fit(X_train, y_train)

# Predicton on test with giniIndex
y_pred = clf.predict(X_test)

print("")
print("Accuracy : ", accuracy_score(y_test, y_pred)*100)
print("")

li = [[c1en, c2en, c3en]]

pred_in = clf.predict(li)
d = pred_in[0]
print(f"number of price digits predicted by the model = {d}")
print("")

if (abs(pri_len-pred_in) >= 2):

    print("---------> price entered is out of the ordinary range")
else:

    print("---------> price entered is in the usual range")

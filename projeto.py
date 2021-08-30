from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pds
from sklearn.neighbors import KNeighborsClassifier

base_1_url = ""
base_2_url = ""

dataset_base_1 = pds.read_csv(base_1_url, header=None)
dataset_base_2 = pds.read_csv(base_2_url, header=None)

columns_1 = len(dataset_base_1)
columns_2 = len(dataset_base_2)

#Para a base 1
y_base_1 = dataset_base_1[0]
x_base_1 = dataset.loc[:,1:columns_1-1]

#Para a base 2
y_base_2 = dataset_base_2[0]
x_base_2 = dataset.loc[:,1:columns_2-1]

X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(x_base_1, y_base_1, test_size=0.2, random_state=None, stratify=y_base_1)

#Separando para a segunda base
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(x_base_2, y_base_2, test_size=0.2, random_state=None, stratify=y_base_2)

#Valor do n, para o KNN

n=3
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pds
from sklearn.neighbors import KNeighborsClassifier

def KNN_Bases(k, metrics, base):

    if(base =='car'):
        base_1_url = './bases/car.data'

        dataset_base_1 = pds.read_csv(base_1_url, header=None)

        x_base_1 = dataset_base_1.loc[:, 0:columns-2]
        y_base_1 = dataset_base_1.loc[:, columns-1]

        X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(x_base_1, y_base_1, test_size=0.2, random_state=None, stratify=y_base_1)

        model = KNeighborsClassifier(n_neighbors=k, metrics=metrics, algorithm='brute')

        model = model.fit(X_train_1, Y_train_1)

        resultado = model.predict(X_test_1)
        acc = metrics.accuracy_score(resultado, Y_test_1)
        result = round(acc * 100)
        print("Base car: {}%".format(result))
        
        return

    elif(base == 'glass'):
        base_2_url = "./bases/glass.data"

        dataset_base_2 = pds.read_csv(base_2_url, header=None)

        #Para a base 2
        x_base_2 = dataset_base_1.loc[:, 0:columns-2]
        y_base_2 = dataset_base_1.loc[:, columns-1]

        #Separando para a segunda base
        X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(x_base_2, y_base_2, test_size=0.2, random_state=None, stratify=y_base_2)

        resultado = model.predict(X_test_2)
        acc = metrics.accuracy_score(resultado, Y_test_2)
        result = round(acc * 100)
        print("Base glass: {}%".format(result))
        
        return

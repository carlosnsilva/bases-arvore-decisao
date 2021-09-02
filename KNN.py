from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pds
from sklearn.neighbors import KNeighborsClassifier

def KNN_Bases(k, metrics, base):

    if(base =='balance'):
        base_1_url = './bases/balance-scale.data'

        dataset = pds.read_csv(base_1_url, header=None)

        columns = len(dataset.columns)

        y = dataset[0]
        x = dataset.loc[:, 1:columns-1]
        

        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=None, stratify=y)

        model = KNeighborsClassifier(n_neighbors=k, metric=metrics, algorithm='brute')
        print( "X =>",X_train)
        print("Y=>",Y_train)


        model = model.fit(X_train, Y_train)

        resultado = model.predict(X_test)
        acc = metrics.accuracy_score(resultado, Y_test)
        result = round(acc * 100)
        print("Base car: {}%".format(result))
        
        return

    elif(base == 'glass'):
        base_2_url = "./bases/glass.data"

        dataset = pds.read_csv(base_2_url, header=None)

        columns = len(dataset.columns)

        #Para a base 2
        y = dataset[0]
        x = dataset.loc[:, 1:columns-1]
        

        #Separando para a segunda base
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=None, stratify=y)


        model = KNeighborsClassifier(n_neighbors=k, metric=metrics, algorithm='brute')
        model = model.fit(X_train, Y_train)

        resultado = model.predict(X_test)
        acc = metrics.accuracy_score(resultado, Y_test)
        result = round(acc * 100)
        print("Base glass: {}%".format(result))
        
        return

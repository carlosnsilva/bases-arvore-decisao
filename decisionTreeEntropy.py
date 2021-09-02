from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pds
from sklearn.neighbors import KNeighborsClassifier

def Entropy():

    base = "./bases/car.data"

    dataset = pds.read_csv(base, header=None)

    X = dataset.loc[:, 0:columns-2]
    Y = dataset.loc[:, columns-1]


    # 20% teste e 80% treinamento
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None, stratify=Y)

    # Treinando a árvore

    model = tree.DecisionTreeClassifier(criterion="entropy")
    model = model.fit(X_train, Y_train)

    resultado = model.predict(X_test)

    result_final = metrics.accuracy_score(resultado, Y_test)

    final = round(result_final * 100)

    print("{}%".format(final))
    
    return 

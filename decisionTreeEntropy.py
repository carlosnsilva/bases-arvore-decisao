from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pds
from sklearn.neighbors import KNeighborsClassifier

def Entropy():

    # base = './bases/' + database + '.data'
    base = './bases/balance-scale.data'

    dataset = pds.read_csv(base, header=None)
    

    index_Y = 0
    index_inicial = 1
    index_final = len(dataset)

    Y = dataset[index_Y]
    X = dataset.loc[:,index_inicial:index_final]

    # 20% teste e 80% treinamento
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None, stratify=Y)

    # Treinando a árvore

    model = tree.DecisionTreeClassifier(criterion="entropy")
    model = model.fit(X_train, Y_train)

    resultado = model.predict(X_test)

    result_final = metrics.accuracy_score(resultado, Y_test)

    final = round(result_final * 100)
    print('Base balance-scale')

    print("{}%".format(final))
    
    return 

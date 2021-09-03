from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pds
from sklearn.neighbors import KNeighborsClassifier

def Entropy(base_name):

    if(base_name == 'wine'):
        nome = "WINE"
        base = "./bases/wine.data"
        TreeEntropy(base,nome)
    elif(base_name == 'balance'):
        nome = "BALANCE"
        base = "./bases/balance-scale.data"
        TreeEntropy(base,nome)
    else:
        print("A base não foi encontrada!!!\n")
        return  

    
def TreeEntropy(base,nome):
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
    
    print("Resultado na base {}: {}%\n".format(nome,final))
    
    return 

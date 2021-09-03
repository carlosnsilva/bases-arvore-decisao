from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pds
from sklearn.neighbors import KNeighborsClassifier

def KNN_Bases(k, metrica, base_name):

    if(base_name =='balance'):
        nome = 'BALANCE'
        base_1_url = './bases/balance-scale.data'
        KNN_Generator(base_1_url,nome,k,metrica)
    elif(base_name == 'wine'):
        nome = 'WINE'
        base_2_url = "./bases/wine.data"
        KNN_Generator(base_2_url,nome,k,metrica)
    else:
        print("A base não foi encontrada!!!\n")
        return       


def KNN_Generator(base,nome, k, metrica):
        dataset = pds.read_csv(base, header=None)

        index_Y = 0
        index_inicial = 1
        index_final = len(dataset.columns)


        y = dataset[index_Y] # extrai a primeira coluna, que é o label
        X = dataset.loc[:,index_inicial:index_final-1]

        # 20% teste e 80% treinamento
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) # 80% treino e 20% teste

        model = KNeighborsClassifier(n_neighbors=k, metric=metrica, algorithm='brute')
        model = model.fit(X_train, y_train)

        result = model.predict(X_test)

        acc = metrics.accuracy_score(result, y_test)

        show = round(acc * 100)
        print("Resultado para a base {} com a metrica {}: {}%\n".format(nome,metrica,show))
        
        return
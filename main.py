from KNN import KNN_Bases
from decisionTreeGini import Gini
from decisionTreeEntropy import Entropy

while True:
    print("Projeto 1 da disciplina de Machine Learning")
    print("Escolha as opções abaixo para verificar os critérios: ")
    print("1 - Para KNN")
    print("2 - Para árvore de decisao gini")
    print("3 - Para árvore de decisao entropy")
    print("0 - Para sair")
    num = int(input("Digite a opção: "))
    if(num == 0):
        print("Encerrando execução!!!")
        break
    elif(num == 1):
        n = int(input("Digite quantas vezes deseja testar o KNN: "))
        for i in range(n):
            print("Bases: wine ou balance")    
            print("Metricas: euclidean, manhattan, minkowski\n")

            k = int(input("Digite o valor desejado para o k: "))
            metric = input("Digite a metrica que deseja utilizar: ")
            base = input("Digite a base que deseja executar: ")
            
            KNN_Bases(k,metric,base)

    elif(num == 2):
        print("Bases: wine ou balance")
        base = input("Digite o nome da base: ")
        
        Gini(base)
        
    elif(num == 3):
        print("Bases: wine ou balance")
        base = input("Digite o nome da base: ")
        Entropy(base)


from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pds
from sklearn.neighbors import KNeighborsClassifier

base = "./bases/car.data"

coluna_nome = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
coluna_feature = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

dataset = pds.read_csv(base, header=None, names=coluna_nome)

la = LabelEncoder()
for i in dataset:
    la.fit(dataset[i])
    dataset[i] = la.transform(dataset[i])

X = dataset[coluna_feature]
Y = dataset['class']


# 20% teste e 80% treinamento
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None, stratify=Y)

# Treinando a árvore

model = tree.DecisionTreeClassifier(criterion="gini")
model = model.fit(X_train, Y_train)

resultado = model.predict(X_test)

result_final = metrics.accuracy_score(resultado, Y_test)

final = round(result_final * 100)

print("{}%".format(final))

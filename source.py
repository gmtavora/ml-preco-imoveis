# Importar bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn import linear_model
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import AdaBoostRegressor

# Importar dados do conjunto de treinamento
features = pd.read_csv("conjunto_de_treinamento.csv")
features = features.sample(frac=1, random_state=12345)

regioes = {
    "centro": ["Recife", "Sto Amaro", "Boa Vista", "Cabanga", "Ilha do Leite", "Paissandu", "Sto Antonio", "S Jose", "Soledade", "Coelhos", "Ilha Joana Bezerra"],
    "norte": ["Arruda", "Campina do Barreto", "Campo Grande", "Encruzilhada", "Hipodromo", "Peixinhos", "Ponto de Parada", "Rosarinho", "Torreao", "Agua Fria", "Alto Santa Terezinha", "Bomba do Hemeterio", "Cajueiro", "Fundao", "Porto da Madeira", "Beberibe", "Dois Unidos", "Linha do Tiro"],
    "noroeste": ["Aflitos", "Alto do Mandu", "Apipucos", "Casa Amarela", "Casa Forte", "Derby", "Dois Irmaos", "Espinheiro", "Gracas", "Jaqueira", "Monteiro", "Parnamirim", "Poco da Panela", "Santana", "Tamarineira", "Sitio dos Pintos", "Alto Jose Bonifácio", "Alto Jose do Pinho", "Mangabeira", "Morro da Conceicao", "Vasco da Gama", "Brejo da Guabiraba", "Brejo do Beberibe", "Corrego do Jenipapo", "Guabiraba", "Macaxeira", "Nova Descoberta", "Passarinho", "Pau Ferro"],
    "oeste": ["Cordeiro", "Ilha do Retiro", "Iputinga", "Madalena", "Prado", "Torre", "Zumbi", "Engenho do Meio", "Torroes", "Caxanga", "Cid Universitaria", "Varzea"],
    "sudoeste": ["Afogados", "Bongi", "Mangueira", "Mustardinha", "San Martin", "Areias", "Cacote", "Estancia", "Jiquia", "Barro", "Coqueiral", "Curado", "Jd S Paulo", "Sancho", "Tejipio", "Toto"],
    "sul": ["Boa Viagem", "Brasilia Teimosa", "Imbiribeira", "Ipsep", "Pina", "Ibura", "Jordao", "Cohab"]
}

for i in range(len(features)):
    if features.loc[i, "bairro"] in regioes["centro"]:
        features.loc[i, "regiao"] = "centro"
    elif features.loc[i, "bairro"] in regioes["norte"]:
        features.loc[i, "regiao"] = "norte"
    elif features.loc[i, "bairro"] in regioes["noroeste"]:
        features.loc[i, "regiao"] = "noroeste"
    elif features.loc[i, "bairro"] in regioes["oeste"]:
        features.loc[i, "regiao"] = "oeste"
    elif features.loc[i, "bairro"] in regioes["sudoeste"]:
        features.loc[i, "regiao"] = "sudoeste"
    elif features.loc[i, "bairro"] in regioes["sul"]:
        features.loc[i, "regiao"] = "sul"
    else:
        features.loc[i, "regiao"] = "outro"
    
    if "copa" in features.loc[i, "diferenciais"].split(" e "):
        features.loc[i, "copa"] = 1
    else:
        features.loc[i, "copa"] = 0
    
    if "vestiario" in features.loc[i, "diferenciais"].split(" e "):
        features.loc[i, "vestiario"] = 1
    else:
        features.loc[i, "vestiario"] = 0
    
    if "children care" in features.loc[i, "diferenciais"].split(" e "):
        features.loc[i, "children_care"] = 1
    else:
        features.loc[i, "children_care"] = 0
    
    if "esquina" in features.loc[i, "diferenciais"].split(" e "):
        features.loc[i, "esquina"] = 1
    else:
        features.loc[i, "esquina"] = 0

# Remover colunas
features = features.drop(
    [
         "Id",
         "diferenciais",
         "bairro"
    ],
    axis=1
)

# Deletar outliers
for n in range(1,9):
    minPrice = features[features["quartos"] == n].preco.quantile(q=0.01)
    maxPrice = features[features["quartos"] == n].preco.quantile(q=0.90)
    
    features = features.drop(features[(features["quartos"] == n) & ((features["preco"] < minPrice) | (features["preco"] > maxPrice))].index)

for x in regioes.keys():
    if len(features[features["regiao"] == x]) > 1:
        q1 = features[features["regiao"] == x].preco.quantile(q=0.25)
        q3 = features[features["regiao"] == x].preco.quantile(q=0.75)
        iqr = q3 - q1
        features = features.drop(features[(features["regiao"] == x) & ((features["preco"] < q1-1.5*iqr) | (features["preco"] > q3+1.5*iqr))].index)    
            
for i in range (0, 6):
    q1 = features[features["suites"] == i].preco.quantile(q=0.25)
    q3 = features[features["suites"] == i].preco.quantile(q=0.75)
    iqr = q3 - q1
    features = features.drop(features[(features["suites"] == i) & ((features["preco"] < q1-1.5*iqr) | (features["preco"] > q3+1.5*iqr))].index)

# Divisão dos dados categóricos
features = pd.get_dummies(features,columns=
    [
         "tipo",
         "regiao"
    ]
)

# Padronização dos dados binários
binarizer = LabelBinarizer()

binaries = [
    "tipo_vendedor"
]

for v in binaries:
    features[v] = binarizer.fit_transform(features[v])

features.T

# Separação do conjunto em label e features
label = np.array(features["preco"]).astype(int)
features = features.drop("preco", axis=1)
feature_list = list(features.columns)

# Determinar os atributos com maior correlação com o preço
best_features = SelectKBest(score_func=f_regression, k=10)
fit = best_features.fit(features, label)

datafr = {"name": feature_list, "score": fit.scores_}
featureScores = pd.DataFrame(datafr)

print(featureScores)

# Excluir as colunas com baixa correlação
used = featureScores.sort_values(by="score").tail(30).name

new_features = features[features.columns.intersection(used)]

# MinMax Scaler
scaler = MinMaxScaler().fit(new_features)
scaled_features = scaler.transform(new_features)

# Separar conjunto em treinamento e teste
train_features, test_features, train_label, test_label = train_test_split(scaled_features, label, test_size = 0.3, random_state = 12345)

# Busca pelo melhor modelo
classifiers = [
    svm.SVR(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression(),
    linear_model.Ridge(alpha=.5),
    linear_model.Lasso(alpha=0.1),
    linear_model.LassoLars(alpha=.1),
    linear_model.ElasticNet(random_state=12345),
    linear_model.LogisticRegression()
]

for classifier in classifiers:
    print(classifier)
    clf = classifier
    clf.fit(train_features, train_label)
    print(clf.score(train_features, train_label))
    predictions = clf.predict(test_features)
    train_predictions = clf.predict(train_features)
    print("RMSPE (train) = ", np.sqrt(np.mean(np.square((train_predictions - train_label)/train_label))))
    print("RMSPE (test) = ", np.sqrt(np.mean(np.square((test_label - predictions)/test_label))))
    print("")

# Modelo base
model = DecisionTreeRegressor(random_state=12345)

model.fit(train_features, train_label)

train_predictions = model.predict(train_features)
print("RMSPE (train) = ", np.sqrt(np.mean(np.square((train_predictions - train_label)/train_label))))

test_predictions = model.predict(test_features)
print("RMSPE (test) = ", np.sqrt(np.mean(np.square((test_predictions - test_label)/test_label))))

# AdaBoostRegressor
regr = AdaBoostRegressor(DecisionTreeRegressor(random_state=12345), n_estimators=105, random_state=12345)
regr.fit(train_features, train_label)
regrtrain = regr.predict(train_features)
print("RMSPE (train) = ", np.sqrt(np.mean(np.square((regrtrain - train_label)/train_label))))
regrpredic = regr.predict(test_features)
print("RMSPE (test) = ", np.sqrt(np.mean(np.square((regrpredic - test_label)/test_label))))

# Conjunto de teste
# Importar dados
test_features = pd.read_csv("conjunto_de_teste.csv")

# Corrigir colunas
for i in range(len(test_features)):
    if test_features.loc[i, "bairro"] in regioes["centro"]:
        test_features.loc[i, "regiao"] = "centro"
    elif test_features.loc[i, "bairro"] in regioes["norte"]:
        test_features.loc[i, "regiao"] = "norte"
    elif test_features.loc[i, "bairro"] in regioes["noroeste"]:
        test_features.loc[i, "regiao"] = "noroeste"
    elif test_features.loc[i, "bairro"] in regioes["oeste"]:
        test_features.loc[i, "regiao"] = "oeste"
    elif test_features.loc[i, "bairro"] in regioes["sudoeste"]:
        test_features.loc[i, "regiao"] = "sudoeste"
    elif test_features.loc[i, "bairro"] in regioes["sul"]:
        test_features.loc[i, "regiao"] = "sul"
    else:
        test_features.loc[i, "regiao"] = "outro"
    
    if "copa" in test_features.loc[i, "diferenciais"].split(" e "):
        test_features.loc[i, "copa"] = 1
    else:
        test_features.loc[i, "copa"] = 0
    
    if "vestiario" in test_features.loc[i, "diferenciais"].split(" e "):
        test_features.loc[i, "vestiario"] = 1
    else:
        test_features.loc[i, "vestiario"] = 0
    
    if "children care" in test_features.loc[i, "diferenciais"].split(" e "):
        test_features.loc[i, "children_care"] = 1
    else:
        test_features.loc[i, "children_care"] = 0
    
    if "esquina" in test_features.loc[i, "diferenciais"].split(" e "):
        test_features.loc[i, "esquina"] = 1
    else:
        test_features.loc[i, "esquina"] = 0

test_features["tipo_Quitinete"] = 0

# Excluir colunas
id_series = test_features["Id"]
test_features = test_features.drop(
    [
         "Id",
         "diferenciais",
         "bairro"
    ],
    axis=1
)

# Colunas categóricas
test_features = pd.get_dummies(test_features,columns=
    [
         "tipo",
         "regiao"
    ]
)

# Colunas binárias
binarizer = LabelBinarizer()

binaries = [
    "tipo_vendedor"
]

for v in binaries:
    test_features[v] = binarizer.fit_transform(test_features[v])

# MinMax scaler
new_test_features = test_features[test_features.columns.intersection(used)]
scaled_test_features = scaler.transform(new_test_features)

test_features = np.array(test_features)

predictions = pd.DataFrame(id_series)
predictions = predictions.set_index("Id")
predictions["preco"] = regr.predict(scaled_test_features)
predictions.to_csv("result.csv")
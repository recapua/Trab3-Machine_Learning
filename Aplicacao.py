
# coding: utf-8

# In[100]:


#importar
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statistics import mean
from numpy.linalg import norm
from matplotlib.colors import ListedColormap
import random
import time
import timeit


# In[2]:


#importação da base de dados
#os dados estão disponíveis abertamente em http://archive.ics.uci.edu/ml/datasets/Covertype

file = "covtype.data"

#nome dos atributos
#estamos descartando os atributos que descrevem o tipo do solo
names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_Rawah', 'Wilderness_Area_Neota', 'Wilderness_Area_Comanche', 'Wilderness_Area_Cache', 'Cover_Type']

#de 0 a 13 são os atributos listados acima, e 54 é a classificação
usecols = list(range(0, 14)) + [54]

#especifico o tipo de alguns parametros(os que não são simplesmente numéricos)
dtype = {'Cover_Type': 'category', 'Wilderness_Area_Rawah' : bool, 'Wilderness_Area_Neota' : bool, 'Wilderness_Area_Comanche' : bool, 'Wilderness_Area_Cache' : bool}

#lê o arquivo num pandas.dataframe
dataset = pandas.read_csv(file, header = None, usecols = usecols, names = names, dtype = dtype)

#adicionando uma coluna adicional para sintetizar os 4 boleanos que representam a Wilderness_area. 
#para uma única instância, somente um dos 4 booleanos pode ser verdadeiro, logo eles, em realidade, funcionam como uma categorização
new_column = pandas.Series([1 if dataset['Wilderness_Area_Rawah'][i] else 
                            2 if dataset['Wilderness_Area_Neota'][i] else
                            3 if dataset['Wilderness_Area_Comanche'][i] else
                            4 for i in range(len(dataset.index)) ], dtype="category")
#elimina as colunas reduzidas
dataset = dataset.drop(columns=['Wilderness_Area_Rawah', 'Wilderness_Area_Neota', 'Wilderness_Area_Comanche', 'Wilderness_Area_Cache'])
#insere nova coluna na posição 10
dataset.insert(loc = 10, column = 'Wilderness_Area', value = new_column)

#atualiza names para refletir a mudança acima
names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area', 'Cover_Type']

print("Dataframe com %d exemplares e %d atributos importado com sucesso" % (dataset.values.shape[0], dataset.values.shape[1]))


# In[3]:


#test options and evaluation metric
seed = 7
scoring = 'accuracy'
#os atributos para treino em X e a classificação correta em Y
X = dataset.values[:, 0:11]
Y = dataset.values[:, 11]


# In[93]:


array = dataset.values
X = array[:, 0:11]
Y = array[:, 11]


#preprocessa dataset
#estava preprocessando, mas não fazia diferença
#X_scale_temp = preprocessing.scale(X[:, 0:10])
#X_scaled = np.append(X_scale_temp, X[:, 10].reshape(-1, 1), axis = 1)

#separamos um pedaço do dataset para usarmos de varificação depois
validation_size = 0.2
seed = 7
#estratificado por Y
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, stratify = Y)#, random_state = seed**2)

print(X.shape, X_train.shape, X_validation.shape)
X_all = np.append(X_train, X_validation, axis = 0)
Y_all = np.append(Y_train, Y_validation, axis = 0)

print(X.shape, X_train.shape, X_validation.shape, X_all.shape)
print(Y.shape, Y_train.shape, Y_validation.shape, Y_all.shape)



# In[98]:


#usamos um 10-fold para estimar a acurácia do método
def EstimarAcuraciaMetodo(metodo, k = 10):
    kfold = model_selection.KFold(n_splits = k, shuffle = True, random_state = 121)#, random_state = seed)
    cv_results = model_selection.cross_validate(metodo, X, Y, cv = kfold, scoring = scoring)
    print(cv_results)
    msg = "%s média entre os %d folds: %f, com desvio padrão de %f ,em " % (scoring, k, cv_results['test_score'].mean(), cv_results['test_score'].std())
    print(msg)
    msg2 = "tempo médio de treino: %f, para %d samples por fit" % (cv_results['fit_time'].mean(), (len(X_train) // k) * (k-1))
    print(msg2)


# # Full Decision Tree 

# In[99]:


dtc = DecisionTreeClassifier()
EstimarAcuraciaMetodo(dtc)


# In[108]:


#verificamos a acurácia do método aplicando-o sobre o set de validação que separamos anteriormente
dtc3 = DecisionTreeClassifier()
t0 = time.process_time()
dtc3.fit(X_train, Y_train)
tf = time.process_time()
print("Training time in seconds, for %d samples: %f" % (len(X_train), tf - t0))
predictions = dtc3.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
reshape = X_validation[0].reshape(1, -1)
N = 1000
print("Tempo médio para uma predição: %f segundos" % (timeit.timeit("dtc3.predict(reshape)", "from __main__ import dtc3, reshape", number = N) / N))


# # KNN

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 11)
AvaliarMetodo(dtc)


# In[112]:


#verificamos a acurácia do método aplicando-o sobre o set de validação que separamos anteriormente
knn = KNeighborsClassifier(n_neighbors = 11)
t0 = time.process_time()
knn.fit(X_train, Y_train)
tf = time.process_time()
print("Training time in seconds, for %d samples: %f" % (len(X_train), tf - t0))
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
reshape = X_validation[0].reshape(1, -1)
N = 1000
print("Tempo médio para uma predição: %f segundos" % (timeit.timeit("knn.predict(reshape)", "from __main__ import knn, reshape", number = N) / N))


# # KNN with three attributes

# In[117]:


#verificamos a acurácia do método aplicando-o sobre o set de validação que separamos anteriormente

#seleciono os tres parametros mais relevantes, baseado na minha exploração dos dados
X_3 = np.zeros((dataset.values.shape[0], 3))
X_3[:, 0] = dataset['Elevation']
X_3[:, 1] = dataset['Horizontal_Distance_To_Roadways']
X_3[:, 2] = dataset['Horizontal_Distance_To_Fire_Points']

#separamos um pedaço do dataset para usarmos de varificação depois
validation_size = 0.2
seed = 7
#estratificado por Y
X_train_3, X_validation_3, Y_train_3, Y_validation_3 = model_selection.train_test_split(X_3, Y, test_size = validation_size, stratify = Y)




knn3 = KNeighborsClassifier(n_neighbors = 5)
t0 = time.process_time()
knn3.fit(X_train_3, Y_train_3)
tf = time.process_time()
print("Training time in seconds, for %d samples: %f" % (len(X_train_3), tf - t0))
predictions = knn3.predict(X_validation_3)
print(accuracy_score(Y_validation_3, predictions))
print(confusion_matrix(Y_validation_3, predictions))
print(classification_report(Y_validation_3, predictions))
reshape = X_validation_3[0].reshape(1, -1)
N = 1000
print("Tempo médio para uma predição: %f segundos" % (timeit.timeit("knn3.predict(reshape)", "from __main__ import knn3, reshape", number = N) / N))


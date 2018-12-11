
# coding: utf-8

# # Exploração Numérica do Problema

# Sumário  
# Intro  
#           --- Importação  
# Exploração Numérica  
#         ---Média, mediana e percentis  
#         ---Média, mediana e percentis por classe  
#         --- Gráficos  
# Árvore de Decisão  
#         ---Completa  
#         ---Reduzida  

# In[1]:


# checar dependencias
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# In[1]:


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
import sklearn as skl

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


#corr = dataset.corr()['Cover_Type'][dataset.corr()['Cover_Type'] < 1].abs()
#corr.sort(ascending = False)
#corr.head()


# In[4]:


#atributos importados e seus tipos de dados
print(dataset.dtypes)


# In[5]:


#quantidade de exemplares por classificação
dataset.groupby('Cover_Type').size()


# In[6]:


#um exemplar da base de dados, para visualização
dataset.head(1)


# In[7]:


#sumário dos dados, geral
dataset.describe()


# In[8]:


#sumário dos dados, por classificação
gp = dataset.groupby('Cover_Type')
for name in names:
    print(name)
    display(gp[name].describe())


# In[9]:


#faz um histograma do atributo:
def histogram(att_name):
    
    fig, ax = plt.subplots()
    dataset[att_name].hist()
    ax.set(xlabel=att_name, ylabel = 'Number of Samples', title=att_name + ' Histogram')
    plt.show()


# In[10]:


histogram('Elevation')


# In[11]:


#um histograma da distância até foco de incêndio, que se provou um importante atributo
histogram('Horizontal_Distance_To_Fire_Points')


# In[12]:


#um histograma da distância até rodovias, que se provou um importante atributo
histogram('Horizontal_Distance_To_Roadways')


# In[13]:


#um histograma distribuição entre as categorias de cobertura
histogram('Cover_Type')


# In[14]:


#um histograma distribuição do sombreamento ao meio dia, que não ajuda muito
histogram('Hillshade_Noon')


# In[15]:



#faz um scatter plot maneiro
seed = 7
N = 250
#markerTypes = {'1':'.', '2': 'x', '3': 'o', '4': '^', '5': '1', '6': '2', '7': '3'}
markerTypes = {'1':'.', '2': '.', '3': '.', '4': '.', '5': '.', '6': '.', '7': '.'}
points = []
markers = []
gp = dataset.groupby('Cover_Type', sort = False)
samples_stratified = gp.apply(lambda x: x.sample(n = N, random_state = seed))
#para cada classificação, seleciona N amostras aleatórias pra serem plotadas

gp_random = dataset
samples_random = gp_random.apply(lambda x: x.sample(n = 7*N, random_state = seed))

#plota grafico name1 x name2
def scatter_plot(name1, name2, stratified = True):
    title_name = name1 + ' x ' + name2 + (' (E)' if stratified else ' (A)')
    samples = samples_stratified if stratified else samples_random
    fig, ax = plt.subplots()
    for cover_type, group in samples.groupby('Cover_Type'):

        plt.scatter(group[name1], group[name2], marker = markerTypes[cover_type])

    ax.set(xlabel=name1, ylabel=name2,
           title=title_name)
    plt.show()


# In[16]:


#plota grafico Elevation x Slope
scatter_plot('Elevation', 'Slope', stratified = True)


# In[17]:


#plota grafico Elevation x Slope
scatter_plot('Elevation', 'Slope', stratified = False)


# In[18]:


#plota grafico Elevation x Slope
scatter_plot('Elevation', 'Horizontal_Distance_To_Roadways', stratified = True)

#plota grafico Elevation x Slope
scatter_plot('Elevation', 'Horizontal_Distance_To_Roadways', stratified = False)


# In[19]:


#plota grafico Elevation x Slope
scatter_plot('Elevation', 'Horizontal_Distance_To_Fire_Points', stratified = True)

#plota grafico Elevation x Slope
scatter_plot('Elevation', 'Horizontal_Distance_To_Fire_Points', stratified = False)


# In[20]:


#plota grafico Elevation x Slope
scatter_plot('Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', stratified = True)

#plota grafico Elevation x Slope
scatter_plot('Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', stratified = False)


# In[21]:


#plota grafico Elevation x Slope
scatter_plot('Elevation', 'Elevation', stratified = True)

#plota grafico Elevation x Slope
scatter_plot('Elevation', 'Elevation', stratified = False)


# In[22]:


#plota grafico Elevation x Slope
scatter_plot('Hillshade_3pm', 'Hillshade_9am', stratified = True)

#plota grafico Elevation x Slope
scatter_plot('Hillshade_3pm', 'Hillshade_9am', stratified = False)


# In[23]:


#plota grafico Elevation x Slope
scatter_plot('Hillshade_Noon', 'Slope', stratified = True)

#plota grafico Elevation x Slope
scatter_plot('Hillshade_Noon', 'Slope', stratified = False)


# In[24]:


#plota grafico Elevation x Slope
scatter_plot('Hillshade_Noon', 'Aspect', stratified = True)

#plota grafico Elevation x Slope
scatter_plot('Hillshade_3pm', 'Aspect', stratified = False)


# # Árvore de Decisão

# In[25]:


#algumas funções auxiliares
def print_feature_importances(dtc):
    fi = dtc.feature_importances_
    for i in range(len(fi)):
        msg = "%s: %f" % (names[i], fi[i])
        print(msg)
       
def print_decision_tree_info(estimator, verbose = False):
    
    print(type(estimator))
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    if(verbose): print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            if(verbose): print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            if (verbose):print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print()
    
    print("Node count:", n_nodes)
    print("Leaf node count:", sum(b for b in is_leaves))
    print("Max node depth:", node_depth.max())
    
def get_decision_tree_info(estimator, verbose = False):
    
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    if(verbose): print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            if(verbose): print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            if (verbose):print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print()
    
    return (n_nodes, sum(b for b in is_leaves), node_depth.max())


# In[26]:


array = dataset.values
X = array[:, 0:11]
Y = array[:, 11]

#preprocessa dataset
#estava preprocessando, mas não fazia diferença
#X_scale_temp = preprocessing.scale(X[:, 0:10])
#X_scaled = np.append(X_scale_temp, X[:, 10].reshape(-1, 1), axis = 1)


validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)


# In[27]:


scoring = 'accuracy'
seed = 7
def EstimarAcuraciaMetodo(metodo, k = 10):
    kfold = model_selection.KFold(n_splits = k, shuffle = True, random_state = 121)#, random_state = seed)
    cv_results = model_selection.cross_validate(metodo, X_train, Y_train, cv = kfold, scoring = scoring, return_train_score = False)
    print(cv_results)
    msg = "%s média entre os %d folds: %f, com desvio padrão de %f ,em " % (scoring, k, cv_results['test_score'].mean(), cv_results['test_score'].std())
    print(msg)
    msg2 = "tempo médio de treino: %f, para %d samples por fit" % (cv_results['fit_time'].mean(), (len(X_train) // k) * (k-1))
    print(msg2)
    msg3 = "tempo médio de avaliação de uma sample: %f" % (cv_results['score_time'].mean() / ((len(X_train) / k)))
    print(msg3)

def EstimarAcuraciaMetodo_dtc(metodo, k = 10):
    kfold = model_selection.KFold(n_splits = k, shuffle = True, random_state = 121)#, random_state = seed)
    cv_results = model_selection.cross_validate(metodo, X_train, Y_train, cv = kfold, scoring = scoring, return_estimator=True)
    #print(cv_results)
    msg = "%s média entre os %d folds: %f, com desvio padrão de %f ,em " % (scoring, k, cv_results['test_score'].mean(), cv_results['test_score'].std())
    print(msg)
    msg2 = "tempo médio de treino: %f, para %d samples por fit" % (cv_results['fit_time'].mean(), (len(X_train) // k) * (k-1))
    print(msg2)
    nodes_n = []
    for dtc in cv_results['estimator']:
        nodes_n.append(get_decision_tree_info(dtc)[0])
    print("Média de %f nós por árvore gerada" % mean(nodes_n))
    print(nodes_n)


# In[28]:


dtcN = DecisionTreeClassifier(random_state = seed)
EstimarAcuraciaMetodo_dtc(dtcN)


# In[29]:


#experimetando com profundidades máximas. Já fiz a sem limite de profundidade aqui em cima
n_list = [1, 3, 5, 10,20, 30]
for i in n_list:
    print("Arvore de decisão limitada a profundidade de %d" % i)
    dtc1 = DecisionTreeClassifier(random_state = seed, max_depth = i)
    EstimarAcuraciaMetodo_dtc(dtc1)
    print(" ")


# In[30]:


#novamente árvore de decisão, porém reduzida a tres attributos
X_3 = np.zeros((array.shape[0], 3))
X_3[:, 0] = dataset['Elevation']
X_3[:, 1] = dataset['Horizontal_Distance_To_Roadways']
X_3[:, 2] = dataset['Horizontal_Distance_To_Fire_Points']
Y = dataset.values[:, 11]
validation_size = 0.2
seed = 7
X_train_3, X_validation_3, Y_train_3, Y_validation_3 = model_selection.train_test_split(X_3, Y, test_size = validation_size, random_state = seed)


# In[31]:


def EstimarAcuraciaMetodo_dtc_3(metodo, k = 10):
    kfold = model_selection.KFold(n_splits = k, shuffle = True, random_state = 121)#, random_state = seed)
    cv_results = model_selection.cross_validate(metodo, X_train_3, Y_train_3, cv = kfold, scoring = scoring, return_estimator=True)
    #print(cv_results)
    msg = "%s média entre os %d folds: %f, com desvio padrão de %f ,em " % (scoring, k, cv_results['test_score'].mean(), cv_results['test_score'].std())
    print(msg)
    msg2 = "tempo médio de treino: %f, para %d samples por fit" % (cv_results['fit_time'].mean(), (len(X_train_3) // k) * (k-1))
    print(msg2)
    nodes_n = []
    for dtc in cv_results['estimator']:
        nodes_n.append(get_decision_tree_info(dtc)[0])
    print("Média de %f nós por árvore gerada" % mean(nodes_n))
    print(nodes_n)


# In[32]:


dtc_3 = DecisionTreeClassifier(random_state = seed)
dtc_3.fit(X_train_3, Y_train_3)
dtc_3.score(X_validation_3, Y_validation_3)


# In[33]:


print_feature_importances(dtc_3)


# In[34]:


print_decision_tree_info(dtc_3)


# In[35]:


dtc_3_N = DecisionTreeClassifier(random_state = seed)
EstimarAcuraciaMetodo_dtc_3(dtc_3_N)


# In[36]:


#verificando contra o conjunto de validação,
#e tentar descobrir o nivel de importancia dos atributos baseado na arvore de decisão
dtc = DecisionTreeClassifier(random_state = seed)
dtc.fit(X_train, Y_train)
dtc.score(X_validation, Y_validation)


# In[37]:


#feature importance diz a importância relativa de cada atributo
print_feature_importances(dtc)


# In[38]:


print_decision_tree_info(dtc)


# # KNN

# In[39]:


#experimetando com profundidades máximas. Já fiz a sem limite de profundidade aqui em cima
n_list = [1, 3, 5, 7,9,11,15,25,51,101]
for i in n_list:
    print("KNN com %d vizinhos" % i)
    knn = KNeighborsClassifier(n_neighbors = i)
    EstimarAcuraciaMetodo(knn)
    print(" ")


# In[40]:


def EstimarAcuraciaMetodo_knn3(metodo, k = 10):
    kfold = model_selection.KFold(n_splits = k, shuffle = True, random_state = 121)#, random_state = seed)
    cv_results = model_selection.cross_validate(metodo, X_train_3, Y_train_3, cv = kfold, scoring = scoring, return_train_score = False)
    print(cv_results)
    msg = "%s média entre os %d folds: %f, com desvio padrão de %f ,em " % (scoring, k, cv_results['test_score'].mean(), cv_results['test_score'].std())
    print(msg)
    msg2 = "tempo médio de treino: %f, para %d samples por fit" % (cv_results['fit_time'].mean(), (len(X_train_3) // k) * (k-1))
    print(msg2)
    msg3 = "tempo médio de avaliação de uma sample: %f" % (cv_results['score_time'].mean() / ((len(X_train_3) / k)))
    print(msg3)

#experimetando KNN com só 3 atributos
n_list = [1, 3, 5, 7, 9, 11, 15]
for i in n_list:
    print("KNN com %d vizinhos" % i)
    knn = KNeighborsClassifier(n_neighbors = i)
    EstimarAcuraciaMetodo_knn3(knn)
    print(" ")


# In[41]:


#comentei esse teste porque demorava de mais e é improdutivo

#def custom_distance(u, v):
#    #para todos os campos "normais", só subtraio valores
#    custom = u - v
#    #mas, para o aspect, faço mod 360 . [1] é aspect
#    custom[1] = (u[1] - v[1]) % 360
#    dist = norm(u-v)
#    return dist
#
##experimetando com a distância customizada, que considera o fato de aspecto ser essencial modular
#n_list = [3]
#for i in n_list:
#    print("KNN customizado com %d vizinhos" % i)
#    knn = KNeighborsClassifier(n_neighbors = i, metric = custom_distance)
#    t0 = time.process_time()
#    knn.fit(X_train, Y_train)
#    tf = time.process_time()
#    print("Training time in seconds, for %d samples: %f" % (len(X_train), tf - t0))
#    knn.score(X_validation, Y_validation)
#    reshape = X_validation[0].reshape(1, -1)
#    N = 10
#    print("Tempo médio para uma predição: %f segundos" % (timeit.timeit("knn.predict(reshape)", "from __main__ import knn, reshape", number = N) / N))
#    print(" ")


# In[42]:




knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(X_train, Y_train)
knn.score(X_validation, Y_validation)


# In[43]:


knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(X_train_3, Y_train_3)
knn.score(X_validation_3, Y_validation_3)


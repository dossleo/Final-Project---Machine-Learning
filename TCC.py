# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

"Modelos utilizados"
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier     
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import cross_val_score

#Importando os dados para o modelo
        

dado_manut= pd.read_excel("Documents/bearing_dados.xlsx")


dado_manut.info()


#transformando variável fault em dummy

dado_manut['Fault'] = dado_manut['Fault'].map({'Sim':1,
                             'Não': 0,
                             },
                             na_action=None)

#Criando base auxiliar para colocar como frequência, e renomenando a variável fault

dado_manu_freq = dado_manut

dado_manu_freq['RPM'] = dado_manu_freq['RPM'].map({1797:29.95,
                             1772: 29.53,
                             1750: 29.17,
                             1725: 28.83,},
                             na_action=None)


dado_manu_freq = dado_manu_freq.rename(columns={"Fault":"Falha","RPM":"Freq"})
dado_manut = dado_manut.rename(columns={"Fault":"Falha"})


    
#-------------------------- Domínio do Tempo --------------------#

plt.figure()
sns.boxplot(x=dado_manut['Freq'],y=dado_manut['Variancia'],hue=dado_manut['Falha'])

plt.figure()
sns.boxplot(x=dado_manut['Freq'],y=dado_manut['RMS'],hue=dado_manut['Falha'])

plt.figure()
sns.boxplot(x=dado_manut['Freq'],y=dado_manut['Media'])

plt.figure()
sns.boxplot(x=dado_manut['Freq'],y=dado_manut['Maximo'])

plt.figure()
sns.boxplot(x=dado_manut['Freq'],y=dado_manut['Minimo'])

plt.figure()
sns.boxplot(x=dado_manut['Freq'],y=dado_manut['Fator Crista'])

plt.figure()
sns.boxplot(x=dado_manut['Freq'],y=dado_manut['Curtose'])

plt.figure()
sns.boxplot(x=dado_manut['Freq'],y=dado_manut['Range'])



plt.figure()
sns.boxplot(x=dado_manut['Falha'],y=dado_manut['Variancia'])

plt.figure()
sns.boxplot(x=dado_manut['Falha'],y=dado_manut['RMS'])

plt.figure()
sns.boxplot(x=dado_manut['Falha'],y=dado_manut['Media'])

plt.figure()
sns.boxplot(x=dado_manut['Falha'],y=dado_manut['Maximo'])

plt.figure()
sns.boxplot(x=dado_manut['Falha'],y=dado_manut['Minimo'])

plt.figure()
sns.boxplot(x=dado_manut['Falha'],y=dado_manut['Fator Crista'])

plt.figure()
sns.boxplot(x=dado_manut['Falha'],y=dado_manut['Curtose'])

plt.figure()
sns.boxplot(x=dado_manut['Falha'],y=dado_manut['Range'])

#---------------------- Domínio frequência --------------------#

plt.figure()
sns.boxplot(x=dado_manu_freq['Freq'],y=dado_manu_freq['Energia'],hue=dado_manu_freq['Falha'])

plt.figure()
sns.boxplot(x=dado_manu_freq['Freq'],y=dado_manu_freq['Potencia'],hue=dado_manu_freq['Falha'])



plt.figure()
sns.boxplot(x=dado_manu_freq['Falha'],y=dado_manu_freq['Energia'])

plt.figure()
sns.boxplot(x=dado_manu_freq['Falha'],y=dado_manu_freq['Potencia'])



dado_manut['Falha'].value_counts()

round(100*dado_manut['Falha'].value_counts()/len(dado_manut),2)

sns.set(font_scale=1.5, style = "whitegrid")
sns.distplot(x=dado_manut['Freq'],kde= True, bins=15)  

sns.set(font_scale=1.5, style = "whitegrid")
sns.distplot(x=dado_manut['RMS'],kde= True, bins=15)    

fig = plt.figure(figsize=(8,6))
plt.scatter(dado_manut['Falha'], dado_manut['Freq'], alpha=0.1)

fig = plt.figure(figsize=(8,6))
plt.scatter(dado_manut['Falha'], dado_manut['RMS'], alpha=0.1)

sns.distplot(dado_manut.loc[dado_manut['Falha'] == 1,'Freq'],label = "Falha")
sns.distplot(dado_manut.loc[dado_manut['Falha'] == 0,'Freq'],label = "Não falha")
plt.legend()

sns.distplot(dado_manut.loc[dado_manut['Falha'] == 1,'RMS'],label = "Falha")
sns.distplot(dado_manut.loc[dado_manut['Falha'] == 0,'RMS'],label = "Não falha")
plt.legend()

sns.distplot(dado_manut.loc[dado_manut['Falha'] == 1,'Potencia'],label = "Falha")
sns.distplot(dado_manut.loc[dado_manut['Falha'] == 0,'Potencia'],label = "Não falha")
plt.legend()

sns.distplot(dado_manut.loc[dado_manut['Falha'] == 1,'Energia'],label = "Falha")
sns.distplot(dado_manut.loc[dado_manut['Falha'] == 0,'Energia'],label = "Não falha")
plt.legend()



#----------------------------------------------------

#Matriz de correlação


plt.figure(figsize=(12,10))
cor = dado_manut.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
plt.show()



#----------------------------------------------------

#Retirando algumas variáveis para ocupação do gráfico
dado_manut1 = dado_manut[['Falha','Media','Fator Crista']]


#----------------------------------------------------

#Pairplot

sns.set(font_scale=1.5,style="whitegrid")
sns.pairplot(dado_manut, hue = 'Falha')  # utiliza as colunas numericas


#----------------------------------------------------


# chamando pacote para validação cruzada
 
kfold = StratifiedKFold(n_splits=10,shuffle = True, random_state = 0) 
 


'--------------------------------------------------------------------------'


#Modelo de Naive Bayes 

lista = [dado_manut,dado_manut1]

for i in lista:
    y = i['Falha']
    X = i.drop(columns='Falha')
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0, stratify=y, shuffle = True)
    
    print(f'Série de treino: {len(X_train)}')              # número de amostras no treino
    print(f'Série de teste: {len(X_test)}')               # número de amostras no teste
    print(f'Fração separada para teste: {round(len(X_test)/len(X),1)}')    # fração no conjunto de teste
    
    classifier = GaussianNB(priors=None, var_smoothing=1e-09)
    
         
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    
    acuracia_nb1 = accuracy_score(y_test, y_pred,normalize=True)    # acurácia(y_real,y_previsto)
    print(f'acurácia: {round(acuracia_nb1,3)}')
    
    cr_nb1 = classification_report(y_test,y_pred,target_names=['Não Falha','Falha'])      # (y_real,y_previsto)
    print(cr_nb1)
    
    cm_nb1 = confusion_matrix(y_test,y_pred)      # matriz de confusão (y_real, y_previsto)
    cm_nb1
    
    matriz_confusao_nb1 = pd.DataFrame(cm_nb1, columns = ['Previsão Não Falha', 'Previsão Falha'], index = ['Real Não Falha', 'Real Falha'] )
    matriz_confusao_nb1       
    
    plot_confusion_matrix(classifier, X_test, y_test,display_labels=['Não Falha','Falha'],
                                     cmap=plt.cm.Blues)
    
    
    '-----------------------------------------------------------------'
    
   
                   
    acuracia_nb_caso1 = cross_val_score(classifier, X = X_train, y = y_train, scoring = "accuracy", cv = kfold)         # validação cruzada caso 1 (é semelhante a um for)
    
    
    
    print(f'Média da cv: {round(np.mean(acuracia_nb_caso1),3)}')        # média da validação cruzada
    print(f'Desvio-padrão cv: {round(np.std(acuracia_nb_caso1),3)}')    # desvio-padrão da validação cruzada 
    print(acuracia_nb_caso1)      # todas as acurácias
    
    
    #Curva ROC
    
    plot_roc_curve(classifier, X_test, y_test) 
    plt.show()

'--------------------------------------------------------------------'



#Modelo Random Forest

lista = [dado_manut,dado_manut1]
for i in lista:
    y = i['Falha']
    X = i.drop(columns='Falha')
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0, stratify=y, shuffle = True)
    
    print(f'Série de treino: {len(X_train)}')              # número de amostras no treino
    print(f'Série de teste: {len(X_test)}')               # número de amostras no teste
    print(f'Fração separada para teste: {round(len(X_test)/len(X),1)}')    # fração no conjunto de teste
   
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    
    classifier.fit(X_train,y_train)
    
    
    
    y_pred = classifier.predict(X_test)                                                                # prevendo na série de teste
    
    # Métricas
    acuracia_rfr = accuracy_score(y_test, y_pred,normalize=True)                                       # Acurácia 
    print(round(acuracia_rfr,3))
    
    # Relatório de Avaliação: Precisão, Recall, F1-score
    cr_rfr = classification_report(y_test,y_pred,target_names=['Não Falha','Falha'])
    print(cr_rfr)
    
    # Matriz de Confusão:
    cm_rfr = confusion_matrix(y_test,y_pred)                        # matriz de confusão (y_real, y_previsto)
    matriz_confusao_rfr = pd.DataFrame(cm_rfr, columns = ['Previsão Não Falha', 'Previsão Falha'], index = ['Real Não Falha', 'Real Falha'] )
    print(matriz_confusao_rfr)  
    
    plot_confusion_matrix(classifier, X_test, y_test,display_labels=['Não Falha','Falha'],
                                     cmap=plt.cm.Blues)
    
    kfold = StratifiedKFold(n_splits=10,shuffle = True, random_state = 0) 
    
                   
    acuracia_nb_caso1 = cross_val_score(classifier, X = X_train, y = y_train, scoring = "accuracy", cv = kfold)         # validação cruzada caso 1 (é semelhante a um for)
    
    
    
    print(f'Média da cv: {round(np.mean(acuracia_nb_caso1),3)}')        # média da validação cruzada
    print(f'Desvio-padrão cv: {round(np.std(acuracia_nb_caso1),3)}')    # desvio-padrão da validação cruzada 
    print(acuracia_nb_caso1)      # todas as acurácias
    
    
    #Curva ROC
    
    plot_roc_curve(classifier, X_test, y_test) 
    plt.show()
    
    

'---------------------------------------------------------------------'


#Modelo Regressão Logística

lista = [dado_manut,dado_manut1]
for i in lista:
    y = i['Falha']
    X = i.drop(columns='Falha')
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0, stratify=y, shuffle = True)
    
    print(f'Série de treino: {len(X_train)}')              # número de amostras no treino
    print(f'Série de teste: {len(X_test)}')               # número de amostras no teste
    print(f'Fração separada para teste: {round(len(X_test)/len(X),1)}')    # fração no conjunto de teste
   

    classifier = LogisticRegression(C=100,  max_iter=200, class_weight=None)                           # chamando classificador regressão logística
    
    # Ajuste
    classifier.fit(X_train,y_train)                                                                    # ajustando os dados na série de treino
    
    # Previsão
    y_pred = classifier.predict(X_test)                                                                # prevendo na série de teste
    
    # Métricas
    acuracia_rl1 = accuracy_score(y_test, y_pred,normalize=True)                                       # Acurácia 
    print(round(acuracia_rl1,3))
    
    # Relatório de Avaliação: Precisão, Recall, F1-score
    cr_rl1 = classification_report(y_test,y_pred,target_names=['Não Falha','Falha'])
    print(cr_rl1)
    
    # Matriz de Confusão:
    cm_rl1 = confusion_matrix(y_test,y_pred)                        # matriz de confusão (y_real, y_previsto)
    matriz_confusao_rl1 = pd.DataFrame(cm_rl1, columns = ['Previsão Não Falha', 'Previsão Falha'], index = ['Real Não Falha', 'Real Falha'] )
    print(matriz_confusao_rl1)  
    
    plot_confusion_matrix(classifier, X_test, y_test,display_labels=['Não Falha','Falha'],
                                     cmap=plt.cm.Blues)
    

    
    # Possíveis valores de C que iremos testar:
    C_param_range = [0.001,0.01,0.1,1,10,100]          
    
    # Tabela para receber as medidas da validação cruzada para cada C de C_param_range:
    rl_acc_table = pd.DataFrame(columns = ['C: '+str(i) for i in C_param_range])
    
    
    for i in C_param_range:   # Para cada valor em C_param_range
    
      # Iniciação, ajuste e previsão
                                                    # importando Regressão Logística
      classifier = LogisticRegression(C=i,  max_iter=200, class_weight=None)                             # chamando classificador regressão logística
    
      # Validação cruzada para cada C = i
      vc = cross_val_score(classifier, X = X_train, y = y_train, scoring = "accuracy", cv = kfold)
      rl_acc_table['C: '+str(i)] = vc         # adicionando na tabela
    
      print(rl_acc_table)
      print(rl_acc_table.mean())
      print(rl_acc_table.std())



    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0, stratify=y, shuffle = True)
    classifiers = [LogisticRegression(C=0.001,  max_iter=200, class_weight=None), 
                   LogisticRegression(C=0.01,  max_iter=200, class_weight=None),
                   LogisticRegression(C=0.1,  max_iter=200, class_weight=None), 
                   LogisticRegression(C=1,  max_iter=200, class_weight=None),
                   LogisticRegression(C=10,  max_iter=200, class_weight=None),
                   LogisticRegression(C=100,  max_iter=200, class_weight=None)]
    
    for cls in classifiers:
        cls.fit(X_train, y_train)
    
    
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
    
    
    for cls, ax in zip(classifiers, axes.flatten()):
        plot_confusion_matrix(cls, 
                              X_test, 
                              y_test, 
                              ax=ax, 
                              cmap='Blues',
                             )
        ax.title.set_text(type(cls).__name__)
    
    plt.tight_layout()  
    plt.show()

    
    
#Curva ROC

    lr_1=LogisticRegression(C=0.001,  max_iter=200, class_weight=None) 
    lr_2=LogisticRegression(C=0.01,  max_iter=200, class_weight=None)
    lr_3=LogisticRegression(C=0.1,  max_iter=200, class_weight=None) 
    lr_4=LogisticRegression(C=1,  max_iter=200, class_weight=None)
    lr_5=LogisticRegression(C=10,  max_iter=200, class_weight=None)
    lr_6=LogisticRegression(C=100,  max_iter=200, class_weight=None)
    
    lr_1.fit(X_train,y_train)
    lr_2.fit(X_train,y_train)  
    lr_3.fit(X_train,y_train)  
    lr_4.fit(X_train,y_train)
    lr_5.fit(X_train,y_train)  
    lr_6.fit(X_train,y_train)  
    
    
    disp = plot_roc_curve(lr_1, X_test, y_test,name='C:0.001') 
    plot_roc_curve(lr_2, X_test, y_test,ax=disp.ax_,name='C:0.01') 
    plot_roc_curve(lr_3, X_test, y_test,ax=disp.ax_,name='C:0.1') 
    plot_roc_curve(lr_4, X_test, y_test,ax=disp.ax_,name='C:1') 
    plot_roc_curve(lr_5, X_test, y_test,ax=disp.ax_,name='C:10') 
    plot_roc_curve(lr_6, X_test, y_test,ax=disp.ax_,name='C:100') 



'---------------------------------------------------------------------------'




lista = [dado_manut,dado_manut1]
for i in lista:
    y = i['Falha']
    X = i.drop(columns='Falha')
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0, stratify=y, shuffle = True)
    
    print(f'Série de treino: {len(X_train)}')              # número de amostras no treino
    print(f'Série de teste: {len(X_test)}')               # número de amostras no teste
    print(f'Fração separada para teste: {round(len(X_test)/len(X),1)}')    # fração no conjunto de teste
   


    classifier = KNeighborsClassifier(n_neighbors=6, p=2, metric='minkowski')                           # chamando classificador KNN
    
     # Ajuste
    classifier.fit(X_train,y_train)                                                                    # ajustando os dados na série de treino
    
    # Previsão
    y_pred = classifier.predict(X_test)                                                                # prevendo na série de teste
    
    
    # Métricas
    acuracia_knn1 = accuracy_score(y_test, y_pred,normalize=True)                                      # Acurácia 
    print(round(acuracia_knn1,3))
    
    
    # Relatório de Avaliação: Precisão, Recall, F1-score
    cr_knn1 = classification_report(y_test,y_pred,target_names=['Não Falha','Falha'])
    print(cr_knn1)
    
    
    # Matriz de Confusão:
    cm_knn1 = confusion_matrix(y_test,y_pred)                        # matriz de confusão (y_real, y_previsto)
    matriz_confusao_knn1 = pd.DataFrame(cm_knn1, columns = ['Previsão Não Falha', 'Previsão Falha'], index = ['Real Não Falha', 'Real Falha'] )
    print(matriz_confusao_knn1)  
    
    plot_confusion_matrix(classifier, X_test, y_test,display_labels=['Não Falha','Falha'],
                                     cmap=plt.cm.Blues)
    
    
                   
    acuracia_knn = cross_val_score(classifier, X = X_train, y = y_train, scoring = "accuracy", cv = kfold)         # validação cruzada caso 1 (é semelhante a um for)
    
    print(f'Média da cv: {round(np.mean(acuracia_knn),3)}')        # média da validação cruzada
    print(f'Desvio-padrão cv: {round(np.std(acuracia_knn),3)}')    # desvio-padrão da validação cruzada 
    print(acuracia_knn)      # todas as acurácias
    
    
    
    n_param_range = [1,2,3,4,5,6]          
    
    # Tabela para receber as medidas da validação cruzada para cada C de C_param_range:
    rl_acc_table = pd.DataFrame(columns = ['N: '+str(i) for i in n_param_range])
    
    
    for i in n_param_range:   # Para cada valor em C_param_range
    
      # Iniciação, ajuste e previsão
                                                    # importando Regressão Logística
      classifier = KNeighborsClassifier(n_neighbors=i, p=2, metric='minkowski')                            # chamando classificador regressão logística
    
      # Validação cruzada para cada C = i
      vc = cross_val_score(classifier, X = X_train, y = y_train, scoring = "accuracy", cv = kfold)
      rl_acc_table['N: '+str(i)] = vc         # adicionando na tabela
    
    print(rl_acc_table)
    
    print(rl_acc_table.mean())
    print(rl_acc_table.std()) 
    plt.figure(figsize=(10,8))   # tamanho da figura
    sns.barplot(x="variable", y="value", data=pd.melt(rl_acc_table))    # boxplot de todas as variáveis do dataframe
    #melhor modelo é o knn com parâmetro n=3
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0, stratify=y, shuffle = True)
    classifiers = [KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski'), 
                   KNeighborsClassifier(n_neighbors=2, p=2, metric='minkowski'),
                   KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski'), 
                   KNeighborsClassifier(n_neighbors=4, p=2, metric='minkowski'),
                   KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'),
                   KNeighborsClassifier(n_neighbors=6, p=2, metric='minkowski')]
    
    for cls in classifiers:
        cls.fit(X_train, y_train)
    
    
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
    
    
    for cls, ax in zip(classifiers, axes.flatten()):
        plot_confusion_matrix(cls, 
                              X_test, 
                              y_test, 
                              ax=ax, 
                              cmap='Blues',
                             )
        ax.title.set_text(type(cls).__name__)
    
    plt.tight_layout()  
    plt.show()
    
#Curva ROC

    knn_1=KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski') 
    knn_2=KNeighborsClassifier(n_neighbors=2, p=2, metric='minkowski')
    knn_3=KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski') 
    knn_4=KNeighborsClassifier(n_neighbors=4, p=2, metric='minkowski')
    knn_5=KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn_6=KNeighborsClassifier(n_neighbors=6, p=2, metric='minkowski')
    
    knn_1.fit(X_train,y_train)
    knn_2.fit(X_train,y_train)  
    knn_3.fit(X_train,y_train)  
    knn_4.fit(X_train,y_train)
    knn_5.fit(X_train,y_train)  
    knn_6.fit(X_train,y_train)  
    
    
    disp = plot_roc_curve(knn_1, X_test, y_test,name='N:1') 
    plot_roc_curve(knn_2, X_test, y_test,ax=disp.ax_,name='N:2') 
    plot_roc_curve(knn_3, X_test, y_test,ax=disp.ax_,name='N:3') 
    plot_roc_curve(knn_4, X_test, y_test,ax=disp.ax_,name='N:4') 
    plot_roc_curve(knn_5, X_test, y_test,ax=disp.ax_,name='N:5') 
    plot_roc_curve(knn_6, X_test, y_test,ax=disp.ax_,name='N:6') 



'----------------------------------------------------------------------'



#Método do  SVM - Support Vector Machine

lista = [dado_manut,dado_manut1]
for i in lista:
    y = i['Falha']
    X = i.drop(columns='Falha')
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0, stratify=y, shuffle = True)

    classifier = SVC(C=50, kernel='rbf', degree=3, gamma='scale', class_weight=None)      # chamando classificador SVC
    
    
    # Ajuste
    classifier.fit(X_train,y_train)                                                                    # ajustando os dados na série de treino
    
    # Previsão
    y_pred = classifier.predict(X_test)                                                                # prevendo na série de teste
    
    # Métricas
    acuracia_svm1 = accuracy_score(y_test, y_pred,normalize=True)                                      # Acurácia 
    print(round(acuracia_svm1,3))
    
    # Relatório de Avaliação: Precisão, Recall, F1-score
    cr_svm1 = classification_report(y_test,y_pred,target_names=['Não Falha','Falha'])
    print(cr_svm1)
    
    # Matriz de Confusão:
    cm_svm1 = confusion_matrix(y_test,y_pred)                        # matriz de confusão (y_real, y_previsto)
    matriz_confusao_svm1 = pd.DataFrame(cm_svm1, columns = ['Previsão Não Falha', 'Previsão Falha'], index = ['Real Não Falha', 'Real Falha'] )
    print(matriz_confusao_svm1)  
    
    plot_confusion_matrix(classifier, X_test, y_test,display_labels=['Não Falha','Falha'],
                                     cmap=plt.cm.Blues)
    
    
    # Possíveis valores de C que iremos testar:
    C_param_range = [0.1,0.5,1,5,10,50]          
    
    # Para cada k, registrar a média da acurácia e o desvio-padrão da acurácia:
    svm_acc_table = pd.DataFrame(columns = ['C: '+str(i) for i in C_param_range])
    
    
    for i in C_param_range:   # Para cada valor em C_param_range
    
      # Classificador
      classifier = SVC(C=i, kernel='rbf', degree=3, gamma='scale', class_weight=None)      # chamando classificador SVC
    
      # Validação cruzada para cada C = i
      vc = cross_val_score(classifier, X = X_train, y = y_train, scoring = "accuracy", cv = kfold)
      svm_acc_table['C: '+str(i)] = vc
      
    
    print(svm_acc_table)
    
    print(svm_acc_table.mean())
    print(svm_acc_table.std())
    
    plt.figure(figsize=(10,8))   # tamanho da figura
    sns.barplot(x="variable", y="value", data=pd.melt(svm_acc_table))    # boxplot de todas as variáveis do dataframe
    
    
       
   
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0, stratify=y, shuffle = True)
    classifiers = [SVC(C=0.1, kernel='rbf', degree=3, gamma='scale', class_weight=None), 
                   SVC(C=0.5, kernel='rbf', degree=3, gamma='scale', class_weight=None),
                   SVC(C=1, kernel='rbf', degree=3, gamma='scale', class_weight=None), 
                   SVC(C=5, kernel='rbf', degree=3, gamma='scale', class_weight=None),
                   SVC(C=10, kernel='rbf', degree=3, gamma='scale', class_weight=None),
                   SVC(C=50, kernel='rbf', degree=3, gamma='scale', class_weight=None)]
    
    for cls in classifiers:
        cls.fit(X_train, y_train)
    
    
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
    
    
    for cls, ax in zip(classifiers, axes.flatten()):
        plot_confusion_matrix(cls, 
                              X_test, 
                              y_test, 
                              ax=ax, 
                              cmap='Blues',
                             )
        ax.title.set_text(type(cls).__name__)
    
    plt.tight_layout()  
    plt.show()
    
 #Curva ROC

    SVC_1=SVC(C=0.1, kernel='rbf', degree=3, gamma='scale', class_weight=None) 
    SVC_2=SVC(C=0.5, kernel='rbf', degree=3, gamma='scale', class_weight=None) 
    SVC_3=SVC(C=1, kernel='rbf', degree=3, gamma='scale', class_weight=None) 
    SVC_4=SVC(C=5, kernel='rbf', degree=3, gamma='scale', class_weight=None) 
    SVC_5=SVC(C=10, kernel='rbf', degree=3, gamma='scale', class_weight=None) 
    SVC_6=SVC(C=50, kernel='rbf', degree=3, gamma='scale', class_weight=None) 
    
    SVC_1.fit(X_train,y_train)
    SVC_2.fit(X_train,y_train)  
    SVC_3.fit(X_train,y_train)  
    SVC_4.fit(X_train,y_train)
    SVC_5.fit(X_train,y_train)  
    SVC_6.fit(X_train,y_train)  
    
    
    disp = plot_roc_curve(SVC_1, X_test, y_test,name='C:0.1') 
    plot_roc_curve(SVC_2, X_test, y_test,ax=disp.ax_,name='C:0.5') 
    plot_roc_curve(SVC_3, X_test, y_test,ax=disp.ax_,name='C:1') 
    plot_roc_curve(SVC_4, X_test, y_test,ax=disp.ax_,name='C:5') 
    plot_roc_curve(SVC_5, X_test, y_test,ax=disp.ax_,name='C:10') 
    plot_roc_curve(SVC_6, X_test, y_test,ax=disp.ax_,name='C:50') 


    
    


import sys
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import collections
        
if not sys.warnoptions:
    warnings.simplefilter("ignore")

divergencia_classificadores = False

#!/usr/bin/env python
# coding: utf-8
k_dynamic_combination = 0
from numpy import mean
from numpy import std
import numpy as np
import pandas  as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.base import BaseDES

import deslib
deslib.__version__


# In[3]:


# define a dataset Binary
#X, y = make_classification(n_samples=10000, n_features=15, n_informative=15, n_redundant=0, random_state=7)

# summarize the dataset
#print(X.shape, y.shape)


#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

porcentagem_escolheu_MAX, porcentagem_escolheu_SOFT, porcentagem_escolheu_HARD, porcentagem_escolheu_MIN, porcentagem_escolheu_G_MEAN, porcentagem_escolheu_sum_weight , porcentagem_escolheu_rede_neural, porcentagem_escolheu_rede_neural_soft , porcentagem_escolheu_rede_neural_soft_div , porcentagem_escolheu_borda , porcentagem_escolheu_naive_bayes , porcentagem_escolheu_peso_ponderado_comite = [],[],[],[], [],[],[],[],[], [],[],[]
resultados_hard = []
resultados_soft = []
resultados_max = []
resultados_min = []
resultados_geometric_mean = []
quantidade_exemplos_divergencia = []
quantidade_classificadores_selecionados = []
#resultados_peso_ponderado_classe_cada_amostra_sem_ajustes = []
#resultados_peso_ponderado_classe_cada_amostra_ajustado_0a1 = []
#resultados_peso_ponderado_comite_classe_distancia_maxima_teste = []
resultados_peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1 = []
#resultados_sum_weight_votes_per_class = []
resultados_sum_weight_0a1_votes_per_class = []
#resultados_sum_weight_line_votes_per_class = []
resultados_sum_weight_0a1_line_votes_per_class = []

#resultados_dynamic_metric_fusionk1 = []
resultados_dynamic_metric_fusionk3 = []

#resultados_sum_weight  = []
#resultados_sum_weight_line  = []
resultados_escolheu_rede_neural  = []
resultados_escolheu_rede_neural_soft  = []
resultados_escolheu_rede_neural_soft_div  = []
resultados_escolheu_borda  = []
resultados_escolheu_naive_bayes  = []

resultados_maximo_na_combinacao = []

# Herdando as características do KNORAE
class MyKnoraE(KNORAE):
    def __init__(self, pool_classifiers=None, k=5, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, random_state=None, voting='hard',
                 knn_classifier='knn', knne=False, DSEL_perc=0.5, n_jobs=-1,
                 voting_type='all'):

        super(MyKnoraE, self).__init__(pool_classifiers=pool_classifiers,
                                     k=k,
                                     DFP=DFP,
                                     with_IH=with_IH,
                                     safe_k=safe_k,
                                     IH_rate=IH_rate,
                                     random_state=random_state,
                                     knn_classifier=knn_classifier,
                                     knne=knne,
                                     DSEL_perc=DSEL_perc,
                                     n_jobs=n_jobs,
                                     voting=voting
                                     )
        # característica nova para conseguir trocar o tipo de voto.
        self.voting_type = voting_type
        self.dynamic_index_usage_neighbors_test = None
        self.dynamic_neighbors_index_train = None
        self.dynamic_neighbors_distances_test = None
        self.classificadores_competentes = None
        self.previsoes_classificadores = None
        self.probabilidades_classificadores = None
        self.previsoes_real = None

        if self.voting_type == "hard":
            self.voting = "hard"
        else:
            self.voting = "soft"
    '''
    def _dynamic_selection(self, competences, predictions, probabilities):
        print('_dynamic_selection')
        #Combine models using dynamic ensemble selection.
        selected_classifiers = self.select(competences)
        print("selected_classifiers")
        print(selected_classifiers)
        votes = np.ma.MaskedArray(predictions, ~selected_classifiers)
        print("Votes in dynamyc selection")
        print(votes)
        if self.voting_type == 'all':
            print('Tentando chamar a votação majoritária')
            votes = sum_votes_per_class(votes, self.n_classes_)
        elif self.voting_type == 'my':
            print('Chamando função customizada!')
            votes = my_vote(votes, self.n_classes_)
        elif self.voting_type == 'jesaias':
            print('Votando aleatoriamente')
            # Sempre que necessário adicionar outra opção... elif com o
            # nome do voto que você quer adicionar
        else:
            pass
        predicted_proba = votes / votes.sum(axis=1)[:, None]
        print("predicted_proba _dynamic_selection")
        print(predicted_proba)
        return predicted_proba


    '''
    def predict(self, X):
        #print("10")
        X = self._check_predict(X)
        preds = np.empty(X.shape[0], dtype=np.intp)
        need_proba = self.needs_proba or self.voting == 'soft'
        #print("11")
        base_preds, base_probas = self._preprocess_predictions(X, need_proba)
        # predict all agree
        #print("12")
        ind_disagreement, ind_all_agree = self._split_agreement(base_preds)
        #print("índice todos concordam")
        #print(ind_all_agree)
        #print("13")
        if ind_all_agree.size:
            preds[ind_all_agree] = base_preds[ind_all_agree, 0]
        # predict with IH
        #print("indice onde tem desacordo ")
        #print(ind_disagreement)
        #print("14")
        if ind_disagreement.size:
            #print("Primeiro if")
            distances, ind_ds_classifier, neighbors = self._IH_prediction(
                X, ind_disagreement, preds, is_proba=False
            )
            #print("neighbors")
            #print(neighbors)
            # Predict with DS - Check if there are still samples to be labeled.
            #print("Onde o desacordo vai ser usado a tecnica DS para classificar")
            #print(ind_ds_classifier)

            if ind_ds_classifier.size:
                #print("No IF")
                DFP_mask = self._get_DFP_mask(neighbors)
                #print("2")
                inds, sel_preds, sel_probas = self._prepare_indices_DS(
                    base_preds, base_probas, ind_disagreement,
                    ind_ds_classifier)
                #print("indices pred juntar")
                #print(inds)
                '''
                self.dynamic_index_usage_neighbors_test = inds
                self.dynamic_neighbors_distances_test = distances
                '''
                #print("dynamic_index_usage_neighbors_test")
                #print(self.dynamic_index_usage_neighbors_test)
                '''
                print("distances")
                print(distances)
                print("neighbors[ind_ds_classifier]")
                print(neighbors[ind_ds_classifier])

                self.dynamic_neighbors_index_train = neighbors[ind_ds_classifier]#Indices dos Vizinhos das amostras que serão classificadas (Indices da base de train)
                '''
                self.dynamic_index_usage_neighbors_test = inds
                self.dynamic_neighbors_distances_test = distances
                self.dynamic_neighbors_index_train = neighbors[ind_ds_classifier]#Indices dos Vizinhos das amostras que serão classificadas (Indices da base de train)
                '''print("Valores")
                print(self.dynamic_index_usage_neighbors_test)
                print(self.dynamic_neighbors_distances_test)
                print(self.dynamic_neighbors_index_train)'''
                #print("dynamic_neighbors_index_train")
                #print(self.dynamic_neighbors_index_train)
                #print("3")
                preds_ds = self.classify_with_ds(sel_preds, sel_probas,
                                                 neighbors, distances,
                                                 DFP_mask)
                #print("4")
                preds[inds] = preds_ds
        '''
        print("neighbors[ind_ds_classifier]")
        print(neighbors[ind_ds_classifier])
        self.dynamic_index_usage_neighbors_test = inds
        self.dynamic_neighbors_distances_test = distances
        self.dynamic_neighbors_index_train = neighbors[ind_ds_classifier]#Indices dos Vizinhos das amostras que serão classificadas (Indices da base de train)
        '''
        #print("final de tudo")
        return self.classes_.take(preds)



    def predict_proba(self, X):
        #print("Entrou aqui")
        """Estimates the posterior probabilities for sample in X.
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        predicted_proba : array of shape (n_samples, n_classes)
                          Probabilities estimates for each sample in X.
        """
        X = self._check_predict(X)
        #print("a")
        self._check_predict_proba()
        probas = np.zeros((X.shape[0], self.n_classes_))
        #print("b")
        base_preds, base_probas = self._preprocess_predictions(X, True)
        # predict all agree
        #print("c")
        ind_disagreement, ind_all_agree = self._split_agreement(base_preds)
        if ind_all_agree.size:
            probas[ind_all_agree] = base_probas[ind_all_agree].mean(axis=1)
        # predict with IH
        #print("d")
        if ind_disagreement.size:
            #print("e")
            distances, ind_ds_classifier, neighbors = self._IH_prediction(
                    X, ind_disagreement, probas, is_proba=True)
            # Predict with DS - Check if there are still samples to be labeled.
            #print("f")
            if ind_ds_classifier.size:
                DFP_mask = self._get_DFP_mask(neighbors)
                #print("g")
                inds, sel_preds, sel_probas = self._prepare_indices_DS(
                    base_preds, base_probas, ind_disagreement,
                    ind_ds_classifier)
                #print("Teste no PP")
                self.dynamic_index_usage_neighbors_test = inds
                self.dynamic_neighbors_distances_test = distances
                self.dynamic_neighbors_index_train = neighbors[ind_ds_classifier]#Indices dos Vizinhos das amostras que serão classificadas (Indices da base de train)
                #print("h")
                probas_ds = self.predict_proba_with_ds(sel_preds,
                                                       sel_probas,
                                                       neighbors, distances,
                                                       DFP_mask)
                #print("i")
                probas[inds] = probas_ds
        #print("j Fim")
        return probas

    def _dynamic_selection(self, competences, predictions, probabilities):
        #print(len(self.pool_classifiers))
        #print("saiu")
        #print(competences)
        #Combine models using dynamic ensemble selection.
        #print('_dynamic_selection')
        selected_classifiers = self.select(competences)
        '''
        print("Classificadores selecionados")
        print(selected_classifiers)
        print("probabilities")
        print(probabilities)
        print("predictions")
        print(predictions)
        print("dynamic_index_usage_neighbors_test")
        print(self.dynamic_index_usage_neighbors_test)
        '''
        try:
            self.previsoes_real = y_val.iloc[self.dynamic_index_usage_neighbors_test]
        except:
            #tratar erro
            self.previsoes_real = None

        self.classificadores_competentes = None
        #self.probabilidades_classificadores = None
        self.previsoes_classificadores = np.ma.MaskedArray(predictions, ~selected_classifiers)
        #print("selected_classifiers in function _dynamic_selection")
        #print(selected_classifiers)
        #print("self.voting")
        #print(self.voting)
        if  self.voting_type == 'my':
            #print('Chamando função my!')
            votes = np.ma.MaskedArray(predictions, ~selected_classifiers)
            #print("votes")
            #print(votes)
            votes = my_vote(votes, self.n_classes_)
            predicted_proba = votes / votes.sum(axis=1)[:, None]
            #print("predicted_proba")
            #print(predicted_proba)

        #elif self.voting_type == 'dynamic_metric_fusionk1':
        #    predicted_proba = self.dynamic_metric_fusion(probabilities, selected_classifiers,predictions,1)

        elif self.voting_type == 'dynamic_metric_fusionk3':
            #predicted_proba = self.dynamic_metric_fusion(probabilities, selected_classifiers,predictions,k_dynamic_combination)
            #predicted_proba = self.dynamic_metric_fusion2(probabilities, selected_classifiers,predictions,vizinhos_no_teste)
            predicted_proba = self.dynamic_metric_fusion2(probabilities, selected_classifiers,predictions,melhor_k)
        elif self.voting_type == 'aleatoria':
            predicted_proba = self.aleatoria(probabilities, selected_classifiers,predictions)

        elif self.voting_type == 'max':
            predicted_proba = self._max_proba(probabilities, selected_classifiers)

        elif self.voting_type == 'min':
            predicted_proba = self._minimun_proba(probabilities, selected_classifiers)

        elif self.voting_type == 'geometric_mean':
            predicted_proba = self.geometric_mean(probabilities, selected_classifiers)

        #salvo
        elif self.voting_type == 'peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1':
            predicted_proba = self._peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1(probabilities, selected_classifiers)

        #SALVO
        elif self.voting_type == 'sum_weight_0a1_votes_per_class':
            votes = np.ma.MaskedArray(predictions, ~selected_classifiers)
            predicted_proba = self._sum_weight_0a1_votes_per_class(votes, self.n_classes_)

        elif self.voting_type == 'rede_neural':
            predicted_proba = self.rede_neural_class(probabilities, selected_classifiers)
        elif self.voting_type == 'rede_neural_soft':
            predicted_proba = self.rede_neural_soft(probabilities, selected_classifiers)
        elif self.voting_type == 'rede_neural_soft_div':
            predicted_proba = self.rede_neural_soft_div(probabilities, selected_classifiers)
        elif self.voting_type == 'borda':
            predicted_proba = borda_class(probabilities, selected_classifiers)
        elif self.voting_type == 'naive_bayes':
            votes = np.ma.MaskedArray(predictions, ~selected_classifiers)
            predicted_proba = self.naive_bayes_combination(votes, self.n_classes_)

        elif self.voting == 'hard':
            votes = np.ma.MaskedArray(predictions, ~selected_classifiers)
            self.previsoes_classificadores = votes
            predicted_proba = sum_votes_per_class(votes, self.n_classes_)
        else:
            predicted_proba = self._mask_proba(probabilities, selected_classifiers)
        return predicted_proba

    def _mask_proba(self, probabilities, selected_classifiers):#Devolve as probabilidades mas remove dos classificadores não competentes
        #print("FUNÇÃO MEDIA DAS PROBABILIDADES")
        # Broadcast the selected classifiers mask
        # to cover the last axis (n_classes):
        selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
        # coloca mais um [] dentro dos elemendos [1 2 3]  => [ [1] [2] [3]]
        #print("selected_classifiers in function _mask_proba using np.expand_dims")
        #print(selected_classifiers)
        selected_classifiers = np.broadcast_to(selected_classifiers, probabilities.shape)
        #Tipo duplica os valores de acordo  com o segundo parametro, OBS que agora está do tamanho de  probabilidades e igual
        #print("selected_classifiers in function _mask_proba using np.broadcast_to")
        #print(selected_classifiers)
        masked_proba = np.ma.MaskedArray(probabilities, ~selected_classifiers)
        self.probabilidades_classificadores = masked_proba
        #retorna a mastriz original de probabiliade mas remove os valores dos classsificadores não selecionados
        #print("masked_proba in function _mask_proba using np.ma.MaskedArray")
        #print(masked_proba)

        predicted_proba = np.mean(masked_proba, axis=1)
        #print("predicted_proba resultado")
        #print(predicted_proba)
        return predicted_proba




    '''
    def _max_proba(self, probabilities, selected_classifiers):#Devolve as probabilidades mas remove dos classificadores não competentes
        print("Função maximo")
        # Broadcast the selected classifiers mask
        # to cover the last axis (n_classes):
        selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
        # coloca mais um [] dentro dos elemendos [1 2 3]  => [ [1] [2] [3]]
        print("selected_classifiers in function _mask_proba using np.expand_dims")
        print(selected_classifiers)
        selected_classifiers = np.broadcast_to(selected_classifiers, probabilities.shape)
        #Tipo duplica os valores de acordo  com o segundo parametro, OBS que agora está do tamanho de  probabilidades e igual
        print("selected_classifiers in function _mask_proba using np.broadcast_to")
        print(selected_classifiers)
        masked_proba = np.ma.MaskedArray(probabilities, ~selected_classifiers)
        #retorna a mastriz original de probabiliade mas remove os valores dos classsificadores não selecionados
        print("masked_proba in function _mask_proba using np.ma.MaskedArray")
        print(masked_proba)
        x, y, z = masked_proba.shape
        #Grupos(exemplos), lindas por grupos(classificadores), colunas(classes)
        print("x y z")
        print(x,y,z)
        for i in range(x):
            max_colun = np.amax(masked_proba[i], axis=0)
            print("max_colun")
            print(max_colun)
            sum_max_coluns = sum(max_colun)
            ajustar_valor_maximo = 1 / sum_max_coluns
            print("ajustar_valor_maximo")
            print(ajustar_valor_maximo)
            for j in range(y):
                for k in range(z):
                    print(masked_proba[i,j,k])
                    if masked_proba[i,j,k] >= 0:
                        pass
                        masked_proba[i,j,k] = max_colun[k] *ajustar_valor_maximo
        print("Novo masked_proba")
        print(masked_proba)
        #max_colun = np.amax(masked_proba, axis=0)
        #print("Maximo das colunas")
        #print(max_colun)
        return masked_proba
    '''



    #Fusão dinamica, escola de qual metrica (max, min, VM, Media)
    def dynamic_metric_fusion2(self, probabilities, selected_classifiers,predictions, vizinhos_val):#Devolve as probabilidades mas remove dos classificadores não competentes
        #print("----------Combinação dinamica 2---------")
        global divergencia_classificadores
        divergencia_classificadores = True
        
        #print("vizinhos_val")
        #print(vizinhos_val)
        #print("Meu X")
        #print(X_train)
        #print(len(X))
        total_exemplos = len(probabilities)
        #print("FUNÇÃO dynamic_metric_fusion")
        #print("probabilities")
        #print(probabilities)
        #print("selected_classifiers")
        #print(selected_classifiers)

        qnt_class_sel = 0
        linha,coluna = selected_classifiers.shape
        for lin in range(linha):
            for col in range(coluna):
                if selected_classifiers[lin,col] == True:
                    qnt_class_sel = qnt_class_sel + 1
                    
        #qnt_class_sel = collections.Counter(selected_classifiers)[True]
        qnt_class_sel = qnt_class_sel / linha
        #print("Qnt sele clas")
        #print(qnt_class_sel)

        #print("predictions")
        #print(predictions)

        probabilities_lista_dividida = np.array_split(probabilities,total_exemplos)
        selected_classifiers_lista_dividida = np.array_split(selected_classifiers,total_exemplos)
        predictions_lista_dividida = np.array_split(predictions,total_exemplos)
        #print("probabilities_lista_dividida")
        #print(probabilities_lista_dividida)
        #print("selected_classifiers_lista_dividida")
        #print(selected_classifiers_lista_dividida)
        #print("predictions_lista_dividida")
        #print(predictions_lista_dividida)

        import random
        from random import randrange
        #print("Juntar")
        #print(np.concatenate(probabilities_lista_dividida))
        resultados = []

        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=len(y_val2), n_jobs = 1)
        neigh.fit(X_val2, y_val2)

        X_test_divergente = X_test.iloc[self.dynamic_index_usage_neighbors_test]
        y_test_divergente = y_test.iloc[self.dynamic_index_usage_neighbors_test]

        #print(self.dynamic_index_usage_neighbors_test)
        #print("X_test_divergente")
        #print(X_test_divergente)        
        '''
        print("self.dynamic_index_usage_neighbors_test")
        print(self.dynamic_index_usage_neighbors_test)
        print("X_test")
        print(X_test)
        print("X_test_divergente")
        print(X_test_divergente)
        print("y_test_divergente")
        print(y_test_divergente)

        print("DO COMITE 1 2 3 4 5")
        print(pool_classifiers[0].predict(X_test_divergente))
        print(pool_classifiers[1].predict(X_test_divergente))
        print(pool_classifiers[2].predict(X_test_divergente))
        print(pool_classifiers[3].predict(X_test_divergente))
        print(pool_classifiers[4].predict(X_test_divergente))
        print("FIM COMITE")
        '''
        
        row_kneighbors_validation = neigh.kneighbors(X=X_test_divergente, n_neighbors=None, return_distance=False)
        #print(X_val2)
        #print("Vizinhos do Teste (os vizinhos estão no validation 2)")
        #print(row_kneighbors_validation)
        #print("self.dynamic_index_usage_neighbors_test")
        #print(self.dynamic_index_usage_neighbors_test)
        
        #lista de vizinhos proximos das classes que serão classificadas, os valores estão na base de teste
        row_unique_kneighbors_validation = np.unique(row_kneighbors_validation, return_counts=False)
        
        #print("VALORES PARA CLASSIFICAR X")
        #MELHORAR AINDA ESTA CONSUMINDO MUITO
        #X_neighbors_val = X_val2.iloc[row_unique_kneighbors_validation]
        #y_neighbors_val = y_val2.iloc[row_unique_kneighbors_validation]
        X_neighbors_val = X_val2
        y_neighbors_val = y_val2

        #print(X_neighbors_val)
        #print("VALORES PARA CLASSIFICAR Y")
        #print(y_neighbors_val)



        #print('TESTE MAX: ')
        modelMAX = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="max" )
        modelMAX.fit(X_train,y_train )
        #model_predictionsMAX = modelMAX.predict(X_neighbors_val)#predizer dados de treinos
        #proba_MAX =  modelMAX.predict_proba(X_test)
        #print(proba_MAX)
        resultados_max.append(modelMAX.score(X_test, y_test))
        #print(modelMAX.score(X_neighbors_val, y_val2))
        '''
        print("modelMAX.predict(X_test)")
        print(modelMAX.predict(X_test))
        print("modelMAX.predict(X_val2)")
        print(modelMAX.predict(X_val2))
        print("real val 2")
        print(y_val2)
        '''


        #print('TESTE SOFT: ')
        modelSOFT = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="soft" )
        modelSOFT.fit(X_train,y_train )
        #model_predictionsSOFT = modelSOFT.predict(X_neighbors_val)#predizer dados de treino
        #proba_SOFT =  modelSOFT.predict_proba(X_test)
        #print(proba_SOFT)
        resultados_soft.append(modelSOFT.score(X_test, y_test))
        #print(modelSOFT.score(X_neighbors_val, y_val2))


        #print('TESTE HARD: ')
        modelHARD = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="hard" )
        modelHARD.fit(X_train,y_train )
        #model_predictionsHARD = modelHARD.predict(X_neighbors_val)#predizer dados de treino
        #proba_HARD =  modelHARD.predict_proba(X_test)
        #print("####################################### PROBABILIDADE HARD ##############################################")
        #print(proba_HARD)
        resultados_hard.append(modelHARD.score(X_test, y_test))
        #print(modelHARD.score(X_neighbors_val, y_val2))


        #print('TESTE MIN: ')
        modelMIN = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="min" )
        modelMIN.fit(X_train,y_train )
        #model_predictionsMIN = modelMIN.predict(X_neighbors_val)#predizer dados de treinos
        #proba_MIN =  modelMIN.predict_proba(X_test)
        resultados_min.append(modelMIN.score(X_test, y_test))
        #print(modelMIN.score(X_neighbors_val, y_val2))


        #print('TESTE geometric_mean: ')
        modelG_MEAN = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="geometric_mean" )
        modelG_MEAN.fit(X_train,y_train )
        #model_predictionsG_MEAN = modelG_MEAN.predict(X_neighbors_val)#predizer dados de treino
        #proba_G_MEAN =  modelG_MEAN.predict_proba(X_test)
        resultados_geometric_mean.append(modelG_MEAN.score(X_test, y_test))
        #print(modelG_MEAN.score(X_neighbors_val, y_val2))



        #print('TESTE rede_neural: ')
        model_Rede_neural = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="rede_neural" )
        model_Rede_neural.fit(X_train,y_train )
        #model_predictionsRede_neural = model_Rede_neural.predict(X_neighbors_val)#predizer dados de trein
        #proba_Rede_neural =  model_Rede_neural.predict_proba(X_test)
        resultados_escolheu_rede_neural.append(model_Rede_neural.score(X_test, y_test))
        #print(model_Rede_neural.score(X_neighbors_val, y_val2))


        #proba_Rede_neural =  model_Rede_neural.predict_proba(X_test)

        #print('TESTE Rede_neural_soft : ')
        modelRede_neural_soft = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="rede_neural_soft" )
        modelRede_neural_soft.fit(X_train,y_train )
        #model_predictionsRede_neural_soft = modelRede_neural_soft.predict(X_neighbors_val)#predizer dados de treino
        #proba_Rede_neural_soft =  modelRede_neural_soft.predict_proba(X_test)
        resultados_escolheu_rede_neural_soft.append(modelRede_neural_soft.score(X_test, y_test))
        #print(modelRede_neural_soft.score(X_neighbors_val, y_val2))


        #print('TESTE Rede_neural_soft_div: ')
        modelRede_neural_soft_div = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="rede_neural_soft_div" )
        modelRede_neural_soft_div.fit(X_train,y_train )
        #model_predictionsRede_neural_soft_div = modelRede_neural_soft_div.predict(X_neighbors_val)#predizer dados de treino
        #proba_Rede_neural_soft_div =  modelRede_neural_soft_div.predict_proba(X_test)
        resultados_escolheu_rede_neural_soft_div.append(modelRede_neural_soft_div.score(X_test, y_test))
        #print(modelRede_neural_soft_div.score(X_neighbors_val, y_val2))



        #print('TESTE borda: ')
        modelBorda = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="borda" )
        modelBorda.fit(X_train,y_train )
        #model_predictionsBorda = modelBorda.predict(X_neighbors_val)#predizer dados de treino
        #proba_Borda =  modelBorda.predict_proba(X_test)
        resultados_escolheu_borda.append(modelBorda.score(X_test, y_test))
        #print(modelBorda.score(X_neighbors_val, y_val2))


        #print('TESTE naive_bayes: ')
        modelNaive_bayes = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="naive_bayes" )
        modelNaive_bayes.fit(X_train,y_train )
        #model_predictionsNaive_bayes = modelNaive_bayes.predict(X_neighbors_val)#predizer dados de treino
        #proba_Naive_bayes =  modelNaive_bayes.predict_proba(X_test)
        resultados_escolheu_naive_bayes.append(modelNaive_bayes.score(X_test, y_test))
        #print(modelNaive_bayes.score(X_neighbors_val, y_val2))


        #####TIRAR
        #print('TESTE Peso_ponderado_classe: ')
        #print("50")
        #modelPeso_ponderado_classe = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="peso_ponderado_classe_cada_amostra_ajustado_0a1" )
        #print("51")
        #modelPeso_ponderado_classe.fit(X_train,y_train )
        #print("52")
        #model_predictionsPeso_ponderado_classe = modelPeso_ponderado_classe.predict(X_neighbors_val)#predizer dados de treino
        #print("53")
        #resultados_peso_ponderado_classe_cada_amostra_ajustado_0a1.append(modelPeso_ponderado_classe.score(X_test, y_test))
        #print("54")
        #proba_Peso_ponderado_classe =  modelPeso_ponderado_classe.predict_proba(X_test)
        #print("55")
        #print(modelPeso_ponderado_classe.score(X_neighbors_val, y_val2))


        #print('TESTE peso_ponderado_comite: ')
        modelPeso_ponderado_comite = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1" )
        #print("60")
        modelPeso_ponderado_comite.fit(X_train,y_train )
        #print("61")
        #model_predictionsPeso_ponderado_comite = modelPeso_ponderado_comite.predict(X_neighbors_val)#predizer dados de treino
        #print("62")
        #proba_Peso_ponderado_comite =  modelPeso_ponderado_comite.predict_proba(X_test)
        #print("63")
        resultados_peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1.append(modelPeso_ponderado_comite.score(X_test, y_test))
        #print("64")
        #print(modelPeso_ponderado_comite.score(X_neighbors_val, y_val2))



        #print('TESTE sum_weight: ')
        modelSum_weight = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="sum_weight_0a1_votes_per_class" )
        #print("71")
        modelSum_weight.fit(X_train,y_train )
        #print("72")
        #model_predictionsSum_weight = modelSum_weight.predict(X_neighbors_val)#predizer dados de treino
        #print("73")
        #proba_Sum_weight =  modelSum_weight.predict_proba(X_test)
        #print("74")
        resultados_sum_weight_0a1_votes_per_class.append(modelSum_weight.score(X_test, y_test))
        #print("75")
        #print(modelSum_weight.score(X_neighbors_val, y_val2))

        #TIRAR
        #print('TESTE Sum_weight_line: ')
        #modelSum_weight_line = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="sum_weight_0a1_line_votes_per_class" )
        #print("81")
        #modelSum_weight_line.fit(X_train,y_train )
        #print("82")
        #model_predictionsSum_weight_line = modelSum_weight_line.predict(X_neighbors_val)#predizer dados de treino
        #print("83")
        #proba_Sum_weight_line =  modelSum_weight_line.predict_proba(X_test)
        #print("84")
        #resultados_sum_weight_0a1_line_votes_per_class.append(modelSum_weight_line.score(X_test, y_test))
        #print("85")
        #print(modelSum_weight_line.score(X_neighbors_val, y_val2))

        #print(len(model_predictionsMAX))
        #print(len(y_neighbors_val))
        #print('RESULTADO TESTE MAX in val: ')
        #print(model_predictionsMAX)


        #print('RESULTADO TESTE VAL REAL: ')
        #print(y_neighbors_val)
        #print('RESULTADO TESTE VAL SOFT: ')
        #print(model_predictionsSOFT)




        #%matplotlib inline

        resultado_MAX =  modelMAX.predict(X_test)
        resultado_SOFT =  modelSOFT.predict(X_test)
        resultado_HARD =  modelHARD.predict(X_test)
        resultado_MIN =  modelMIN.predict(X_test)
        resultado_G_MEAN =  modelG_MEAN.predict(X_test)
        resultado_Rede_neural =  model_Rede_neural.predict(X_test)
        resultado_Rede_neural_soft =  modelRede_neural_soft.predict(X_test)
        resultado_Rede_neural_soft_div =  modelRede_neural_soft_div.predict(X_test)
        resultado_Borda =  modelBorda.predict(X_test)
        resultado_Naive_bayes =  modelNaive_bayes.predict(X_test)
        resultado_Peso_ponderado_comite =  modelPeso_ponderado_comite.predict(X_test)
        resultado_Sum_weight =  modelSum_weight.predict(X_test)

        
        acertos_maximo = 0
        for i in range(len(y_test)):
            if  (resultado_SOFT[i] == y_test.iloc[i]) or (resultado_HARD[i] == y_test.iloc[i]) or (resultado_G_MEAN[i] == y_test.iloc[i]) or (resultado_Rede_neural[i] == y_test.iloc[i]) or (resultado_Rede_neural_soft[i] == y_test.iloc[i]) or (resultado_Rede_neural_soft_div[i] == y_test.iloc[i]) or (resultado_Borda[i] == y_test.iloc[i]) or (resultado_Naive_bayes[i] == y_test.iloc[i]):
                #if (resultado_MAX[i] == y_test.iloc[i]) or (resultado_SOFT[i] == y_test.iloc[i]) or (resultado_HARD[i] == y_test.iloc[i]) or (resultado_MIN[i] == y_test.iloc[i]) or (resultado_G_MEAN[i] == y_test.iloc[i]) or (resultado_Rede_neural[i] == y_test.iloc[i]) or (resultado_Rede_neural_soft[i] == y_test.iloc[i]) or (resultado_Rede_neural_soft_div[i] == y_test.iloc[i]) or (resultado_Borda[i] == y_test.iloc[i]) or (resultado_Naive_bayes[i] == y_test.iloc[i]) or (resultado_Peso_ponderado_comite[i] == y_test.iloc[i]) or (resultado_Sum_weight[i] == y_test.iloc[i]):
                acertos_maximo  = acertos_maximo + 1
        
            
        resultados_maximo_na_combinacao.append(acertos_maximo / len(y_test))


        
 
        #print(len(self.dynamic_index_usage_neighbors_test))




        '''
        print("vizinhos_val")
        print(vizinhos_val)
        print("row_kneighbors_validation")
        print(row_kneighbors_validation)
        print("row_kneighbors_validation.shape")
        print(row_kneighbors_validation.shape)
        print("self.dynamic_index_usage_neighbors_test")
        print(self.dynamic_index_usage_neighbors_test.shape)
        '''
        #print("MOSTRAR RESULTADO")
        lin, col = row_kneighbors_validation.shape
        resultados = []
        #print("MAX SOFT HARD MIN G_MEAN")
        escolheu_MAX = 0
        escolheu_SOFT = 0
        escolheu_HARD = 0
        escolheu_MIN = 0
        escolheu_G_MEAN = 0

        #escolheu_peso_ponderado_classe = 0
        escolheu_peso_ponderado_comite = 0
        escolheu_sum_weight = 0
        #escolheu_sum_weight_line = 0
        escolheu_rede_neural = 0
        escolheu_rede_neural_soft = 0
        escolheu_rede_neural_soft_div = 0
        escolheu_borda = 0
        escolheu_naive_bayes = 0
        #print("Os desacordo")
        #print(self.dynamic_index_usage_neighbors_test)

        pMAX = modelMAX.predict(X_val2)
        #print("pMAX")
        #print(pMAX)
        pSOFT = modelSOFT.predict(X_val2)
        pHARD = modelHARD.predict(X_val2)
        pMIN = modelMIN.predict(X_val2)
        pG_MEAN = modelG_MEAN.predict(X_val2)
        pPeso_ponderado_comite = modelPeso_ponderado_comite.predict(X_val2)
        pSum_weight = modelSum_weight.predict(X_val2)
        pRede_neural = model_Rede_neural.predict(X_val2)
        pRede_neural_soft = modelRede_neural_soft.predict(X_val2)
        pRede_neural_soft_class = modelRede_neural_soft_div.predict(X_val2)
        pBorda = modelBorda.predict(X_val2)
        pNaive_bayes = modelNaive_bayes.predict(X_val2)

        # I  de 0 a x de quantos precisam do comitê
        for i in range(len(self.dynamic_index_usage_neighbors_test)):
            
            numero_colunas = 0
            continuar_MAX = True
            continuar_SOFT = True
            continuar_HARD = True
            continuar_MIN = True
            continuar_G_MEAN = True

            #continuar_peso_ponderado_classe = True
            continuar_peso_ponderado_comite = True
            continuar_sum_weight = True
            #continuar_sum_weight_line = True
            continuar_rede_neural = True
            continuar_rede_neural_soft = True
            continuar_rede_neural_soft_div = True
            continuar_borda = True
            continuar_naive_bayes = True
            #print("######## VIZINHO "+str(i)+ "########")
            
            acertos_MAX = 0
            acertos_SOFT = 0
            acertos_HARD = 0
            acertos_MIN = 0
            acertos_G_MEAN = 0
            acertos_peso_ponderado_comite = 0
            acertos_sum_weight = 0
            acertos_rede_neural = 0
            acertos_rede_neural_soft = 0
            acertos_rede_neural_soft_div = 0
            acertos_borda = 0
            acertos_naive_bayes = 0

            #print("self.dynamic_index_usage_neighbors_test[i]")
            #print(self.dynamic_index_usage_neighbors_test[i])
            a_classificar_X_teste = X_test.iloc[self.dynamic_index_usage_neighbors_test[i]]
            #print(a_classificar_X_teste)
            #print("Classificar")
            #print(a_classificar_X_teste)

            qt_vizinhos_classificados = 0
            # primeiro, segundo divergente  dos vizinhos de teste
            #print("OS vizinhos")
            #print(row_kneighbors_validation[i])
            for vizinho in row_kneighbors_validation[i]:
                #print("vizinho ######################################################################")
                #print(vizinho)
                qt_vizinhos_classificados = qt_vizinhos_classificados + 1

                
                
                
                real_y_val2_classificar = y_val2.iloc[vizinho]
                '''
                preveuMAX = modelMAX.predict([X_val2.iloc[vizinho]])
                preveuSOFT = modelSOFT.predict([X_val2.iloc[vizinho]])
                preveuHARD = modelHARD.predict([X_val2.iloc[vizinho]])
                preveuMIN = modelMIN.predict([X_val2.iloc[vizinho]])
                preveuG_MEAN = modelG_MEAN.predict([X_val2.iloc[vizinho]])
                preveuPeso_ponderado_comite = modelPeso_ponderado_comite.predict([X_val2.iloc[vizinho]])
                preveuSum_weight = modelSum_weight.predict([X_val2.iloc[vizinho]])
                preveuRede_neural = model_Rede_neural.predict([X_val2.iloc[vizinho]])
                preveuRede_neural_soft = modelRede_neural_soft.predict([X_val2.iloc[vizinho]])
                preveuRede_neural_soft_class = modelRede_neural_soft_div.predict([X_val2.iloc[vizinho]])
                preveuBorda = modelBorda.predict([X_val2.iloc[vizinho]])
                preveuNaive_bayes = modelNaive_bayes.predict([X_val2.iloc[vizinho]])
                '''

                preveuMAX = pMAX[vizinho]
                preveuSOFT = pSOFT[vizinho]
                preveuHARD = pHARD[vizinho]
                preveuMIN = pMIN[vizinho]
                preveuG_MEAN = pG_MEAN[vizinho]
                preveuPeso_ponderado_comite = pPeso_ponderado_comite[vizinho]
                preveuSum_weight = pSum_weight[vizinho]
                preveuRede_neural = pRede_neural[vizinho]
                preveuRede_neural_soft = pRede_neural_soft[vizinho]
                preveuRede_neural_soft_class = pRede_neural_soft_class[vizinho]
                preveuBorda = pBorda[vizinho]
                preveuNaive_bayes = pNaive_bayes[vizinho]                
                #print("Real "+str(real_y_val2_classificar)+" MAX "+str(preveuMAX)+" HARD "+str(preveuHARD)+" SOFT "+str(preveuSOFT)+" MAX "+str(preveuMAX)+" MIN "+str(preveuMIN)+" G_MEAN "+str(preveuG_MEAN)+" Peso_ponderado_comite "+str(preveuPeso_ponderado_comite)+" Sum_weight "+str(preveuSum_weight)+" Rede_neural "+str(preveuRede_neural)+" Rede_neural_soft "+str(preveuRede_neural_soft)+" Rede_neural_soft_class "+str(preveuRede_neural_soft_class)+" Borda "+str(preveuBorda)+" Naive_bayes "+str(preveuNaive_bayes))

                #print( str(preveuMAX)+ " - "+ str(modelMAX.predict([X_val2.iloc[vizinho]]))   )
                '''
                if(continuar_MAX and preveuMAX == real_y_val2_classificar):
                    #print("acertou max")
                    acertos_MAX = acertos_MAX + 1
                '''
                if( continuar_SOFT and  preveuSOFT == real_y_val2_classificar):
                    #print("acertou soft")
                    acertos_SOFT = acertos_SOFT + 1

                if( continuar_HARD and  preveuHARD == real_y_val2_classificar):
                    #print("acertou HARD")
                    acertos_HARD = acertos_HARD + 1
                '''
                if( continuar_MIN and  preveuMIN == real_y_val2_classificar):
                    #print("acertou HARD")
                    acertos_MIN = acertos_MIN + 1
                '''
                if( continuar_G_MEAN and  preveuG_MEAN == real_y_val2_classificar):
                    #print("acertou HARD")
                    acertos_G_MEAN = acertos_G_MEAN + 1

                '''
                if( continuar_peso_ponderado_comite and  preveuPeso_ponderado_comite == real_y_val2_classificar):
                    #print("acertou HARD")
                    acertos_peso_ponderado_comite = acertos_peso_ponderado_comite + 1

                if( continuar_sum_weight and  preveuSum_weight == real_y_val2_classificar):
                    #print("acertou HARD")
                    acertos_sum_weight = acertos_sum_weight + 1
                '''
                if( continuar_rede_neural and  preveuRede_neural == real_y_val2_classificar):
                    #print("acertou HARD")
                    acertos_rede_neural = acertos_rede_neural + 1

                if( continuar_rede_neural_soft and  preveuRede_neural_soft == real_y_val2_classificar):
                    #print("acertou HARD")
                    acertos_rede_neural_soft = acertos_rede_neural_soft + 1

                if( continuar_rede_neural_soft_div and  preveuRede_neural_soft_class == real_y_val2_classificar):
                    #print("acertou HARD")
                    acertos_rede_neural_soft_div = acertos_rede_neural_soft_div + 1

                if( continuar_borda and  preveuBorda == real_y_val2_classificar):
                    #print("acertou HARD")
                    acertos_borda = acertos_borda + 1
                    
                if( continuar_naive_bayes and  preveuNaive_bayes == real_y_val2_classificar):
                    #print("acertou HARD")
                    acertos_naive_bayes = acertos_naive_bayes + 1                    
                    

                #ESCOLHER (se) QUEM MAIS ACERTOU 
                if qt_vizinhos_classificados >= vizinhos_val:
                    '''
                    if acertos_MAX > max( acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        #print("############################# ESCOLHEU MAX ############################")
                        resultados.append(modelMAX.predict_proba([a_classificar_X_teste]))
                        escolheu_MAX = escolheu_MAX + 1
                        break
                    '''
                    if acertos_HARD > max(acertos_MAX, acertos_SOFT, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        #print("############################# ESCOLHEU HARD ############################")
                        resultados.append(modelHARD.predict_proba([a_classificar_X_teste]))
                        escolheu_HARD = escolheu_HARD + 1
                        break
                    if acertos_SOFT > max(acertos_MAX, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        #print("############################# ESCOLHEU SOFT ############################")
                        resultados.append(modelSOFT.predict_proba([a_classificar_X_teste]))
                        escolheu_SOFT = escolheu_SOFT + 1
                        break
                    '''
                    if acertos_MIN > max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        resultados.append(modelMIN.predict_proba([a_classificar_X_teste]))
                        escolheu_MIN = escolheu_MIN + 1
                        break
                    '''
                    if acertos_G_MEAN > max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        resultados.append(modelG_MEAN.predict_proba([a_classificar_X_teste]))
                        escolheu_G_MEAN = escolheu_G_MEAN + 1
                        break
                    '''
                    if acertos_peso_ponderado_comite > max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        resultados.append(modelPeso_ponderado_comite.predict_proba([a_classificar_X_teste]))
                        escolheu_peso_ponderado_comite = escolheu_peso_ponderado_comite + 1
                        break
                    if acertos_sum_weight > max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        resultados.append(modelSum_weight.predict_proba([a_classificar_X_teste]))
                        escolheu_sum_weight = escolheu_sum_weight + 1
                        break
                    '''
                    if acertos_rede_neural > max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        resultados.append(model_Rede_neural.predict_proba([a_classificar_X_teste]))
                        escolheu_rede_neural = escolheu_rede_neural + 1
                        break
                    if acertos_rede_neural_soft > max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        resultados.append(modelRede_neural_soft.predict_proba([a_classificar_X_teste]))
                        escolheu_rede_neural_soft = escolheu_rede_neural_soft + 1
                        break
                    if acertos_rede_neural_soft_div > max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_borda, acertos_naive_bayes ):
                        resultados.append(modelRede_neural_soft_div.predict_proba([a_classificar_X_teste]))
                        escolheu_rede_neural_soft_div = escolheu_rede_neural_soft_div + 1
                        break
                    if acertos_borda > max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_naive_bayes ):
                        resultados.append(modelBorda.predict_proba([a_classificar_X_teste]))
                        escolheu_borda = escolheu_borda + 1
                        break
                    if acertos_naive_bayes > max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda ):
                        resultados.append(modelNaive_bayes.predict_proba([a_classificar_X_teste]))
                        escolheu_naive_bayes = escolheu_naive_bayes + 1
                        break
                    


                    #DESABILITAR QUEM ERROU E ALGUEM ACERTOU
                    '''
                    if acertos_MAX < max( acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        continuar_MAX = False
                    '''
                    if acertos_HARD < max(acertos_MAX, acertos_SOFT, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        continuar_HARD = False
                    if acertos_SOFT < max(acertos_MAX, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        continuar_SOFT = False
                    '''
                    if acertos_MIN < max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        continuar_MIN = False
                    '''
                    if acertos_G_MEAN < max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        continuar_G_MEAN = False
                    '''
                    if acertos_peso_ponderado_comite < max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        continuar_peso_ponderado_comite = False
                    if acertos_sum_weight < max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        continuar_sum_weight = False
                    '''
                    if acertos_rede_neural < max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        continuar_rede_neural = False
                    if acertos_rede_neural_soft < max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft_div, acertos_borda, acertos_naive_bayes ):
                        continuar_rede_neural_soft = False
                    if acertos_rede_neural_soft_div < max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_borda, acertos_naive_bayes ):
                        continuar_rede_neural_soft_div = False
                    if acertos_borda < max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_naive_bayes ):
                        continuar_borda = False
                    if acertos_naive_bayes < max(acertos_MAX, acertos_SOFT, acertos_HARD, acertos_MIN, acertos_G_MEAN, acertos_peso_ponderado_comite, acertos_sum_weight, acertos_rede_neural, acertos_rede_neural_soft, acertos_rede_neural_soft_div, acertos_borda ):
                        continuar_naive_bayes = False

                    
                    #print("---")
                    if qt_vizinhos_classificados == len(y_val2):
                        #print("############## Chegou ao final ###########################")

                        if continuar_HARD:
                            resultados.append(modelHARD.predict_proba([a_classificar_X_teste]))
                            escolheu_HARD = escolheu_HARD + 1
                            #print("############################# ESCOLHEU HARD ############################")
                            break
                        elif continuar_SOFT:
                            resultados.append(modelSOFT.predict_proba([a_classificar_X_teste]))
                            escolheu_SOFT = escolheu_SOFT + 1
                            #print("############################# ESCOLHEU SOFT ############################")
                            break
                            '''
                            elif continuar_MAX:
                                resultados.append(modelMAX.predict_proba([a_classificar_X_teste]))
                                escolheu_MAX = escolheu_MAX + 1
                                #print("############################# ESCOLHEU MAX ############################")
                                break
                            elif continuar_MIN:
                                resultados.append(modelMIN.predict_proba([a_classificar_X_teste]))
                                escolheu_MIN = escolheu_MIN + 1
                                break
                            '''
                        elif continuar_G_MEAN:
                            resultados.append(modelG_MEAN.predict_proba([a_classificar_X_teste]))
                            escolheu_G_MEAN = escolheu_G_MEAN + 1
                            break
                            '''
                            elif continuar_peso_ponderado_comite:
                                resultados.append(modelPeso_ponderado_comite.predict_proba([a_classificar_X_teste]))
                                escolheu_peso_ponderado_comite = escolheu_peso_ponderado_comite + 1
                                break
                            elif continuar_sum_weight:
                                resultados.append(modelSum_weight.predict_proba([a_classificar_X_teste]))
                                escolheu_sum_weight = escolheu_sum_weight + 1
                                break
                            '''
                        elif continuar_rede_neural:
                            resultados.append(model_Rede_neural.predict_proba([a_classificar_X_teste]))
                            escolheu_rede_neural = escolheu_rede_neural + 1
                            break
                        elif continuar_rede_neural_soft:
                            resultados.append(modelRede_neural_soft.predict_proba([a_classificar_X_teste]))
                            escolheu_rede_neural_soft = escolheu_rede_neural_soft + 1
                            break
                        elif continuar_rede_neural_soft_div:
                            resultados.append(modelRede_neural_soft_div.predict_proba([a_classificar_X_teste]))
                            escolheu_rede_neural_soft_div = escolheu_rede_neural_soft_div + 1
                            break
                        elif continuar_borda:
                            resultados.append(modelBorda.predict_proba([a_classificar_X_teste]))
                            escolheu_borda = escolheu_borda + 1
                            break
                        else:
                            resultados.append(modelNaive_bayes.predict_proba([a_classificar_X_teste]))
                            escolheu_naive_bayes = escolheu_naive_bayes + 1
                            #print("empate")
                            break

                        #print("ErrO")

                                                      
                            


                
                    

            


            



 





        '''
        preds[inds] = preds_ds

        return self.classes_.take(preds)
        '''


        #print("RESULTADOS")
        #print(resultados)

        #print(np.concatenate(resultados))
        #print(resultados)
        #return resultados
        #print("Resultado função juntado")
        #print(np.concatenate(resultados))



        escolheu_TOTAL = sum([escolheu_borda, escolheu_naive_bayes, escolheu_peso_ponderado_comite,escolheu_rede_neural_soft_div, escolheu_MAX,escolheu_SOFT,escolheu_HARD, escolheu_MIN, escolheu_G_MEAN, escolheu_sum_weight, escolheu_rede_neural,escolheu_rede_neural_soft, ])

        porcentagem_escolheu_MAX.append((escolheu_MAX / escolheu_TOTAL) * 100)
        porcentagem_escolheu_SOFT.append((escolheu_SOFT / escolheu_TOTAL) * 100)
        porcentagem_escolheu_HARD.append((escolheu_HARD / escolheu_TOTAL) * 100)
        porcentagem_escolheu_MIN.append((escolheu_MIN / escolheu_TOTAL) * 100)
        porcentagem_escolheu_G_MEAN.append((escolheu_G_MEAN / escolheu_TOTAL) * 100)

        porcentagem_escolheu_sum_weight.append((escolheu_sum_weight / escolheu_TOTAL) * 100)
        #porcentagem_escolheu_sum_weight_line.append((escolheu_sum_weight_line / escolheu_TOTAL) * 100)
        #porcentagem_escolheu_peso_ponderado_classe.append((escolheu_peso_ponderado_classe / escolheu_TOTAL) * 100)
        porcentagem_escolheu_peso_ponderado_comite.append((escolheu_peso_ponderado_comite / escolheu_TOTAL) * 100)

        porcentagem_escolheu_rede_neural.append((escolheu_rede_neural / escolheu_TOTAL) * 100)
        porcentagem_escolheu_rede_neural_soft.append((escolheu_rede_neural_soft / escolheu_TOTAL) * 100)
        porcentagem_escolheu_rede_neural_soft_div.append((escolheu_rede_neural_soft_div / escolheu_TOTAL) * 100)
        porcentagem_escolheu_borda.append((escolheu_borda / escolheu_TOTAL) * 100)
        porcentagem_escolheu_naive_bayes.append((escolheu_naive_bayes / escolheu_TOTAL) * 100)


        quantidade_exemplos_divergencia.append(len(X_test_divergente))
        quantidade_classificadores_selecionados.append(qnt_class_sel)
        #print("Q exemplos divergentes")
        #print(len(X_test_divergente))

        #print("Q class selec")
        #print(qnt_class_sel)

        '''
        print(escolheu_MAX)
        print(escolheu_SOFT)
        print(escolheu_HARD)
        print(escolheu_MIN)
        print(escolheu_G_MEAN)
        '''

        #print("Devolve")
        #print (np.concatenate(resultados))

        return np.concatenate(resultados)
        '''
        #lista de vizinhos proximos das classes que serão classificadas, os valores estão na base de teste
        list_neighbors_train_index = np.unique(self.dynamic_neighbors_index_train, return_counts=False)
        print("VALORES TRAIN PARA CLASSIFICAR")
        print(list_neighbors_train_index)
        #from sklearn.neighbors import KNeighborsClassifier
        #neigh = KNeighborsClassifier(n_neighbors=3, n_jobs = -1)
        #neigh.fit(X_train, y_train)
        samples_neighbors_X = X_train.iloc[list_neighbors_train_index]
        samples_neighbors_y = y_train.iloc[list_neighbors_train_index]
        print(samples_neighbors_X)
        print(samples_neighbors_y)
        #print("kneighbors")
        #print(neigh.kneighbors(X=samples_neighbors_X, n_neighbors=None, return_distance=False))
        '''
    def aleatoria(self, probabilities, selected_classifiers,predictions):#Devolve as probabilidades mas remove dos classificadores não competentes
        total_exemplos = len(probabilities)
        #print("FUNÇÃO ALEATORIO")
        #print("probabilities")
        #print(probabilities)
        #print("selected_classifiers")
        #print(selected_classifiers)
        #print("predictions")
        #print(predictions)

        probabilities_lista_dividida = np.array_split(probabilities,total_exemplos)
        selected_classifiers_lista_dividida = np.array_split(selected_classifiers,total_exemplos)
        predictions_lista_dividida = np.array_split(predictions,total_exemplos)
        #print("probabilities_lista_dividida")
        #print(probabilities_lista_dividida)
        #print("selected_classifiers_lista_dividida")
        #print(selected_classifiers_lista_dividida)
        #print("predictions_lista_dividida")
        #print(predictions_lista_dividida)

        import random
        from random import randrange
        #print("Juntar")
        #print(np.concatenate(probabilities_lista_dividida))
        resultados = []
        for prob, selec_classf, pred  in zip(probabilities_lista_dividida, selected_classifiers_lista_dividida, predictions_lista_dividida):
            '''
            print(self._max_proba(prob, selec_classf))
            print("Resultado juntar enm cima")
            '''
            #print("CHAMOU FUNÇÃO")
            #x = random.choice([ self._max_proba(prob, selec_classf), self._mask_proba(prob, selec_classf)])
            sort = randrange(3)

            if sort == 0:
                #print("self._max_proba(prob, selec_classf)")
                resultados.append(self._max_proba(prob, selec_classf))
            elif sort == 1:
                #print("self._mask_proba(prob, selec_classf)")
                resultados.append(self._mask_proba(prob, selec_classf))
            elif sort == 2:
                #print("self._minimun_proba(prob, selec_classf)")
                resultados.append(self._minimun_proba(prob, selec_classf))
            elif sort == 3:
                #print("self.geometric_mean(prob, selec_classf)")
                resultados.append(self.geometric_mean(prob, selec_classf))
            else:
                votes = np.ma.MaskedArray(pred, ~selec_classf)
                resultados.append(sum_votes_per_class(votes, self.n_classes_))
        #print("RESULTADOS")
        #print(resultados)
        #print("Resultado função aleatoria juntado")
        #print(np.concatenate(resultados))
        return  np.concatenate(resultados)

    def _max_proba(self, probabilities, selected_classifiers):#Devolve as probabilidades mas remove dos classificadores não competentes
        #print("FUNÇÃO MAXIMO DA PROBABILIDADE")
        selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
        # coloca mais um [] dentro dos elemendos [1 2 3]  => [ [1] [2] [3]]
        #print("selected_classifiers in function _max_proba using np.expand_dims")
        #print(selected_classifiers)
        selected_classifiers = np.broadcast_to(selected_classifiers, probabilities.shape)
        #Tipo duplica os valores de acordo  com o segundo parametro, OBS que agora está do tamanho de  probabilidades e igual
        #print("selected_classifiers in function _max_proba using np.broadcast_to")
        #print(selected_classifiers)
        masked_proba = np.ma.MaskedArray(probabilities, ~selected_classifiers)
        #print("masked_proba in function _max_proba using np.ma.MaskedArray")
        #print(masked_proba)

        predicted_proba = np.amax(masked_proba, axis=1)
        #print("predicted_proba in function _max_proba using np.amax")
        #print(predicted_proba)

        normalizar_linha = 1 / predicted_proba.sum(axis=1)[:, None]
        #print("normalizar_linha in function _max_proba using 1 / predicted_proba.sum(axis=1)[:, None]")
        #print(normalizar_linha)
        predicted_proba = predicted_proba * normalizar_linha

        #print("predicted_proba resultado")
        #print(predicted_proba)

        return predicted_proba

    def _minimun_proba(self, probabilities, selected_classifiers):#Devolve as probabilidades mas remove dos classificadores não competentes
        #print("FUNÇÃO MINIMO DA PROBABILIDADE")
        selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
        # coloca mais um [] dentro dos elemendos [1 2 3]  => [ [1] [2] [3]]
        #print("selected_classifiers in function _minimun_proba using np.expand_dims")
        #print(selected_classifiers)
        selected_classifiers = np.broadcast_to(selected_classifiers, probabilities.shape)
        #Tipo duplica os valores de acordo  com o segundo parametro, OBS que agora está do tamanho de  probabilidades e igual
        #print("selected_classifiers in function _minimun_proba using np.broadcast_to")
        #print(selected_classifiers)
        masked_proba = np.ma.MaskedArray(probabilities, ~selected_classifiers)
        #print("masked_proba in function _minimun_proba using np.ma.MaskedArray")
        #print(masked_proba)

        predicted_proba = np.amin(masked_proba, axis=1)
        #print("predicted_proba in function _minimun_proba using np.amax")
        #print(predicted_proba)

        normalizar_linha = 1 / predicted_proba.sum(axis=1)[:, None]
        #print("normalizar_linha in function _minimun_proba using 1 / predicted_proba.sum(axis=1)[:, None]")
        #print(normalizar_linha)
        predicted_proba = predicted_proba * normalizar_linha

        #print("predicted_proba resultado")
        #print(predicted_proba)

        return predicted_proba

    def geometric_mean(self, probabilities, selected_classifiers):#Devolve as probabilidades mas remove dos classificadores não competentes
        #print("FUNÇÃO GEOMETRIC MEAN")
        selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
        # coloca mais um [] dentro dos elemendos [1 2 3]  => [ [1] [2] [3]]
        #print("selected_classifiers in function media_geometrica using np.expand_dims")
        #print(selected_classifiers)
        selected_classifiers = np.broadcast_to(selected_classifiers, probabilities.shape)
        #Tipo duplica os valores de acordo  com o segundo parametro, OBS que agora está do tamanho de  probabilidades e igual
        #print("selected_classifiers in function media_geometrica using np.broadcast_to")
        #print(selected_classifiers)
        masked_proba = np.ma.MaskedArray(probabilities, ~selected_classifiers)
        #print("masked_proba in function media_geometrica using np.ma.MaskedArray")
        #print(masked_proba)


        predicted_proba = np.prod(masked_proba, axis=1)
        #print("predicted_proba in function media_geometrica using np.prod(masked_proba, axis=1)")
        #print(predicted_proba)



        normalizar_linha = 1 / predicted_proba.sum(axis=1)[:, None]
        #print("normalizar_linha in function media_geometrica using 1 / predicted_proba.sum(axis=1)[:, None]")
        #print(normalizar_linha)
        predicted_proba = predicted_proba * normalizar_linha

        #print("predicted_proba resultado")
        #print(predicted_proba)

        return predicted_proba



    #### SALVO ##########
    #Ponderação por peso
    def _peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1(self, probabilities, selected_classifiers):#Devolve as probabilidades mas remove dos classificadores não competentes
        #Ponderação por peso de classe A(0,6) B(0,4) x probabilidade do comite. Os pesos de cada classe se alteram em cada amostra de teste de acordo com a media dos vizinhos daquela classea a distancia usada é do maior vizinho das amostras de teste. Os valores estão entre 0 e 1 formatados, quanto mais proximo mais alto o peso


        #print("FUNÇÃO PESO PONDERADO DISTANCIA")
        # Broadcast the selected classifiers mask
        # to cover the last axis (n_classes):
        selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
        # coloca mais um [] dentro dos elemendos [1 2 3]  => [ [1] [2] [3]]
        #print("selected_classifiers in function _mask_proba using np.expand_dims")
        #print(selected_classifiers)
        selected_classifiers = np.broadcast_to(selected_classifiers, probabilities.shape)
        #Tipo duplica os valores de acordo  com o segundo parametro, OBS que agora está do tamanho de  probabilidades e igual
        #print("selected_classifiers in function _mask_proba using np.broadcast_to")
        #print(selected_classifiers)
        masked_proba = np.ma.MaskedArray(probabilities, ~selected_classifiers)
        #retorna a mastriz original de probabiliade mas remove os valores dos classsificadores não selecionados
        #print("masked_proba in function _mask_proba using np.ma.MaskedArray")
        #print(masked_proba)

        predicted_proba = np.mean(masked_proba, axis=1)
        #print("predicted_proba resultado")
        #print(predicted_proba)
        '''
        print("dynamic_index_usage_neighbors_test")
        print(self.dynamic_index_usage_neighbors_test )
        print("dynamic_neighbors_index_train")
        print(self.dynamic_neighbors_index_train)
        print("dynamic_neighbors_distances_test")
        print(self.dynamic_neighbors_distances_test)
        '''
        max_distance = max(np.unique(self.dynamic_neighbors_distances_test))
        #print("max_distance")
        #print(max_distance)
        #row_kneighbors_teste =
        # self.classes_.take(preds)
        #print("self.classes_")
        classes = self.classes_

        qt_classes = len(self.classes_)

        predicted_pesos = np.zeros(predicted_proba.shape)
        #print("self.dynamic_neighbors_index_train")
        #print(self.dynamic_neighbors_index_train)
        x, y = self.dynamic_neighbors_index_train.shape
        pesos_pela_distancia_para_classes = []
        for i in range(x):
            cont_class_neighbors_line = [0] * qt_classes
            distance_class_neighbors_line = [0] * qt_classes
            quant_div_distance_class_neighbors_line = [0] * qt_classes
            #print("cont_class_neighbors_line")
            #print(cont_class_neighbors_line)
            for j in range(y):
                for l in range(qt_classes):
                    '''
                    print("self.dynamic_neighbors_index_train[i,j]")
                    print(self.dynamic_neighbors_index_train[i,j])
                    print("y_train[j] == classes[l]")#em baixo o erro
                    print(y_train.iloc[self.dynamic_neighbors_index_train[i,j]])
                    print("1")
                    print(classes[l])
                    print("2")
                    '''
                    if y_train.iloc[self.dynamic_neighbors_index_train[i,j]] == classes[l]:
                        #cont_class_neighbors_line[l] =  cont_class_neighbors_line[l] + 1
                        #print("3")
                        distance_class_neighbors_line[l] = distance_class_neighbors_line[l] + self.dynamic_neighbors_distances_test[i,j]
                        #print("4")
                        quant_div_distance_class_neighbors_line[l] = quant_div_distance_class_neighbors_line[l] + 1
                        #print("5")
            #print("JOTA SAI COM " + str(j))
            #print("distance_class_neighbors_line: " + str(distance_class_neighbors_line) )
            #print([(j+1)] * qt_classes)
            vizinhos_total_lista = ([(j+1)] * qt_classes)
            quant_div_distance_class_neighbors_line = [1 if value==0 else value for value in quant_div_distance_class_neighbors_line]# colocar o 1 para evitar divisão por 0
            mean_distance_class_neighbors_line = np.array(distance_class_neighbors_line) / np.array(quant_div_distance_class_neighbors_line)
            #print("mean_distance_class_neighbors_line: "+str(mean_distance_class_neighbors_line))
            #weight_distane_class_line = max_distance - mean_distance_class_neighbors_line
            #trocado por
            weight_distane_class_line = 1 / mean_distance_class_neighbors_line

            #print("weight_distane_class_line")
            #print(weight_distane_class_line)
            pesos_pela_distancia_para_classes.append(weight_distane_class_line)
            #print("-------------------------------------------------------")

            try:
                predicted_pesos[i] = predicted_proba[i] * (weight_distane_class_line * (1/np.sum(weight_distane_class_line)))
            except:
                pass

            #ou
            #predicted_pesos[i] = predicted_proba[i] * weight_distane_class_line
        #print("pesos_pela_distancia_para_classes")
        #print(pesos_pela_distancia_para_classes)
        #print("Resultado")
        #print(predicted_pesos)

        predicted_pesos = predicted_pesos / predicted_pesos.sum(axis=1)[:, None]
        #print("predicted_pesos resultado devolvido")
        #print(predicted_pesos)

        return predicted_pesos



    ############# SALVO #####
    def _sum_weight_0a1_votes_per_class(self, predictions, n_classes):
        #Votação ponderada por distancia em relação ao vizinho de teste mais distante (Geral dos teste)
        #Peso formatado aplicado na votação que o comitê deu
        #print("FUNÇÃO VOTO COM PESA POR CLASSE")
        """Sum the number of votes for each class. Accepts masked arrays as input.
        Parameters
        ----------
        sum_weight_votes_per_class
        predictions : array of shape (n_samples, n_classifiers),
            The votes obtained by each classifier for each sample. Can be a masked
            array.
        n_classes : int
            Number of classes.
        Returns
        -------
        summed_votes : array of shape (n_samples, n_classes)
            Summation of votes for each class
        """
        #from sklearn.neighbors import KNeighborsClassifier
        #neigh = KNeighborsClassifier(n_neighbors=5, n_jobs = 1)
        #neigh.fit(X_val, y_val)
        #row_kneighbors_validation = neigh.kneighbors(X=X_test, n_neighbors=None, return_distance=False)
        #print("Vizinhos do Teste (os vizinhos estão no validation)")
        #print(row_kneighbors_validation)
        #lista de vizinhos proximos das classes que serão classificadas, os valores estão na base de teste
        #row_unique_kneighbors_validation = np.unique(row_kneighbors_validation, return_counts=False)
        #print("VALORES PARA CLASSIFICAR X")
        #MELHORAR AINDA ESTA CONSUMINDO MUITO
        #X_neighbors_val = X_val.iloc[row_unique_kneighbors_validation]
        #y_neighbors_val = y_val.iloc[row_unique_kneighbors_validation]
        '''
        print("dynamic_index_usage_neighbors_test")
        print(self.dynamic_index_usage_neighbors_test )
        print("dynamic_neighbors_index_train")
        print(self.dynamic_neighbors_index_train)
        print("dynamic_neighbors_distances_test")
        print(self.dynamic_neighbors_distances_test)
        '''
        max_distance = max(np.unique(self.dynamic_neighbors_distances_test))
        #print("max_distance")
        #print(max_distance)
        #row_kneighbors_teste =
        # self.classes_.take(preds)
        #print("self.classes_")
        classes = self.classes_






        '''
        print("predictions sum votes")
        print(predictions)
        print("n_classes sum votes")
        print(n_classes)
        '''
        votes = np.zeros((predictions.shape[0], n_classes), dtype=np.int64)
        #print("Votes instance sum votes (Cria amatriz zerada)")
        #print(votes)
        for label in range(n_classes):
            votes[:, label] = np.sum(predictions == label, axis=1)#Conta label nas coluna(eixo 1)
        #print("Votos majory resultado")
        #print(votes)
        '''
        predicted_proba = votes / votes.sum(axis=1)[:, None]
        print("RESULTADO predicted_proba")
        print(predicted_proba)
        '''
        #print("RESULTADO predicted_proba")
        #print(votes / votes.sum(axis=1)[:, None])
        #ponderacao = predicted_proba
        qt_classes = len(self.classes_)

        predicted_pesos = np.zeros(votes.shape)

        x, y = self.dynamic_neighbors_index_train.shape
        pesos_pela_distancia_para_classes = []
        for i in range(x):
            cont_class_neighbors_line = [0] * qt_classes
            distance_class_neighbors_line = [0] * qt_classes
            quant_div_distance_class_neighbors_line = [0] * qt_classes
            #print("cont_class_neighbors_line")
            #print(cont_class_neighbors_line)
            for j in range(y):
                for l in range(qt_classes):
                    #print("y_train[j] == classes[l]")
                    #print(y_train.iloc[self.dynamic_neighbors_index_train[i,j]])
                    #print(classes[l])
                    if y_train.iloc[self.dynamic_neighbors_index_train[i,j]] == classes[l]:
                        #cont_class_neighbors_line[l] =  cont_class_neighbors_line[l] + 1
                        distance_class_neighbors_line[l] = distance_class_neighbors_line[l] + self.dynamic_neighbors_distances_test[i,j]
                        quant_div_distance_class_neighbors_line[l] = quant_div_distance_class_neighbors_line[l] + 1
            #print("JOTA SAI COM " + str(j))
            #print("distance_class_neighbors_line: " + str(distance_class_neighbors_line) )
            #print([(j+1)] * qt_classes)
            vizinhos_total_lista = ([(j+1)] * qt_classes)
            quant_div_distance_class_neighbors_line = [1 if value==0 else value for value in quant_div_distance_class_neighbors_line]
            mean_distance_class_neighbors_line = np.array(distance_class_neighbors_line) / np.array(quant_div_distance_class_neighbors_line)
            #print("mean_distance_class_neighbors_line: "+str(mean_distance_class_neighbors_line))
            ##############weight_distane_class_line = max_distance - mean_distance_class_neighbors_line############
            weight_distane_class_line = 1 / mean_distance_class_neighbors_line

            #print("weight_distane_class_line")
            #print(weight_distane_class_line)
            pesos_pela_distancia_para_classes.append(weight_distane_class_line)
            #print("-------------------------------------------------------")

            #predicted_pesos[i] = votes[i] * weight_distane_class_line
            #OU
            try:
                predicted_pesos[i] = votes[i] * weight_distane_class_line * (1/np.sum(weight_distane_class_line))
            except:
                pass
            #predicted_pesos[i] = predicted_proba[i] * weight_distane_class_line * (1/np.sum(weight_distane_class_line))

        #print("pesos_pela_distancia_para_classes")
        #print(pesos_pela_distancia_para_classes)
        #print("Resultado")
        #print(predicted_pesos)

        predicted_pesos = predicted_pesos / predicted_pesos.sum(axis=1)[:, None]
        #print("predicted_pesos")
        #print(predicted_pesos)
        return predicted_pesos


    # Borda por meio de Votação



    def naive_bayes_combination(self, predictions, n_classes):
        #print("FUNÇÃO naive_bayes_combination")
        """Sum the number of votes for each class. Accepts masked arrays as input.
        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers),
            The votes obtained by each classifier for each sample. Can be a masked
            array.
        n_classes : int
            Number of classes.
        Returns
        -------
        summed_votes : array of shape (n_samples, n_classes)
            Summation of votes for each class
        """
        from sklearn.metrics import confusion_matrix
        m_confusoes = []
        for m in range(len(self.pool_classifiers)):
            m_confusoes.append(confusion_matrix(y_val, self.pool_classifiers[m].predict(X_val),labels=self.classes_))
        cont_elementos = [0] * len(self.classes_)
        for i in range(len(self.classes_)):
            for v in range(len(y_val)):
                if(y_val.iloc[v] == self.classes_[i]):
                    cont_elementos[i] = cont_elementos[i] + 1
        xxx, yyy = predictions.shape
        predicted_proba_resultado = []
        for x in range(xxx):
            resultado = [1] * len(self.classes_)
            for y in range(yyy):
                if (predictions[x,y] >= 0):
                    matriz_c = m_confusoes[y]
                    #print(matriz_c)
                    #print(matriz_c[:,predictions[x,y]])
                    resultado = resultado * matriz_c[:,predictions[x,y]]
            class_resultado = resultado / cont_elementos
            if(class_resultado.sum() >0 ):#Erro de quando for 0
                predicted_pesos = class_resultado / class_resultado.sum()
            else:
                predicted_pesos = [(1/len(self.classes_))] * len(self.classes_)
            predicted_proba_resultado.append([predicted_pesos])
        #return predicted_proba_resultado
        return np.concatenate(predicted_proba_resultado)


    def rede_neural_class(self, probabilities, selected_classifiers):#Devolve as probabilidades mas remove dos classificadores não competentes
        #rede_neural_class
        #print("FUNÇÃO rede_neural_class")
        #print(probabilities)
        slclsf = selected_classifiers
        pbb = probabilities
        selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
        # coloca mais um [] dentro dos elemendos [1 2 3]  => [ [1] [2] [3]]
        #print("selected_classifiers in function _max_proba using np.expand_dims")
        #print(selected_classifiers)
        selected_classifiers = np.broadcast_to(selected_classifiers, probabilities.shape)
        #Tipo duplica os valores de acordo  com o segundo parametro, OBS que agora está do tamanho de  probabilidades e igual
        #print("selected_classifiers in function _max_proba using np.broadcast_to")
        #print(selected_classifiers)
        masked_proba = np.ma.MaskedArray(probabilities, ~selected_classifiers)
        #print("masked_proba in function _max_proba using np.ma.MaskedArray")
        #print(masked_proba)
        '''
        x, y, z = masked_proba.shape
        #Grupos(exemplos), lindas por grupos(classificadores), colunas(classes)
        #print("x y z")
        #print(x,y,z)
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    max_linha = np.amax(masked_proba[i,j])
                    for sub in range(z):
                        if masked_proba[i,j,sub] == max_linha :
                            masked_proba[i,j,sub] = (z - k -1) * -1
                            break

        #print("masked_proba no final")
        #print(masked_proba)

        #print("masked_proba no final")
        #print(masked_proba * -1)
        masked_proba = masked_proba * -1

        #print("Outra")
        #print(np.sum(masked_proba,axis=1))

        masked_proba = np.sum(masked_proba,axis=1)

        #print(" masked_proba.sum(axis=1)[:, None]")
        #print( masked_proba.sum(axis=1)[:, None])

        #print(" masked_proba.sum(axis=1)")
        #print( masked_proba.sum(axis=1))


        predicted_pesos = masked_proba / masked_proba.sum(axis=1)[:, None]
        print("predicted_pesos")
        print(predicted_pesos)

        classPegarPeso = self
        classPegarPeso.voting_type = "hard"
        classPegarPeso.pesos_classificadores()
        '''

        modelHARD = MyKnoraE(pool_classifiers=self.pool_classifiers, k=5, voting_type="hard" )
        modelHARD.fit(X_train,y_train )
        model_predictionsHARD = modelHARD.predict(X_val)#predizer dados de treino
        #proba_HARD =  modelHARD.predict_proba(X_test)

        #self.previsoes_real
        #self.previsoes_classificadores = predictions

        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(200,), solver='adam', alpha=0.0001, max_iter=1000, random_state=1)
        #clf = MLPClassifier(hidden_layer_sizes=(200,), solver='adam', alpha=0.0001, random_state=1)
        #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        try:
            clf.fit(modelHARD.previsoes_classificadores, modelHARD.previsoes_real)
            #print("Real")
            #print(y_test.iloc[self.dynamic_index_usage_neighbors_test])
            #print("Previsto")
            #print(clf.predict(self.previsoes_classificadores))
            #print("Probabilidade")
            predicted_pesos  = clf.predict_proba(self.previsoes_classificadores)
            #predicted_pesos  = clf.predict_proba()
        except:
            predicted_pesos = np.array([[0]])




        x, y, = predicted_pesos.shape
        if(len(self.classes_) > y):
            #print("Ajuste rede 1")
            df_dados_x = pd.DataFrame()
            x = 1
            for pool in self.pool_classifiers:
               df_dados_x[str(x)] = pool.predict(X_val)
               x = x + 1

            clf = MLPClassifier(hidden_layer_sizes=(200,), solver='adam', alpha=0.0001, max_iter=1000, random_state=1)
            clf.fit(df_dados_x, y_val)
            predicted_pesos  = clf.predict_proba(self.previsoes_classificadores)

        #print (predicted_pesos)

        x, y, = predicted_pesos.shape
        if(len(self.classes_) > y):
            #print(y_val)
            predicted_pesos = self._mask_proba(pbb, slclsf)
        return predicted_pesos


    def rede_neural_soft(self, probabilities, selected_classifiers):#Devolve as probabilidades mas remove dos classificadores não competentes
        slclsf = selected_classifiers
        pbb = probabilities
        #rede_neural_class
        #print("FUNÇÃO rede_neural_soft")
        #print(probabilities)
        selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
        # coloca mais um [] dentro dos elemendos [1 2 3]  => [ [1] [2] [3]]
        #print("selected_classifiers in function _max_proba using np.expand_dims")
        #print(selected_classifiers)
        selected_classifiers = np.broadcast_to(selected_classifiers, probabilities.shape)
        #Tipo duplica os valores de acordo  com o segundo parametro, OBS que agora está do tamanho de  probabilidades e igual
        #print("selected_classifiers in function _max_proba using np.broadcast_to")
        #print(selected_classifiers)
        masked_proba = np.ma.MaskedArray(probabilities, ~selected_classifiers)
        #print("masked_proba in function _max_proba using np.ma.MaskedArray")
        #print(masked_proba)

        try:
            modelSOFT = MyKnoraE(pool_classifiers=self.pool_classifiers, k=5, voting_type="soft" )
            modelSOFT.fit(X_train,y_train )
            model_predictionsSOFT = modelSOFT.predict(X_val)#predizer dados de treino

            #print("modelSOFT.probabilidades_classificadores")
            #print(modelSOFT.probabilidades_classificadores)
            xxx, yyy, zzz = modelSOFT.probabilidades_classificadores.shape
            lista_treinamento = []
            for x in range(xxx):
                #for y in range(yyy):
                #print(modelSOFT.probabilidades_classificadores[x, y])
                sub = modelSOFT.probabilidades_classificadores[x].flatten().tolist()
                for z in range(len(sub)):
                    if sub[z] is None:
                        #sub[z] = 1/len(self.classes_)
                        sub[z] = -1
                lista_treinamento.append(sub)
                #De 3D para 2D
        except:
            #DEU erro ao importar os dados dentreino da rede pq esta vazio o treinamento
            pass

        from sklearn.neural_network import MLPClassifier

        xxx, yyy, zzz = masked_proba.shape
        masked_proba2D = []
        for x in range(xxx):
            sub = masked_proba[x].flatten().tolist()
            for z in range(len(sub)):
                if sub[z] is None:
                    #sub[z] = 1/len(self.classes_)
                    sub[z] = -1
            masked_proba2D.append(sub)

            #De 3D para 2D

        clf = MLPClassifier(hidden_layer_sizes=(200,), solver='adam', alpha=0.0001, max_iter=1000, random_state=1)
        try:
            #clf.fit(modelSOFT.previsoes_real)
            clf.fit(lista_treinamento, modelSOFT.previsoes_real)
            predicted_pesos  = clf.predict_proba(masked_proba2D)
        except:
            predicted_pesos = np.array([[0]])


        #print(clf.predict(masked_proba2D))
        #print("Probabilidade")
        #print(predicted_pesos)


        #clf.predict([[2., 2.], [-1., -2.]])




        x, y, = predicted_pesos.shape
        if(len(self.classes_) > y):
            #print("Ajuste rede 2")
            df_dados_x = []
            #print(X_val)
            #print("fim")
            for amostra in range(len(X_val)):
                #print(X_val.iloc[amostra:amostra+1,:])
                sub = np.array([])
                for pool in self.pool_classifiers:
                    p = pool.predict_proba(X_val.iloc[amostra:amostra+1,:]).flatten().tolist()
                    sub = np.append(sub, p)
                    #print("saiu")
                dados_amostra = sub.flatten().tolist()
                #aaa, bbb, = sub.shape
                #for a in range(aaa):
                #    for b in range(bbb):
                #        dados_amostra.append(sub[a,b])
                #print(dados_amostra)
                df_dados_x.append(dados_amostra)

            #print(df_dados_x)
            clf = MLPClassifier(hidden_layer_sizes=(200,), solver='adam', alpha=0.0001, max_iter=1000, random_state=1)
            clf.fit(df_dados_x, y_val)
            predicted_pesos  = clf.predict_proba(masked_proba2D)


        x, y, = predicted_pesos.shape
        if(len(self.classes_) > y):
            #print(y_val)
            predicted_pesos = self._mask_proba(pbb, slclsf)

        return predicted_pesos

    def rede_neural_soft_div(self, probabilities, selected_classifiers):#Devolve as probabilidades mas remove dos classificadores não competentes
        slclsf = selected_classifiers
        pbb = probabilities
        #rede_neural_class
        #print("FUNÇÃO rede_neural_soft")
        #print(probabilities)
        selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
        # coloca mais um [] dentro dos elemendos [1 2 3]  => [ [1] [2] [3]]
        #print("selected_classifiers in function _max_proba using np.expand_dims")
        #print(selected_classifiers)
        selected_classifiers = np.broadcast_to(selected_classifiers, probabilities.shape)
        #Tipo duplica os valores de acordo  com o segundo parametro, OBS que agora está do tamanho de  probabilidades e igual
        #print("selected_classifiers in function _max_proba using np.broadcast_to")
        #print(selected_classifiers)
        masked_proba = np.ma.MaskedArray(probabilities, ~selected_classifiers)
        #print("masked_proba in function _max_proba using np.ma.MaskedArray")
        #print(masked_proba)

        try:
            modelSOFT = MyKnoraE(pool_classifiers=self.pool_classifiers, k=5, voting_type="soft" )
            modelSOFT.fit(X_train,y_train )
            model_predictionsSOFT = modelSOFT.predict(X_val)#predizer dados de treino

            #print("modelSOFT.probabilidades_classificadores")
            #print(modelSOFT.probabilidades_classificadores)
            xxx, yyy, zzz = modelSOFT.probabilidades_classificadores.shape
            lista_treinamento = []
            for x in range(xxx):
                #for y in range(yyy):
                #print(modelSOFT.probabilidades_classificadores[x, y])
                sub = modelSOFT.probabilidades_classificadores[x].flatten().tolist()
                for z in range(len(sub)):
                    if sub[z] is None:
                        sub[z] = 1/len(self.classes_)
                lista_treinamento.append(sub)
                #De 3D para 2D
        except:
            #DEU erro ao importar os dados dentreino da rede pq esta vazio o treinamento
            pass

        xxx, yyy, zzz = masked_proba.shape
        masked_proba2D = []
        for x in range(xxx):
            sub = masked_proba[x].flatten().tolist()
            for z in range(len(sub)):
                if sub[z] is None:
                    sub[z] = 1/len(self.classes_)
            masked_proba2D.append(sub)

            #De 3D para 2D


        #print(clf.predict(masked_proba2D))
        #print("Probabilidade")
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(200,), solver='adam', alpha=0.0001, max_iter=1000)
        try:
            #clf.fit(modelSOFT.previsoes_real)
            clf.fit(lista_treinamento, modelSOFT.previsoes_real)
            predicted_pesos  = clf.predict_proba(masked_proba2D)
        except:
            predicted_pesos = np.array([[0]])



        x, y, = predicted_pesos.shape
        if(len(self.classes_) > y):
            #print("Ajuste rede 3")
            df_dados_x = []
            #print(X_val)
            #print("fim")
            for amostra in range(len(X_val)):
                #print(X_val.iloc[amostra:amostra+1,:])
                sub = np.array([])
                for pool in self.pool_classifiers:
                    p = pool.predict_proba(X_val.iloc[amostra:amostra+1,:]).flatten().tolist()
                    sub = np.append(sub, p)
                    #print("saiu")
                dados_amostra = sub.flatten().tolist()
                #aaa, bbb, = sub.shape
                #for a in range(aaa):
                #    for b in range(bbb):
                #        dados_amostra.append(sub[a,b])
                #print(dados_amostra)
                df_dados_x.append(dados_amostra)

            #print(df_dados_x)
            clf = MLPClassifier(hidden_layer_sizes=(200,), solver='adam', alpha=0.0001, max_iter=1000, random_state=1)
            clf.fit(df_dados_x, y_val)
            predicted_pesos  = clf.predict_proba(masked_proba2D)

        x, y, = predicted_pesos.shape
        if(len(self.classes_) > y):
            #print(y_val)
            predicted_pesos = self._mask_proba(pbb, slclsf)

        return predicted_pesos


    '''
    def _max_proba(self, probabilities, selected_classifiers):#Devolve as probabilidades mas remove dos classificadores não competentes
        print("Função maximo")
        # Broadcast the selected classifiers mask
        # to cover the last axis (n_classes):
        selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
        # coloca mais um [] dentro dos elemendos [1 2 3]  => [ [1] [2] [3]]
        print("selected_classifiers in function _mask_proba using np.expand_dims")
        print(selected_classifiers)
        selected_classifiers = np.broadcast_to(selected_classifiers, probabilities.shape)
        #Tipo duplica os valores de acordo  com o segundo parametro, OBS que agora está do tamanho de  probabilidades e igual
        print("selected_classifiers in function _mask_proba using np.broadcast_to")
        print(selected_classifiers)
        masked_proba = np.ma.MaskedArray(probabilities, ~selected_classifiers)
        #retorna a mastriz original de probabiliade mas remove os valores dos classsificadores não selecionados
        print("masked_proba in function _mask_proba using np.ma.MaskedArray")
        print(masked_proba)
        x, y, z = masked_proba.shape
        #Grupos(exemplos), lindas por grupos(classificadores), colunas(classes)
        print("x y z")
        print(x,y,z)
        for i in range(x):
            max_colun = np.amax(masked_proba[i], axis=0)
            print("max_colun")
            print(max_colun)
            sum_max_coluns = sum(max_colun)
            ajustar_valor_maximo = 1 / sum_max_coluns
            print("ajustar_valor_maximo")
            print(ajustar_valor_maximo)
            for j in range(y):
                for k in range(z):
                    print(masked_proba[i,j,k])
                    if masked_proba[i,j,k] >= 0:
                        pass
                        masked_proba[i,j,k] = max_colun[k] *ajustar_valor_maximo
        print("Novo masked_proba")
        print(masked_proba)
        #max_colun = np.amax(masked_proba, axis=0)
        #print("Maximo das colunas")
        #print(max_colun)
        return masked_proba
    '''


def borda_class(probabilities, selected_classifiers):#Devolve as probabilidades mas remove dos classificadores não competentes
    #BORDA
    #print("FUNÇÃO Borda")
    #print(probabilities)
    selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
    # coloca mais um [] dentro dos elemendos [1 2 3]  => [ [1] [2] [3]]
    #print("selected_classifiers in function _max_proba using np.expand_dims")
    #print(selected_classifiers)
    selected_classifiers = np.broadcast_to(selected_classifiers, probabilities.shape)
    #Tipo duplica os valores de acordo  com o segundo parametro, OBS que agora está do tamanho de  probabilidades e igual
    #print("selected_classifiers in function _max_proba using np.broadcast_to")
    #print(selected_classifiers)
    masked_proba = np.ma.MaskedArray(probabilities, ~selected_classifiers)
    #print("masked_proba in function _max_proba using np.ma.MaskedArray")
    #print(masked_proba)

    x, y, z = masked_proba.shape
    #Grupos(exemplos), lindas por grupos(classificadores), colunas(classes)
    #print("x y z")
    #print(x,y,z)
    for i in range(x):
        for j in range(y):
            for k in range(z):
                max_linha = np.amax(masked_proba[i,j])
                for sub in range(z):
                    if masked_proba[i,j,sub] == max_linha :
                        masked_proba[i,j,sub] = (z - k -1) * -1
                        break

    #print("masked_proba no final")
    #print(masked_proba)

    #print("masked_proba no final")
    #print(masked_proba * -1)
    masked_proba = masked_proba * -1

    #print("Outra")
    #print(np.sum(masked_proba,axis=1))

    masked_proba = np.sum(masked_proba,axis=1)

    #print(" masked_proba.sum(axis=1)[:, None]")
    #print( masked_proba.sum(axis=1)[:, None])

    #print(" masked_proba.sum(axis=1)")
    #print( masked_proba.sum(axis=1))


    predicted_pesos = masked_proba / masked_proba.sum(axis=1)[:, None]
    #print("predicted_pesos")
    #print(predicted_pesos)

    return predicted_pesos



# Votação majoritária
def sum_votes_per_class(predictions, n_classes):
    #print("FUNÇÃO VOTO MAJORITARIO")
    """Sum the number of votes for each class. Accepts masked arrays as input.
    Parameters
    ----------
    predictions : array of shape (n_samples, n_classifiers),
        The votes obtained by each classifier for each sample. Can be a masked
        array.
    n_classes : int
        Number of classes.
    Returns
    -------
    summed_votes : array of shape (n_samples, n_classes)
        Summation of votes for each class
    """
    #print("predictions sum votes")
    #print(predictions)
    #print("n_classes sum votes")
    #print(n_classes)
    votes = np.zeros((predictions.shape[0], n_classes), dtype=np.int64)
    #print("Votes instance sum votes (Cria amatriz zerada)")
    #print(votes)
    for label in range(n_classes):
        votes[:, label] = np.sum(predictions == label, axis=1)#Conta label nas coluna(eixo 1)
    #print("Votos majory resultado")
    #print(votes.shape)
    #print(votes)

    predicted_proba = votes / votes.sum(axis=1)[:, None]
    #print("RESULTADO predicted_proba")
    #print(predicted_proba)

    return predicted_proba




# Votação utilizando o 1 rótulo do 1 classificador
def my_vote(predictions, n_classes):
    votes = np.zeros((predictions.shape[0], n_classes), dtype=np.int64)
    print("Votes my_vote")
    print(votes)
    print("predictions my vote")
    print(predictions)
    for label in range(n_classes):
        votes[:, label] = predictions[0][0]
        print("predictions[0][0]")
        print(predictions[0][0])
        print("votes[:, label]")
        print(votes[:, label])
    print("My class Votos")
    print(votes.shape)
    print(votes)
    return votes

# OUTROS Métodos de votos



# Como utilizar
#model = MyKnoraE(voting_type='my')

#n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
#model.fit(X, y)
#print("XXX")
#print('Classification accuracy of KNORA-U: ', model.score(X, y))
from sklearn.model_selection import train_test_split#Importar divisão
#qualidade = pd.read_csv("wine.csv")
#qualidade = pd.read_csv("IRIS.csv")
#X = qualidade.drop('species',axis=1)
#y = qualidade['species']
'''
loans = pd.read_csv("loan_data.csv")
from sklearn.preprocessing import LabelEncoder
number= LabelEncoder()
loans['purpose'] = number.fit_transform(loans['purpose'])


X = loans.drop('not.fully.paid',axis=1)#dados
y = loans['not.fully.paid']#classificação
'''


#qualidadeTT, qualidadeVAL = train_test_split(qualidade, test_size=0.20, random_state=777)#Dividir os dados em treino e  validaçã
'''
X = qualidadeTT.drop('quality',axis=1)
y = qualidadeTT['quality']
'''
'''
X_val = qualidadeVAL.drop('quality',axis=1)
y_val = qualidadeVAL['quality']
'''
'''
qualidade = pd.read_csv("IRIS.csv")
X = qualidade.drop('species',axis=1)
y = qualidade['species']
'''

'''
qualidade = pd.read_csv("vinho_qualidade.csv")
X = qualidade.drop('quality',axis=1)
y = qualidade['quality']
'''

'''

        r = modelo.score(testeX, testeY)
        #print("ctz")
        resultados.append(r)
        #print("é")
    print(resultados)

'''


'''
qualidade = pd.read_csv("seismic.csv")
X = qualidade.drop('class',axis=1)
y = qualidade['class']
'''


'''
#0.25
X_train, X_test, y_train, y_test = train_test_split(X, #train.drop('Survived',axis=1) remove a coluna e a retorna mas não altera na variavel
                                                    y, test_size=0.25,#0.008
                                                    random_state=(4566))#Dividir os dados em treino e teste
#0.333333333
X_train, X_val, y_train, y_val = train_test_split(X_train, #train.drop('Survived',axis=1) remove a coluna e a retorna mas não altera na variavel
                                                  y_train, test_size=0.33333,#0.008
                                                  random_state=(45))#Dividir os dados em treino e  validaçã
#print("Valid")
#print(len(y_val))
from sklearn.calibration import CalibratedClassifierCV
# Importing dataset and preprocessing routines
from sklearn.datasets import fetch_openml
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from deslib.des import KNORAU

from deslib.static import StackedClassifier



#print(np.amax(a, axis=0))

rng = np.random.RandomState(42)
model_perceptron = CalibratedClassifierCV(Perceptron(max_iter=100,random_state=rng),cv=3)

model_perceptron.fit(X_train, y_train)
model_svc = SVC(probability=True, gamma='auto',random_state=rng).fit(X_train, y_train)
model_bayes = GaussianNB().fit(X_train, y_train)
model_tree = DecisionTreeClassifier(random_state=rng,max_depth=60).fit(X_train, y_train)
model_knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)


#pool_classifiers = [model_perceptron,model_svc,model_bayes,model_tree,model_knn]

from sklearn.ensemble import BaggingClassifier
#model = SVC(C=10, probability=True, gamma='auto',random_state=rng).fit(X_train, y_train)
model = DecisionTreeClassifier(random_state=rng,max_depth=15)
pool_classifiers = BaggingClassifier(model, n_estimators=30)
pool_classifiers.fit(X_train, y_train)

print("Treino Teste Validação")
print(str(len(y_train))+"  -  "+ str(len(y_test)) + "  -  " + str(len(y_val)) )


print("HARD")
model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=5, voting_type="hard" )
model2.fit(X_train,y_train )
print("Resultado hard: " + str(model2.score(X_test, y_test)*100))

print("SOFT")
model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=5, voting_type="soft" )
model2.fit(X_train,y_train )
print("Resultado SOFT: " + str(model2.score(X_test, y_test)*100))


print("rede neural soft")
model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=5, voting_type="rede_neural" )
model2.fit(X_train,y_train )
print("Resultado rede neural: " + str(model2.score(X_test, y_test)*100))


print("rede neural soft")
model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=5, voting_type="rede_neural_soft" )
model2.fit(X_train,y_train )
print("Resultado rede neural soft: " + str(model2.score(X_test, y_test)*100))


print("rede neural soft div")
model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=5, voting_type="rede_neural_soft_div" )
model2.fit(X_train,y_train )
print("Resultado rede neural soft div: " + str(model2.score(X_test, y_test)*100))
'''
'''
loans = pd.read_csv("loan_data.csv")
from sklearn.preprocessing import LabelEncoder
number= LabelEncoder()
loans['purpose'] = number.fit_transform(loans['purpose'])
X = loans.drop('not.fully.paid',axis=1)#dados
y = loans['not.fully.paid']#classificação
'''
foi_melhor = 0
foi_igual = 0
lugar_melhor = []
lugar_igual = []

melhor_tecnicas_basicas = 0
onde_melhor_tecnicas_basicas = 0
melhor_dynamic_metric_fusionk = 0
onde_melhor_dynamic_metric_fusionk = 0

lista_k_teste = []
lista_melhor_tecnicas_basicas = []
lista_melhor_dynamic_metric_fusionk = []

vizinhos_no_teste = 100000
#maximo_teste = 7
#while (vizinhos_no_teste <= maximo_teste):
v_total_teste = [5,10,15,20,25,30,60]
for vizinhos_no_teste in v_total_teste:
    print("")
    print("------------------- K do POOL  "+ str(vizinhos_no_teste) +" de "+str(v_total_teste)  +" --------------------------------------------------------------------")
    melhor_do_k_em_tecnicas_basicas = 0
    melhor_do_k_em_dynamic_selection = 0
    #for melhor_k in [3,5,7,11,13,17]:
    for melhor_k in [3,7,11]:
    #for melhor_k in [11]:
        porcentagem_escolheu_MAX, porcentagem_escolheu_SOFT, porcentagem_escolheu_HARD, porcentagem_escolheu_MIN, porcentagem_escolheu_G_MEAN, porcentagem_escolheu_sum_weight , porcentagem_escolheu_rede_neural, porcentagem_escolheu_rede_neural_soft , porcentagem_escolheu_rede_neural_soft_div , porcentagem_escolheu_borda , porcentagem_escolheu_naive_bayes , porcentagem_escolheu_peso_ponderado_comite = [],[],[],[], [],[],[],[],[], [],[],[]
        resultados_no_melhor_caso = []
        resultados_hard = []
        resultados_soft = []
        resultados_max = []
        resultados_min = []
        resultados_geometric_mean = []
        quantidade_exemplos_divergencia = []
        quantidade_classificadores_selecionados = []
        #resultados_peso_ponderado_classe_cada_amostra_sem_ajustes = []
        #resultados_peso_ponderado_classe_cada_amostra_ajustado_0a1 = []
        #resultados_peso_ponderado_comite_classe_distancia_maxima_teste = []
        resultados_peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1 = []
        #resultados_sum_weight_votes_per_class = []
        resultados_sum_weight_0a1_votes_per_class = []
        #resultados_sum_weight_line_votes_per_class = []
        #resultados_sum_weight_0a1_line_votes_per_class = []

        #resultados_dynamic_metric_fusionk1 = []
        resultados_dynamic_metric_fusionk3 = []

        #resultados_sum_weight  = []
        #resultados_sum_weight_line  = []
        resultados_escolheu_rede_neural  = []
        resultados_escolheu_rede_neural_soft  = []
        resultados_escolheu_rede_neural_soft_div  = []
        resultados_escolheu_borda  = []
        resultados_escolheu_naive_bayes  = []
        #resultados_escolheu_peso_ponderado_classe  = []
        #resultados_escolheu_peso_ponderado_comite  = []

        resultados_maximo_na_combinacao = []

        '''
        resultados_dynamic_metric_fusionk5 = []
        resultados_dynamic_metric_fusionk7 = []
        resultados_dynamic_metric_fusionk13 = []
        resultados_dynamic_metric_fusionk21 = []
        '''

        resultados_hard3 = []

        resultados_hard5 = []
        resultados_hard7 = []
        resultados_hard11 = []
        resultados_hard17 = []

        print("")
        print("")
        print("")




        '''
        qualidade = pd.read_csv("movement_libras.data")
        X = qualidade.iloc[:,0:-1]
        y = qualidade.iloc[:,-1]
        '''
        from sklearn.preprocessing import StandardScaler #importar preprocessamento para normalizar dados,

        print("------------------------------ INICIO KNORA K = "+ str(melhor_k)+" ------------------------------")
        k_dynamic_combination = melhor_k
        total_execucoes = 1
        for iii in range(total_execucoes):
            #print("")
            print("----------------------- Teste " + str(iii+1)+ " De "+str(total_execucoes)+" ------------------------------------------")



            '''
            loans = pd.read_csv("loan_data.csv")
            from sklearn.preprocessing import LabelEncoder
            number= LabelEncoder()
            loans['purpose'] = number.fit_transform(loans['purpose'])
            X = loans.drop('not.fully.paid',axis=1)#dados
            y = loans['not.fully.paid']#classificação
            '''

            '''            
            qualidade = pd.read_csv("IRIS2.csv")
            X = qualidade.drop('species',axis=1)
            y = qualidade['species']
            '''

            '''
            qualidade = pd.read_csv("IRIS2.csv")
            scaler = StandardScaler() # inicializar
            scaler.fit(qualidade.drop('species',axis=1))#treinar o modelo de normalização para padronizar,
            scaled_features = scaler.transform(qualidade.drop('species',axis=1))#normaliza por media edesvio padrão
            df_feat = pd.DataFrame(scaled_features,columns=qualidade.columns[:-1])
            X = df_feat
            y = qualidade['species']
            '''






            '''
            qualidade = pd.read_csv("diabetes.csv")
            X = qualidade.drop('Outcome',axis=1)
            y = qualidade['Outcome']
            '''

            '''
            qualidade = pd.read_csv("diabetes.csv")
            scaler = StandardScaler() # inicializar
            scaler.fit(qualidade.drop('Outcome',axis=1))#treinar o modelo de normalização para padronizar,
            scaled_features = scaler.transform(qualidade.drop('Outcome',axis=1))#normaliza por media edesvio padrão
            df_feat = pd.DataFrame(scaled_features,columns=qualidade.columns[:-1])
            X = df_feat
            y = qualidade['Outcome']
            '''


            '''
            qualidade = pd.read_csv("abalone.data")
            qualidade = qualidade[qualidade['Rings'] > 3]
            qualidade = qualidade[qualidade['Rings'] < 22]


            X = qualidade.drop('Rings',axis=1)
            y = qualidade['Rings']
            '''

            '''
            qualidade = pd.read_csv("vinho_qualidade.csv")
            X = qualidade.drop('quality',axis=1)
            y = qualidade['quality']
            #print(X)
            '''
            
            '''
            qualidade = pd.read_csv("vinho_qualidade.csv")
            scaler = StandardScaler() # inicializar
            scaler.fit(qualidade.drop('quality',axis=1))#treinar o modelo de normalização para padronizar,
            scaled_features = scaler.transform(qualidade.drop('quality',axis=1))#normaliza por media edesvio padrão
            df_feat = pd.DataFrame(scaled_features,columns=qualidade.columns[:-1])
            X = df_feat
            y = qualidade['quality']
            '''

            '''
            qualidade = pd.read_csv("car2.data")
            X = qualidade.drop('class',axis=1)
            y = qualidade['class']
            '''

            '''
            qualidade = pd.read_csv("car2.data")
            scaler = StandardScaler() # inicializar
            scaler.fit(qualidade.drop('class',axis=1))#treinar o modelo de normalização para padronizar,
            scaled_features = scaler.transform(qualidade.drop('class',axis=1))#normaliza por media edesvio padrão
            df_feat = pd.DataFrame(scaled_features,columns=qualidade.columns[:-1])
            X = df_feat
            y = qualidade['class']
            '''

            '''
            qualidade = pd.read_csv("movement_libras.data")
            #print(qualidade)

            scaler = StandardScaler() # inicializar
            scaler.fit(qualidade.iloc[:,0:-1])#treinar o modelo de normalização para padronizar,
            scaled_features = scaler.transform(qualidade.iloc[:,0:-1])#normaliza por media edesvio padrão
            X = pd.DataFrame(scaled_features)
            #print(X)

            #X = qualidade.iloc[:,0:-1]
            y = qualidade.iloc[:,-1]
            '''
            
            '''
            qualidade = pd.read_csv("movement_libras.data")
            X = qualidade.iloc[:,0:-1]
            y = qualidade.iloc[:,-1]
            '''
            
            '''
            dados = pd.read_csv("movement_libras.data")

            from sklearn.preprocessing import StandardScaler #importar preprocessamento para normalizar dados,
            scaler = StandardScaler() # inicializar
            scaler.fit(dados.drop('TARGET CLASS',axis=1))#treinar o modelo de normalização para padronizar,
            scaled_features = scaler.transform(dados.drop('TARGET CLASS',axis=1))#normaliza por media edesvio padrão
            df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])



            qualidade
            X = qualidade.iloc[:,0:-1]
            y = qualidade.iloc[:,-1]
            '''

            '''
            qualidade = pd.read_csv("seismic.csv")
            X = qualidade.drop('class',axis=1)
            y = qualidade['class']
            '''

            '''
            qualidade = pd.read_csv("seismic.csv")
            scaler = StandardScaler() # inicializar
            scaler.fit(qualidade.drop('class',axis=1))#treinar o modelo de normalização para padronizar,
            scaled_features = scaler.transform(qualidade.drop('class',axis=1))#normaliza por media edesvio padrão
            df_feat = pd.DataFrame(scaled_features,columns=qualidade.columns[:-1])
            X = df_feat
            y = qualidade['class']
            '''

            
            loans = pd.read_csv("loan_data.csv")
            from sklearn.preprocessing import LabelEncoder
            number= LabelEncoder()
            loans['purpose'] = number.fit_transform(loans['purpose'])


            X = loans.drop('not.fully.paid',axis=1)#dados
            y = loans['not.fully.paid']#classificação
            

            #0.25
            X_train, X_test, y_train, y_test = train_test_split(X, #train.drop('Survived',axis=1) remove a coluna e a retorna mas não altera na variavel
                                                                y, test_size=0.5,
                                                                random_state=(vizinhos_no_teste+iii+67637))#Dividir os dados em treino e teste
            #0.333333333
            X_test, X_val, y_test, y_val = train_test_split(X_test, #train.drop('Survived',axis=1) remove a coluna e a retorna mas não altera na variavel
                                                              y_test, test_size=0.33333,
                                                              random_state=(vizinhos_no_teste+iii+2343))#Dividir os dados em treino e  validaçã

            X_test, X_val2, y_test, y_val2 = train_test_split(X_test, #train.drop('Survived',axis=1) remove a coluna e a retorna mas não altera na variavel
                                                              y_test, test_size=0.50,
                                                              random_state=(vizinhos_no_teste+iii+234))#Dividir os dados em treino e  validaçã

            if(iii == 0):
                print("Treino - Teste - Validação 1 - Validação 2 - ")
                print(str(len(y_train))+ "    -   "+str(len(y_test))+"    -    "+str(len(y_val)) +"    -    "+str(len(y_val2)) )
                print("Número de Exemplos: " +str(len(X)))
                print("Número de Colunas: " +str(len(X.columns)))
                print("Número de classes: " + str(len(np.unique(y, return_counts=False))))

            from sklearn.calibration import CalibratedClassifierCV
            # Importing dataset and preprocessing routines
            from sklearn.datasets import fetch_openml
            from sklearn.ensemble import VotingClassifier

            from sklearn.linear_model import Perceptron
            from sklearn.model_selection import train_test_split
            from sklearn.naive_bayes import GaussianNB
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            from sklearn.tree import DecisionTreeClassifier

            from deslib.des import KNORAU

            from deslib.static import StackedClassifier



            #print(np.amax(a, axis=0))
            '''
            from numpy.random import default_rng

            rng = default_rng()
            numbers = rng.choice(range(1, 10), size=(3, 3), replace=False)
            print(numbers)
            print(np.amax(numbers, axis=0))
            '''
            from sklearn.ensemble import BaggingClassifier

            rng = np.random.RandomState(42)
            '''
            model_perceptron = CalibratedClassifierCV(Perceptron(max_iter=100,random_state=rng),cv=3)

            model_perceptron.fit(X_train, y_train)
            model_svc = SVC(probability=True, gamma='auto',random_state=rng).fit(X_train, y_train)
            model_bayes = GaussianNB().fit(X_train, y_train)
            model_tree = DecisionTreeClassifier(random_state=rng,max_depth=10).fit(X_train, y_train)
            model_knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)

            pool_classifiers_perceptron = BaggingClassifier(model_perceptron, n_estimators=6).fit(X_train, y_train)
            pool_classifiers_svc = BaggingClassifier(model_svc, n_estimators=6).fit(X_train, y_train)
            pool_classifiers_bayes = BaggingClassifier(model_bayes, n_estimators=6).fit(X_train, y_train)
            pool_classifiers_tree = BaggingClassifier(model_tree, n_estimators=6).fit(X_train, y_train)
            pool_classifiers_knn = BaggingClassifier(model_knn, n_estimators=6).fit(X_train, y_train)

            pool_classifiers = []

            for v in pool_classifiers_perceptron:
                pool_classifiers.append(v)

            for v in pool_classifiers_svc:
                pool_classifiers.append(v)

            for v in pool_classifiers_bayes:
                pool_classifiers.append(v)

            for v in pool_classifiers_tree:
                pool_classifiers.append(v)

            for v in pool_classifiers_knn:
                pool_classifiers.append(v)

            '''

            #print(qualidade)


            #pool_classifiers = [model_perceptron, model_svc, model_bayes, model_tree, model_knn,model_perceptron, model_svc, model_bayes, model_tree, model_knn,model_perceptron, model_svc, model_bayes, model_tree, model_knn,model_perceptron, model_svc, model_bayes, model_tree, model_knn,model_perceptron, model_svc, model_bayes, model_tree, model_knn,model_perceptron, model_svc, model_bayes, model_tree, model_knn ]
            '''
            pool_classifiers =[
                CalibratedClassifierCV(Perceptron(max_iter=100,random_state=np.random.RandomState(8765)),cv=4).fit(X_train, y_train),
                CalibratedClassifierCV(Perceptron(max_iter=50,random_state=np.random.RandomState(76)),cv=4).fit(X_train, y_train),
                CalibratedClassifierCV(Perceptron(max_iter=150,random_state=np.random.RandomState(234)),cv=3).fit(X_train, y_train),
                CalibratedClassifierCV(Perceptron(max_iter=200,random_state=np.random.RandomState(532)),cv=3).fit(X_train, y_train),
                CalibratedClassifierCV(Perceptron(max_iter=75,random_state=np.random.RandomState(342)),cv=3).fit(X_train, y_train),
                CalibratedClassifierCV(Perceptron(max_iter=250,random_state=np.random.RandomState(1042)),cv=3).fit(X_train, y_train),
                SVC(probability=True, kernel="linear", gamma='auto',random_state=np.random.RandomState(23)).fit(X_train, y_train),
                SVC(probability=True, kernel="poly", gamma='auto',random_state=np.random.RandomState(43)).fit(X_train, y_train),
                SVC(probability=True, kernel="rbf", gamma='auto',random_state=np.random.RandomState(676)).fit(X_train, y_train),
                SVC(probability=True, kernel="sigmoid", gamma='auto',random_state=np.random.RandomState(45)).fit(X_train, y_train),
                SVC(probability=True, kernel="sigmoid", gamma='scale',random_state=np.random.RandomState(457)).fit(X_train, y_train),
                SVC(probability=True, kernel="rbf", gamma='scale',random_state=np.random.RandomState(468)).fit(X_train, y_train),
                GaussianNB(var_smoothing=0.000000001).fit(X_train, y_train),
                GaussianNB(var_smoothing=0.00000001).fit(X_train, y_train),
                GaussianNB(var_smoothing=0.0000001).fit(X_train, y_train),
                GaussianNB(var_smoothing=0.000001).fit(X_train, y_train),
                GaussianNB(var_smoothing=0.00000000001).fit(X_train, y_train),
                GaussianNB(var_smoothing=0.0000000001).fit(X_train, y_train),
                DecisionTreeClassifier(random_state=np.random.RandomState(6385),max_depth=10).fit(X_train, y_train),
                DecisionTreeClassifier(random_state=np.random.RandomState(2435),max_depth=15).fit(X_train, y_train),
                DecisionTreeClassifier(random_state=np.random.RandomState(3463),max_depth=20).fit(X_train, y_train),
                DecisionTreeClassifier(random_state=np.random.RandomState(3464),max_depth=8).fit(X_train, y_train),
                DecisionTreeClassifier(random_state=np.random.RandomState(9467),max_depth=13).fit(X_train, y_train),
                DecisionTreeClassifier(random_state=np.random.RandomState(7246),max_depth=18).fit(X_train, y_train),
                KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train),
                KNeighborsClassifier(n_neighbors=5 ,weights='uniform').fit(X_train, y_train),
                KNeighborsClassifier(n_neighbors=5 ,weights='distance').fit(X_train, y_train),
                KNeighborsClassifier(n_neighbors=7,weights='uniform').fit(X_train, y_train),
                KNeighborsClassifier(n_neighbors=7,weights='distance').fit(X_train, y_train),
                KNeighborsClassifier(n_neighbors=9).fit(X_train, y_train)
                ]
            '''

            #model = SVC(C=10, probability=True, gamma='auto',random_state=rng).fit(X_train, y_train)
            #'''
            model = DecisionTreeClassifier(random_state=rng,max_depth=15)
            pool_classifiers = BaggingClassifier(model, n_estimators=vizinhos_no_teste)
            pool_classifiers.fit(X_train, y_train)
            #'''
            #print("Classification KNORA-U resultados_dynamic_metric_fusionk3")
            #model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="dynamic_metric_fusionk5" )
            #model2.fit(X_train,y_train )
            #resultados_dynamic_metric_fusionk3.append(model2.score(X_test, y_test))

            '''
            # EM busca do melhor K -----------------------------------------------
            #print("Classification KNORA-U HARD")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=3, voting_type="hard" )
            model2.fit(X_train,y_train )
            resultados_hard3.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U HARD")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=5, voting_type="hard" )
            model2.fit(X_train,y_train )
            resultados_hard5.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U HARD")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=7, voting_type="hard" )
            model2.fit(X_train,y_train )
            resultados_hard7.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U HARD")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=11, voting_type="hard" )
            model2.fit(X_train,y_train )
            resultados_hard11.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U HARD")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=17, voting_type="hard" )
            model2.fit(X_train,y_train )
            resultados_hard17.append(model2.score(X_test, y_test))
            #------------------------------------------------
            '''


            '''
            #print("Classification KNORA-U HARD")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="hard" )
            model2.fit(X_train,y_train )
            resultados_hard.append(model2.score(X_test, y_test))



            #print("Classification KNORA-U SOFT")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="soft" )
            model2.fit(X_train,y_train )
            resultados_soft.append(model2.score(X_test, y_test))





            #Adicionados

            #print("Classification KNORA-U MAX")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="max" )
            model2.fit(X_train,y_train )
            resultados_max.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U MIN")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="min" )
            model2.fit(X_train,y_train )
            resultados_min.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U GEOMETRIC MEAN")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="geometric_mean" )
            model2.fit(X_train,y_train )
            resultados_geometric_mean.append(model2.score(X_test, y_test))

            #Tecnicas Personalizadas

            #Peso ponderado
            #print("Classification KNORA-U resultados_peso_ponderado_classe_cada_amostra_sem_ajustes")
            #model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="peso_ponderado_classe_cada_amostra_sem_ajustes" )
            #model2.fit(X_train,y_train )
            #resultados_peso_ponderado_classe_cada_amostra_sem_ajustes.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U resultados_peso_ponderado_classe_cada_amostra_ajustado_0a1")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="peso_ponderado_classe_cada_amostra_ajustado_0a1" )
            model2.fit(X_train,y_train )
            resultados_peso_ponderado_classe_cada_amostra_ajustado_0a1.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U resultados_peso_ponderado_comite_classe_distancia_maxima_teste")
            #model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="peso_ponderado_comite_classe_distancia_maxima_teste" )
            #model2.fit(X_train,y_train )
            #resultados_peso_ponderado_comite_classe_distancia_maxima_teste.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U resultados_peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1" )
            model2.fit(X_train,y_train )
            resultados_peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1.append(model2.score(X_test, y_test))

            #Voto Ponderado
            #print("Classification KNORA-U resultados_sum_weight_votes_per_class")
            #model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="sum_weight_votes_per_class" )
            #model2.fit(X_train,y_train )
            #resultados_sum_weight_votes_per_class.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U resultados_sum_weight_0a1_votes_per_class")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="sum_weight_0a1_votes_per_class" )
            model2.fit(X_train,y_train )
            resultados_sum_weight_0a1_votes_per_class.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U resultados_sum_weight_line_votes_per_class")
            #model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="sum_weight_line_votes_per_class" )
            #model2.fit(X_train,y_train )
            #resultados_sum_weight_line_votes_per_class.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U resultados_sum_weight_0a1_line_votes_per_class")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="sum_weight_0a1_line_votes_per_class" )
            model2.fit(X_train,y_train )
            resultados_sum_weight_0a1_line_votes_per_class.append(model2.score(X_test, y_test))

            #Fusão/Combinação dinamica


            #print('TESTE rede_neural: ')
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="rede_neural" )
            model2.fit(X_train,y_train )
            resultados_escolheu_rede_neural.append(model2.score(X_test, y_test))


            #print('TESTE Rede_neural_soft : ')
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="rede_neural_soft" )
            model2.fit(X_train,y_train )
            resultados_escolheu_rede_neural_soft.append(model2.score(X_test, y_test))


            #print('TESTE Rede_neural_soft_div: ')
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="rede_neural_soft_div" )
            model2.fit(X_train,y_train )
            resultados_escolheu_rede_neural_soft_div.append(model2.score(X_test, y_test))



            #print('TESTE borda: ')
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="borda" )
            model2.fit(X_train,y_train )
            resultados_escolheu_borda.append(model2.score(X_test, y_test))


            #print('TESTE naive_bayes: ')
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=k_dynamic_combination, voting_type="naive_bayes" )
            model2.fit(X_train,y_train )
            resultados_escolheu_naive_bayes.append(model2.score(X_test, y_test))
            '''




            #print("Classification KNORA-U resultados_dynamic_metric_fusionk1")
            #model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=5, voting_type="dynamic_metric_fusionk1" )
            #model2.fit(X_train,y_train )
            #resultados_dynamic_metric_fusionk1.append(model2.score(X_test, y_test))

            '''
            print("DO COMITE 1 2 3 4 5")
            print(pool_classifiers[0].predict(X_test))
            print(pool_classifiers[1].predict(X_test))
            print(pool_classifiers[2].predict(X_test))
            print(pool_classifiers[3].predict(X_test))
            print(pool_classifiers[4].predict(X_test))
            '''
            
            divergencia_classificadores = False
            #print("Classification KNORA-U resultados_dynamic_metric_fusionk3")
            #print(divergencia_classificadores)
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="dynamic_metric_fusionk3" )
            model2.fit(X_train,y_train )
            #print("####################################### PROBABILIDADE ##############################################")
            #model2.predict_proba(X_test)
            #print("####################################### SCORE ##############################################")
            o_resultado = model2.score(X_test, y_test)
            resultados_dynamic_metric_fusionk3.append(o_resultado)
            #print(divergencia_classificadores)

            if divergencia_classificadores == False:
                print("#####        Resultados iguais dos classificadores         #####")
                resultados_maximo_na_combinacao.append(o_resultado)
                resultados_max.append(o_resultado)
                resultados_soft.append(o_resultado)
                resultados_hard.append(o_resultado)
                resultados_min.append(o_resultado)
                resultados_geometric_mean.append(o_resultado)
                resultados_escolheu_rede_neural.append(o_resultado)
                resultados_escolheu_rede_neural_soft.append(o_resultado)
                resultados_escolheu_rede_neural_soft_div.append(o_resultado)
                resultados_escolheu_borda.append(o_resultado)
                resultados_escolheu_naive_bayes.append(o_resultado)
                resultados_peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1.append(o_resultado)
                resultados_sum_weight_0a1_votes_per_class.append(o_resultado)

                t_class = 12
                porcentagem_escolheu_MAX.append(100/t_class)
                porcentagem_escolheu_SOFT.append(100/t_class)
                porcentagem_escolheu_HARD.append(100/t_class)
                porcentagem_escolheu_MIN.append(100/t_class)
                porcentagem_escolheu_G_MEAN.append(100/t_class)

                porcentagem_escolheu_sum_weight.append(100/t_class)
                porcentagem_escolheu_peso_ponderado_comite.append(100/t_class)

                porcentagem_escolheu_rede_neural.append(100/t_class)
                porcentagem_escolheu_rede_neural_soft.append(100/t_class)
                porcentagem_escolheu_rede_neural_soft_div.append(100/t_class)
                porcentagem_escolheu_borda.append(100/t_class)
                porcentagem_escolheu_naive_bayes.append(100/t_class)

                quantidade_exemplos_divergencia.append(0)
                #quantidade_classificadores_selecionados.append(0)



            '''
            #print("Classification KNORA-U resultados_dynamic_metric_fusionk5")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="dynamic_metric_fusionk5" )
            model2.fit(X_train,y_train )
            resultados_dynamic_metric_fusionk5.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U resultados_dynamic_metric_fusionk7")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="dynamic_metric_fusionk7" )
            model2.fit(X_train,y_train )
            resultados_dynamic_metric_fusionk7.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U resultados_dynamic_metric_fusionk13")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="dynamic_metric_fusionk13" )
            model2.fit(X_train,y_train )
            resultados_dynamic_metric_fusionk13.append(model2.score(X_test, y_test))

            #print("Classification KNORA-U resultados_dynamic_metric_fusionk k-21")
            model2 = MyKnoraE(pool_classifiers=pool_classifiers, k=melhor_k, voting_type="dynamic_metric_fusionk21" )
            model2.fit(X_train,y_train )
            resultados_dynamic_metric_fusionk21.append(model2.score(X_test, y_test))
            '''


            #print("Carregando "+str() + " de "+ str())

        print("-#-#-# Resultado #-#-#-")

        '''
        print("MELHOR K")
        print("")
        print("HARD K 3")
        #print(resultados_hard)
        print("Média: " + str(np.mean(resultados_hard3)*100))
        print("Desvio padrão: " + str(np.std(resultados_hard3)*100))
        print("")
        print("HARD K 5")
        #print(resultados_hard)
        print("Média: " + str(np.mean(resultados_hard5)*100))
        print("Desvio padrão: " + str(np.std(resultados_hard5)*100))
        print("")
        print("HARD K 7")
        #print(resultados_hard)
        print("Média: " + str(np.mean(resultados_hard7)*100))
        print("Desvio padrão: " + str(np.std(resultados_hard7)*100))
        print("")
        print("HARD K 11")
        #print(resultados_hard)
        print("Média: " + str(np.mean(resultados_hard11)*100))
        print("Desvio padrão: " + str(np.std(resultados_hard11)*100))
        print("")
        print("HARD K 17")
        #print(resultados_hard)
        print("Média: " + str(np.mean(resultados_hard17)*100))
        print("Desvio padrão: " + str(np.std(resultados_hard17)*100))
        print("")
        '''
        maximo_lista = max([np.mean(resultados_hard), np.mean(resultados_soft), np.mean(resultados_max), np.mean(resultados_min), np.mean(resultados_geometric_mean), np.mean(resultados_peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1), np.mean(resultados_sum_weight_0a1_votes_per_class), np.mean(resultados_escolheu_rede_neural), np.mean(resultados_escolheu_rede_neural_soft), np.mean(resultados_escolheu_rede_neural_soft_div), np.mean(resultados_escolheu_borda), np.mean(resultados_escolheu_naive_bayes)])




        if maximo_lista > melhor_do_k_em_tecnicas_basicas:
            melhor_do_k_em_tecnicas_basicas = maximo_lista
        if np.mean(resultados_dynamic_metric_fusionk3) > melhor_do_k_em_dynamic_selection:
            melhor_do_k_em_dynamic_selection = np.mean(resultados_dynamic_metric_fusionk3)
            
        
        
        if np.mean(resultados_dynamic_metric_fusionk3) > maximo_lista:
            print("####################################### MELHOROU ###########################################")
            print(maximo_lista)
            print("##################################################")
            foi_melhor = foi_melhor + 1
            lugar_melhor.append(vizinhos_no_teste)

        if np.mean(resultados_dynamic_metric_fusionk3) == maximo_lista:
            foi_igual = foi_igual + 1
            lugar_igual.append(vizinhos_no_teste)
            
        if maximo_lista >= melhor_tecnicas_basicas:
            melhor_tecnicas_basicas = maximo_lista
            onde_melhor_tecnicas_basicas = vizinhos_no_teste
        if np.mean(resultados_dynamic_metric_fusionk3) > melhor_dynamic_metric_fusionk :
            melhor_dynamic_metric_fusionk = np.mean(resultados_dynamic_metric_fusionk3)
            onde_melhor_dynamic_metric_fusionk = vizinhos_no_teste

        print("Resultados maximo na combinacao")
        #print(resultados_hard)
        print("Média: " + str(np.mean(resultados_maximo_na_combinacao)*100))
        print("Desvio padrão: " + str(np.std(resultados_maximo_na_combinacao)*100))
        print("")

        print("Qnt. Exemplos divergentes")
        print("Média: "+ str(np.mean(quantidade_exemplos_divergencia)) )
        print("Desvio padrão: " + str(np.std(quantidade_exemplos_divergencia)) )
        print("")

        print("Qnt. Classificadores selecionados ")
        print("Média: "+ str(np.mean(quantidade_classificadores_selecionados)) )
        print("Desvio padrão: " + str(np.std(quantidade_classificadores_selecionados)) )
        print("")
        

        print("HARD")
        #print(resultados_hard)
        print("Média: " + str(np.mean(resultados_hard)*100))
        print("Desvio padrão: " + str(np.std(resultados_hard)*100))
        print("")

        print("SOFT")
        #print(resultados_soft)
        print("Média: " + str(np.mean(resultados_soft)*100))
        print("Desvio padrão: " + str(np.std(resultados_soft)*100))
        print("")

        print("MAX")
        #print(resultados_max)
        print("Média: " + str(np.mean(resultados_max)*100))
        print("Desvio padrão: " + str(np.std(resultados_max)*100))
        print("")

        print("MIN")
        #print(resultados_min)
        print("Média: " + str(np.mean(resultados_min)*100))
        print("Desvio padrão: " + str(np.std(resultados_min)*100))
        print("")

        print("Geometric mean")
        #print(resultados_geometric_mean)
        print("Média: " + str(np.mean(resultados_geometric_mean)*100))
        print("Desvio padrão: " + str(np.std(resultados_geometric_mean)*100))
        print("")

        #print("Peso_ponderado_classe_cada_amostra_sem_ajustes")
        #print(resultados_peso_ponderado_classe_cada_amostra_sem_ajustes)
        #print("Média: " + str(np.mean(resultados_peso_ponderado_classe_cada_amostra_sem_ajustes)*100))
        #print("Desvio padrão: " + str(np.std(resultados_peso_ponderado_classe_cada_amostra_sem_ajustes)*100))
        #print("")

        #print("Peso_ponderado_classe_cada_amostra_ajustado_0a1")
        #print(resultados_peso_ponderado_classe_cada_amostra_ajustado_0a1)
        #print("Média: " + str(np.mean(resultados_peso_ponderado_classe_cada_amostra_ajustado_0a1)*100))
        #print("Desvio padrão: " + str(np.std(resultados_peso_ponderado_classe_cada_amostra_ajustado_0a1)*100))
        #print("")

        #print("Peso_ponderado_comite_classe_distancia_maxima_teste")
        #print(resultados_peso_ponderado_comite_classe_distancia_maxima_teste)
        #print("Média: " + str(np.mean(resultados_peso_ponderado_comite_classe_distancia_maxima_teste)*100))
        #print("Desvio padrão: " + str(np.std(resultados_peso_ponderado_comite_classe_distancia_maxima_teste)*100))
        #print("")

        print("Peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1")
        #print(resultados_peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1)
        print("Média: " + str(np.mean(resultados_peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1)*100))
        print("Desvio padrão: " + str(np.std(resultados_peso_ponderado_comite_classe_distancia_maxima_teste_ajustado_0a1)*100))
        print("")



        #print("Sum_weight_votes_per_class")
        #print(resultados_sum_weight_votes_per_class)
        #print("Média: " + str(np.mean(resultados_sum_weight_votes_per_class)*100))
        #print("Desvio padrão: " + str(np.std(resultados_sum_weight_votes_per_class)*100))
        #print("")

        print("Sum_weight_0a1_votes_per_class")
        #print(resultados_sum_weight_0a1_votes_per_class)
        print("Média: " + str(np.mean(resultados_sum_weight_0a1_votes_per_class)*100))
        print("Desvio padrão: " + str(np.std(resultados_sum_weight_0a1_votes_per_class)*100))
        print("")

        #print("Sum_weight_line_votes_per_class")
        #print(resultados_sum_weight_line_votes_per_class)
        #print("Média: " + str(np.mean(resultados_sum_weight_line_votes_per_class)*100))
        #print("Desvio padrão: " + str(np.std(resultados_sum_weight_line_votes_per_class)*100))
        #print("")

        #print("Sum_weight_0a1_line_votes_per_class")
        #print(resultados_sum_weight_0a1_line_votes_per_class)
        #print("Média: " + str(np.mean(resultados_sum_weight_0a1_line_votes_per_class)*100))
        #print("Desvio padrão: " + str(np.std(resultados_sum_weight_0a1_line_votes_per_class)*100))
        #print("")



        #############
        print("Rede_neural")
        #print(resultados_escolheu_rede_neural)
        print("Média: " + str(np.mean(resultados_escolheu_rede_neural)*100))
        print("Desvio padrão: " + str(np.std(resultados_escolheu_rede_neural)*100))
        print("")


        print("Rede_neural_soft")
        #print(resultados_escolheu_rede_neural_soft)
        print("Média: " + str(np.mean(resultados_escolheu_rede_neural_soft)*100))
        print("Desvio padrão: " + str(np.std(resultados_escolheu_rede_neural_soft)*100))
        print("")


        print("Rede_neural_soft_div")
        #print(resultados_escolheu_rede_neural_soft_div)
        print("Média: " + str(np.mean(resultados_escolheu_rede_neural_soft_div)*100))
        print("Desvio padrão: " + str(np.std(resultados_escolheu_rede_neural_soft_div)*100))
        print("")


        print("Borda")
        #print(resultados_escolheu_borda)
        print("Média: " + str(np.mean(resultados_escolheu_borda)*100))
        print("Desvio padrão: " + str(np.std(resultados_escolheu_borda)*100))
        print("")

        print("Naive bayes")
        #print(resultados_escolheu_naive_bayes)
        print("Média: " + str(np.mean(resultados_escolheu_naive_bayes)*100))
        print("Desvio padrão: " + str(np.std(resultados_escolheu_naive_bayes)*100))
        print("")



        #print("resultados_dynamic_metric_fusion k-1")
        #print(resultados_dynamic_metric_fusionk1)
        #print("Média: " + str(np.mean(resultados_dynamic_metric_fusionk1)*100))
        #print("Desvio padrão: " + str(np.std(resultados_dynamic_metric_fusionk1)*100))
        #print("")


        print("resultados_dynamic_metric_fusion ")
        #print(resultados_dynamic_metric_fusionk3)
        print("Média: " + str(np.mean(resultados_dynamic_metric_fusionk3)*100))
        print("Desvio padrão: " + str(np.std(resultados_dynamic_metric_fusionk3)*100))
        print("")


        
            

        





        '''
        print("resultados_dynamic_metric_fusionk5")
        #print(resultados_dynamic_metric_fusionk5)
        print("Média: " + str(np.mean(resultados_dynamic_metric_fusionk5)*100))
        print("Desvio padrão: " + str(np.std(resultados_dynamic_metric_fusionk5)*100))
        print("")

        print("resultados_dynamic_metric_fusion k-7")
        #print(resultados_dynamic_metric_fusionk7)
        print("Média: " + str(np.mean(resultados_dynamic_metric_fusionk7)*100))
        print("Desvio padrão: " + str(np.std(resultados_dynamic_metric_fusionk7)*100))
        print("")

        print("resultados_dynamic_metric_fusionk13")
        #print(resultados_dynamic_metric_fusionk13)
        print("Média: " + str(np.mean(resultados_dynamic_metric_fusionk13)*100))
        print("Desvio padrão: " + str(np.std(resultados_dynamic_metric_fusionk13)*100))
        print("")


        print("resultados_dynamic_metric_fusion k-21")
        #print(resultados_dynamic_metric_fusionk21)
        print("Média: " + str(np.mean(resultados_dynamic_metric_fusionk21)*100))
        print("Desvio padrão: " + str(np.std(resultados_dynamic_metric_fusionk21)*100))
        print("")
        '''



        print("Média de porcentagem que escolheu SOFT: " + str(np.mean(porcentagem_escolheu_SOFT)))
        print("Média de porcentagem que escolheu HARD: " + str(np.mean(porcentagem_escolheu_HARD)))
        print("Média de porcentagem que escolheu G_MEAN: " + str(np.mean(porcentagem_escolheu_G_MEAN)))
        '''
        print("Média de porcentagem que escolheu MAX: " + str(np.mean(porcentagem_escolheu_MAX)))
        print("Média de porcentagem que escolheu MIN: " + str(np.mean(porcentagem_escolheu_MIN)))
        print("Média de porcentagem que escolheu sum_weight: " + str(np.mean(porcentagem_escolheu_sum_weight)))
        print("Média de porcentagem que escolheu peso_ponderado_comite: " + str(np.mean(porcentagem_escolheu_peso_ponderado_comite)))
        '''
        

        print("Média de porcentagem que escolheu rede_neural: " + str(np.mean(porcentagem_escolheu_rede_neural)))
        print("Média de porcentagem que escolheu rede_neural_soft: " + str(np.mean(porcentagem_escolheu_rede_neural_soft)))
        print("Média de porcentagem que escolheu rede_neural_soft_div: " + str(np.mean(porcentagem_escolheu_rede_neural_soft_div)))
        print("Média de porcentagem que escolheu borda: " + str(np.mean(porcentagem_escolheu_borda)))
        print("Média de porcentagem que escolheu naive_bayes: " + str(np.mean(porcentagem_escolheu_naive_bayes)))













        print("^^^^--------------------------- FIM  K = "+ str(melhor_k)+" ---------------------------^^")

    lista_k_teste.append(vizinhos_no_teste)
    lista_melhor_tecnicas_basicas.append(melhor_do_k_em_tecnicas_basicas)
    lista_melhor_dynamic_metric_fusionk.append(np.mean(melhor_do_k_em_dynamic_selection))
    #vizinhos_no_teste = vizinhos_no_teste + 2

print("")
print("")
print("################################## RESULTADO GERAL #############################################")
print("")
print("")
print("Dynamic metric fusionk foi melhor "+str(foi_melhor)+" vezes, das "+ str(3 * len(lista_k_teste)) +" vezes")
print("O lugar melhor de Dynamic metric fusion k "+ str(lugar_melhor))
print("")
print("")
print("Dynamic metric fusionk foi igual as basicas "+str(foi_igual)+" vezes, das "+ str(3 * len(lista_k_teste)) +" vezes")
print("O lugar onde foi igual o Dynamic metric fusion k "+ str(lugar_igual))
print("")
print("")
print("Melhor das técnicas basicas nos testes geral foi: "+ str(melhor_tecnicas_basicas))
print("Onde foi melhor tecnicas_basicas: "+str(onde_melhor_tecnicas_basicas))
print("")
print("Melhor dynamic_metric fusionk nos testes geral foi: "+str(melhor_dynamic_metric_fusionk))
print("Onde foi melhor o Dynamic metric fusion k :  "+str(onde_melhor_dynamic_metric_fusionk))


#Figura que mostra o DB e CR usando k vizinhos de 2 a 20
fig, ax = plt.subplots(figsize=(10,8))
# plt.figure(figsize=(10,8))
plt.plot(lista_k_teste,lista_melhor_tecnicas_basicas,color='red', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10,label='Melhores técnicas básicas')
plt.plot(lista_k_teste,lista_melhor_dynamic_metric_fusionk,color='blue', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10,label='Melhores dynamic metric fusion k')
ax.set_xticks(lista_k_teste)
ax.set_xticklabels(lista_k_teste)
plt.title('Score, Melhores técnicas básicas vs. Melhores dynamic metric fusion k')
plt.xlabel('K')
plt.ylabel('Score')
plt.grid(True)
plt.legend()
plt.show()




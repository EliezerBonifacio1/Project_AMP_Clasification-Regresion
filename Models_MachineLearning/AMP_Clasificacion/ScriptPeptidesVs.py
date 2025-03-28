# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:08:44 2023

@author: eliez
"""


from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import cross_val_score
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.metrics import *
import sklearn
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
import pygad
import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns


# Carga de data:
print("LOADING DATA")
data_pos = pd.read_pickle(
    "C:/Users/eliez/OneDrive/Tesis y titulacion/Antimicrobial_Peptides_QSAR_Prediction_Project/Data/DataSet_AntimicrobialPeptides_Descriptors_Final.pkl")
data_neg = pd.read_pickle(
    "C:/Users/eliez/OneDrive/Tesis y titulacion/Antimicrobial_Peptides_QSAR_Prediction_Project/Data/DataSet_NegativesPeptides_Descriptors_Final.pkl")

# Seleccion de peptidos AMP
# convertimos a numero la columna Mesure_log
data_pos.Mesure_log = pd.to_numeric(data_pos.Mesure_log, errors='coerce')

# Eliminamos las columnas que no tienen medida de logMIC (Mesure_log)
data_pos = data_pos[pd.notna(data_pos.Mesure_log)]

# ==============================================================================
#                     SELCCION DE DATOS
# ==============================================================================

#Filtramos segun corresponda al dataset 
MIC = 'MIC'
Especificidad = ['Igual_A', 'Menor_A', 'Menor_Igual', 'Rango']
Unidad = 'µg/ml'

Target_type = ['Bacteria Gram Positive']

""" 
Target_type= ['Bacteria Gram Positive',
              'Bacteria Gram Negative',
              'Bacteria No Gram',
              'Virus'
              'Hongo',
              'Protozoo']
"""
    
Genero = ["Bacillus"]

filtro1 = data_pos.Cuantification_Type == MIC
filtro2 = data_pos.Quantitation_Specificity.isin(Especificidad)
filtro3 = data_pos.Unity == Unidad
filtro4 = data_pos.Target_Type.isin(Target_type)
filtro5 = data_pos.Genero.isin(Genero)

data_filter = data_pos.copy()
data_filter = data_pos[filtro1 & filtro2 & filtro3 ]

pref = "VsBacillus/PeptidesVsBacillus_"  # PrefijoResultados:

#Descripcion de la data
data_desc = data_filter.describe()

#Definicion de AMP y NO-AMP

umbral_amp = 25      # ACTIVO
umbral_no_amp = 100   # NO ACTIVO
rs = 321  # Random State (Reproducibilidad de los algoritmos)

## Clasificamos los peptidos segun su actividad en log de MIC

def asignar_clase(numero):
    if numero > mt.log10(umbral_no_amp):
        return 0    # 0 =  NO AMP
    elif numero < mt.log10(umbral_amp):
        return 1    # 1 =  NO AMP
    else:
        return -1  # -1 =  AMP NO CLASIFICABLE


# Aplicar la función a una nueva columna 'AMP'
data_filter['AMP'] = data_filter['Mesure_log'].apply(asignar_clase)

# Quitamos los que amp no clasificados "-1"
data_filter_2 = data_filter[data_filter.AMP != -1]
print(data_filter_2['Object_Organism'].value_counts())

# Calcular el promedio de los valores numéricos agrupados por la columna 'Secuencias'
data_agrupada1 = data_filter_2.groupby('Sequence').mean()
data_agrupada1['AMP'] = data_filter_2.groupby('Sequence')['AMP'].mean()

umbral = 0.5
condiciones = [
    data_agrupada1['AMP'] >= umbral,
    data_agrupada1['AMP'] < umbral
]
valores = [1, 0]

data_agrupada1['AMP'] = np.select(condiciones, valores, default=-1)

# Seleccionamos descriptores del data set de peptidos
data_amp_final = data_agrupada1.iloc[0:, 1:]
n_AMP, no_AMP = (data_amp_final["AMP"] == 1).sum(
), (data_amp_final["AMP"] == 0).sum()

print('#AMPs: ', n_AMP, '#NO-AMPs: ', no_AMP)

# Seleccionamos descriptores del dataset negativo
df_neg_amp_final = data_neg.select_dtypes(include=['float64', 'int']).copy()
df_neg_amp_final['AMP'] = 0


# DATASET FINAL PARA EL ENTRENAMIENTO DEL MODELO, Se equilibran los datos 
negative_samples = n_AMP-no_AMP
df_final = pd.concat([data_amp_final, df_neg_amp_final.sample(
    n=negative_samples, random_state=rs)], axis=0)
n_AMP, no_AMP = (df_final["AMP"] == 1).sum(), (df_final["AMP"] == 0).sum()
print('#AMPs: ', n_AMP, '#NO-AMPs: ', no_AMP)

# ==============================================================================
#                      DIVISION DE DE DATOS
# ==============================================================================

#Se definen variables dependienentes / independiente
x_var = df_final.iloc[0:, 0:-1]
y_var = df_final.iloc[0:, -1]

#Split de los datos en entrenamiento y prueba (test)
x_train, x_test, y_train, y_test = train_test_split(x_var,
                                                    y_var,
                                                    test_size=0.2,
                                                    random_state=rs,
                                                    shuffle=True)

AMPData_Train = pd.concat([x_train, y_train], axis=1)
AMPData_Test = pd.concat([x_test, y_test], axis=1)

AMPData_Train.to_pickle(pref+"AMP_Data_Train.pkl")
AMPData_Test.to_pickle(pref+"AMP_Data_Test.pkl")

# ==============================================================================
#                      PREPROCESAMIENTO DE DATOS
# ==============================================================================

# ------------REESCALADO----------------------

preprocessor = ColumnTransformer(
    [('scale', StandardScaler(), x_train.columns)],
    remainder='passthrough')
preprocessor.fit(x_train)

# Save pipeline de normalizacion y escalado de variables
dump(preprocessor, pref + 'PipelineNormScalAllDescriptors.joblib')


xtrain_ssnn = pd.DataFrame(preprocessor.transform(
    x_train), columns=preprocessor.get_feature_names_out())
xtest_ssnn = pd.DataFrame(preprocessor.transform(
    x_test), columns=preprocessor.get_feature_names_out())

print(xtrain_ssnn.shape)
print(xtest_ssnn.shape)
print(y_train.shape)
print(y_test.shape)

n_AMP, no_AMP = (y_train == 1).sum(), (y_train == 0).sum()
print('TRAINING DATA: \n', '#AMPs: ', n_AMP, '#NO-AMPs: ', no_AMP)

n_AMP, no_AMP = (y_test == 1).sum(), (y_test == 0).sum()
print('TEST DATA: \n', '#AMPs: ', n_AMP, '#NO-AMPs: ', no_AMP)


# Eliminacion de variable de baja varianza 

pipe_VarSel = Pipeline([('LowVariance', VarianceThreshold(0.001)),
                        ('SelectKbest', SelectKBest(f_classif, k=250))
                        ], verbose=True)
pipe_VarSel.fit(xtrain_ssnn, y_train)

xtrain_vs = pd.DataFrame(pipe_VarSel.transform(
    xtrain_ssnn), columns=pipe_VarSel.get_feature_names_out())
xtest_vs = pd.DataFrame(pipe_VarSel.transform(
    xtest_ssnn), columns=pipe_VarSel.get_feature_names_out())


# SELECCION DE CARACTERISTICAS por RFE

print("RUNNING RFEVC")
t1 = time.time()
SelectorRFE = RFECV(
    estimator=RandomForestClassifier(),
    step=5,
    cv=StratifiedKFold(5),
    scoring="matthews_corrcoef",
    min_features_to_select=1,
    n_jobs=-1,
    verbose=True
)
SelectorRFE.fit(xtrain_vs, y_train)
t2 = time.time()
print(t2-t1)

rfecv_resultados = pd.DataFrame(SelectorRFE.cv_results_)
print("SCORE BEST CV: ", max(rfecv_resultados.mean_test_score))
plt.plot(rfecv_resultados.mean_test_score, 'o')
plt.savefig(pref+"squares.png")

# Save RFE
dump(SelectorRFE, pref + 'RFECV.joblib')

SelectorRFE.get_feature_names_out()
SelectorRFE.ranking_

len(SelectorRFE.get_feature_names_out())

# Seleccionando descriptores basados en RFE
x_train_rfe = pd.DataFrame(SelectorRFE.transform(
    xtrain_vs), columns=SelectorRFE.get_feature_names_out())
x_test_rfe = pd.DataFrame(SelectorRFE.transform(
    xtest_vs), columns=SelectorRFE.get_feature_names_out())

descript = []
for a, b in enumerate(SelectorRFE.feature_names_in_):
    if (SelectorRFE.ranking_ <= 3)[a] == True:
        descript.append(b)
len(descript)

#x_train_rfe= pd.DataFrame(SelectorRFE.transform(xtrain_vs), columns=SelectorRFE.get_feature_names_out())
#x_test_rfe= pd.DataFrame(SelectorRFE.transform(xtest_vs), columns=SelectorRFE.get_feature_names_out())

x_train_rfe = xtrain_vs[descript]
x_test_rfe = xtest_vs[descript]

x_train_rfe.head(5)

# SELECCION DE CARACTERISTICAS por ALGORITMO GENETICO

c = 0
scores_ga = []


def fitness_func_binarC(AG, solution, solution_idx):
    global c
    c += 1
    # ENTRENAMIENTO
    t1 = time.time()

    x_train_selected = x_train_rfe.iloc[:, [
        i for i, valor in enumerate(solution) if valor == 1]]

    model = RandomForestClassifier(max_depth=10, random_state=rs, n_jobs=-1)
    cv_scores = cross_val_score(model,
                                x_train_selected,
                                y_train,
                                scoring='matthews_corrcoef',
                                cv=StratifiedKFold(5))
    mc = np.mean(cv_scores)
    scores_ga.append(cv_scores)
    t2 = time.time()
    print(t2-t1, mc, c, solution_idx, sep=' / ')
    return mc


fitness_function = fitness_func_binarC

num_generations = 200  # NUMERO DE GENERACIONES A EVOLUCIONAR
num_parents_mating = 2  # NUMERO DE PADRES QUE HACEN CRUZA

sol_per_pop = 20        # POBLACION(CANTIDAD)
# NUMERO DE "GENES" Segun el output deseado
num_genes = len(x_train_rfe.columns)

gene_space = [0, 1]  # Valores permitidos para los genes
genes = x_train.columns

parent_selection_type = "sss"  # Tipo de seleccion de padres:
keep_parents = 1               # Se mantienen a los padres o no

crossover_type = "single_point"  # Tipo de cruza entre padres
crossover_proba = 0.8  # Probabilidad de cruza

mutation_type = "random"  # Tipo de mutacion
mutation_percent_genes = 7.5  # Probabilidad de mutacion

t1 = time.time()

ga_instanceC = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        crossover_probability=crossover_proba,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        gene_space=gene_space,
                        save_best_solutions=True,
                        stop_criteria='saturate_50',
                        allow_duplicate_genes=False,
                        save_solutions=True,
                        suppress_warnings=True,
                        random_seed=rs
                        )
print("RUNNING GENETIC ALGORITHM")
print("tiempo / Acurracy / MC / ESPECIE /  GENERACION")
ga_instanceC.run()

t2 = time.time()

print("TIEMPO DE ALGORITMO GENETICO: ", t2-t1)

pd.DataFrame(scores_ga).to_csv(
    pref+"scores_ga.csv")

solutionC, solution_fitnessC, solution_idxC = ga_instanceC.best_solution()
print("Parameters of the best solution : {solution}".format(
    solution=solutionC))
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitnessC))
print('Total variables: ', sum(solutionC))

best_solution_gen = ga_instanceC.best_solutions_fitness

ga_instanceC.save(filename=pref+"GeneticAlgortm")

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

fig, ax = plt.subplots(1)
plt.rcParams['figure.figsize'] = [8, 5]
fig.suptitle('Generación vs Precisión', size=25)
ax.plot(best_solution_gen, 'o-', linewidth=2.5)
plt.xlabel('Precisión', size=18)
plt.ylabel('Número de generación', size=18)
plt.savefig(pref+'PlotFitnessGA.svg')

plt.savefig(pref+'PlotFitnessGA.svg')

best_variables = pd.DataFrame(x_train_rfe.columns)
best_variables['VarSelected'] = solutionC
best_variables.to_excel(pref+"BestTrainingDescriptorsGA.xlsx")



#DETERMINACION DEL DOMINIO DE APLICABILIDAD:
    
from sklearn.neighbors import LocalOutlierFactor

pipeDomain = Pipeline([('AD_LOF',  LocalOutlierFactor(n_neighbors=20,contamination= 'auto', novelty=True))
                       ], verbose=True)
pipeDomain.fit(x_train_GA)

#ALLDATA: DOMAIN

trainAD = pipeDomain.decision_function(x_train_GA)
testAD = pipeDomain.decision_function(x_test_GA)

x_train_AD = x_train_GA.copy()
x_train_AD['SetClass'] = "Train"
x_train_AD['ScoreAD'] = trainAD

x_test_AD = x_test_GA.copy()
x_test_AD['SetClass'] = "Test"
x_test_AD['ScoreAD'] = testAD

AllPeptidesDomain = pd.concat([x_train_AD, x_test_AD], axis=0)

AllPeptidesDomain["OutAD"] = (AllPeptidesDomain.ScoreAD < 0).tolist()

df = AllPeptidesDomain.sample(frac = 1).reset_index()
sns.scatterplot(data=df, x= range(len(AllPeptidesDomain)), y="ScoreAD", hue="SetClass")
df.to_json("AllTrainTestPeptidesDomainAnalisis.json")
AD=AD.sample(frac = 1)

sns.scatterplot(data=AD, x=AD.index, y="ScoreAD", hue="SetClass")
plt.hlines(0, 0, 4000,'b')





# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
#                     GENERACION Y OPTIMIZACION DE MODELOS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.metrics import *
import sklearn
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
import pygad
import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

prefijos = ["VsAll/PeptidesVsAll_",
            "VsGramN/PeptidesVsGramN_",
            "VsGramP/PeptidesVsGramP_",
            "VsEscherichia/PeptidesVsEscherichia_",
            "VsPseudomonas/PeptidesVsPseudomonas_",
            "VsStaphylococcus/PeptidesVsStaphylococcus_",
            "VsBacillus/PeptidesVsBacillus_"
    ]


for i in prefijos:
    pref = i
    
    #pref = "VsGramN/PeptidesVsGramN_"  # SubfijoResultados:
    
    data_train = pd.read_pickle(pref+'AMP_Data_Train.pkl')
    data_test = pd.read_pickle(pref+'AMP_Data_Test.pkl')
    
    pipe_prepross = load(pref+'PipelineNormScalAllDescriptors.joblib')
    best_variables = pd.read_excel(pref+"BestTrainingDescriptorsGA.xlsx", index_col=None)
    rs=321
    umbral_amp = 25      # ACTIVO
    umbral_no_amp = 100   # NO ACTIVO
    umbral=0.5
    negative_samples = 0000
    #best variables read 
    
    
    names = [x[7:] for x in best_variables[best_variables.VarSelected == 1.0][0]]
    
    x_train_final= data_train[names]
    x_test_final= data_test[names]
    
    y_train = data_train['AMP']
    y_test = data_test['AMP']
    
    # ------------FINAL PREPRECESSOR----------------------
    
    preprocessorFinal = ColumnTransformer(
        [('scale', StandardScaler(), x_train_final.columns)],
        remainder='passthrough').fit(x_train_final)
    
    x_train_final = pd.DataFrame(preprocessorFinal.transform(x_train_final),
                                       columns=preprocessorFinal.get_feature_names_out())
    x_test_final = pd.DataFrame(preprocessorFinal.transform(x_test_final),
                                       columns=preprocessorFinal.get_feature_names_out())
    
    # Save pipeline de normalizacion y escalado de variables
    
    dump(preprocessorFinal, pref + 'FinalPreprocessing.joblib')
    
    
    # Grid de hiperparámetros evaluados
    # ==============================================================================
    
    rf_param_grid = {'n_estimators': [200, 300, 500, 600, 700, 800, 1000, 1200, 1500],  # 1000
                     'max_features': [4, 5, 6, 7, 10, 12],  # auto
                     'max_depth': [None, 2, 4],
                     'random_state': [rs]
                     }
    
    lr_param_grid = {'C': [0.75, 1.0, 1.25, 1.50, 1.75],
                     'penalty': ['l1', "l2", "elasticnet", "None"],
                     'random_state': [rs]
                     }
    
    svm_param_grid = {'C': [0.75, 1.0, 1.25, 1.50, 1.75],
                      'degree': [2, 3, 4, 5],
                      'random_state': [rs]
                      }
    
    gbm_param_grid = {'n_estimators': [250, 500, 1000, 1500, 2000],
                      'max_features': [2, 5, 6, 8, 10],
                      'max_depth': [None, 2, 4, 8, 16, 32],
                      'random_state': [rs]
                      }
    
    
    
    # Búsqueda por grid search con validación cruzada
    # ==============================================================================
    
    
    models = {"ModelRF": RandomForestClassifier(),
              "ModelLR": LogisticRegression(),
              "ModelSVMPoly": SVC(kernel='poly',probability=True),
              "ModelGBM": GradientBoostingClassifier(),
              }
    
    models_param = {"ModelRF": rf_param_grid,
                    "ModelLR": lr_param_grid,
                    "ModelSVMPoly": svm_param_grid,
                    "ModelGBM": gbm_param_grid
                    }
    
    grid_search_resultados = {}
    mejores_modelos = {}
    
    for i in tqdm(models, desc="Procesando", ncols=75, ascii=True):
        print("GRID SEARCH IN PROGRESS  ", i)
        
        t1 = time.time()
        grid = GridSearchCV(
            estimator=models[i],
            param_grid=models_param[i],
            scoring='matthews_corrcoef',
            n_jobs=-1,
            cv=StratifiedKFold(5),
            refit=True,
            verbose=True,
            return_train_score=True
        )
        
        grid.fit(X=x_train_final, y=y_train)
        print('     Time search:', time.time() - t1)
        
        result = pd.DataFrame(grid.cv_results_).sort_values('mean_test_score', ascending=False)
        grid_search_resultados[i] = result
        mejores_modelos[i] = grid.best_estimator_
        dump(grid.best_estimator_, pref + i+ '.joblib')
     
    
    dump(grid_search_resultados,pref+"gridsearch_resultados.joblib" )
    
    
    #=============================================================================
    #DETERMINACION DEL DOMINIO DE APLICABILIDAD:
    #=============================================================================
    
    from sklearn.neighbors import LocalOutlierFactor
    
    pipeDomain = Pipeline([('AD_LOF',  LocalOutlierFactor(n_neighbors=20,contamination= 'auto', novelty=True))
                           ], verbose=True)
    pipeDomain.fit(x_train_final)
    
    dump(pipeDomain, pref + 'pipeApDomain.joblib')
    
    #ALLDATA: DOMAIN
    
    trainAD = pipeDomain.decision_function(x_train_final)
    testAD = pipeDomain.decision_function(x_test_final)
    
    x_train_AD = x_train_final.copy()
    x_train_AD['SetClass'] = "Train"
    x_train_AD['ScoreAD'] = trainAD
    
    x_test_AD = x_test_final.copy()
    x_test_AD['SetClass'] = "Test"
    x_test_AD['ScoreAD'] = testAD
    
    AllPeptidesDomain = pd.concat([x_train_AD, x_test_AD], axis=0)
    
    AllPeptidesDomain["OutAD"] = (AllPeptidesDomain.ScoreAD < 0).tolist()
    
    df = AllPeptidesDomain.sample(frac = 1).reset_index()
    sns.scatterplot(data=df, x= range(len(AllPeptidesDomain)), y="ScoreAD", hue="SetClass")
    plt.hlines(0, 0, len(df),'#AEB6BF',linestyles="dashed")
    plt.savefig(pref+"plot_AD.svg")
    
    df.to_csv(pref+"AllTrainTestPeptidesDomainAnalisis.csv")
    
    #METRICAS Y FIGURAS PARA LOS MODELOS
    from sklearn.metrics import *
    from sklearn.ensemble import *
    from sklearn.linear_model import *
    from sklearn.svm import *
    
    
    def roc_curva(modelo, x_test,y_test):
        RocCurveDisplay.from_estimator(modelo, x_test,y_test)
        plt.show()
        
    
    
    def evaluacion_modelos_clasificacion (y_test,y_predict): 
        #MATRIZ DE CONFUSION
        print("===================================================================")
        df_cm = pd.DataFrame(confusion_matrix(y_test,y_predict), columns= ['Predicted 0 ', 'Predicted 1'], 
                             index = ['Real 0 ', 'Real 1'])
        print("\nMATRIZ DE CONFUSION \n", df_cm)
        
        df_report = classification_report(y_test, y_predict)
        
        print("\n\nREPORTE DE CLASIFICACION \n",df_report)
    
    
    def metrics(y_test,y_predict): 
        mc = matthews_corrcoef(y_test,y_predict)
        pr = precision_score(y_test, y_predict) 
        Sn = recall_score(y_test, y_predict, pos_label=1)
        Sp = recall_score(y_test, y_predict, pos_label=0)
        f1 = f1_score(y_test, y_predict)
        ac = accuracy_score(y_test, y_predict)
        #ck = cohen_kappa_score (y_test,y_predict)
        #print("Kappa de Cohen: ", ck)
        resultados = {'Mathew Correlation': mc ,
                      'Exactitud(accuray )':ac,
                      'Precision':pr,
                      'Sensibilidad':Sn,
                      'Especificidad': Sp,
                      'F1':f1,
                      'Umbral_amp': umbral_amp,
                      'Umbral_no_amp': umbral_no_amp,
                      'Umbral': umbral,
                      'AddNegat': negative_samples,
            }
        
        print(resultados)
    
        return resultados
    
    
    def all_metrics(modelo, x_test,y_test): 
        y_predict = modelo.predict(x_test)
        roc_curva(modelo, x_test,y_test)
        evaluacion_modelos_clasificacion(y_test,y_predict)
        m = metrics(y_test,y_predict)
        m['ROC'] = roc_auc_score(y_test, modelo.predict_proba(x_test)[:, 1])
        
        return m
    
    #RESULTADOS DE LOS MODELOS GENERADOS PARA CADA CONJUNTO DE DATOS
    
    resultados_train = []
    roc_data = {}
    from sklearn import metrics as mtr
    
    
    for i in mejores_modelos:
        print(i)
        resultados_train.append(all_metrics(mejores_modelos[i], x_train_final, y_train))
        
    resultados_train= pd.DataFrame(resultados_train, index = models.keys())
    resultados_train['Class'] = "Train"
    
    
    resultados_test = []
    for i in mejores_modelos:
        print(i)
        resultados_test.append(all_metrics(mejores_modelos[i], x_test_final, y_test))
    
    resultados_test= pd.DataFrame(resultados_test, index = models.keys())
    resultados_test['Class'] = "Test"
    
    Resultados = pd.concat([resultados_train,resultados_test])
    Resultados.to_excel(pref+'ResultadosModelosVsAll.xlsx')
    
    #GRAFICO CURVA ROC DE LOS MODELOS GENERADOS PARA CADA CONJUNTO DE DATOS
    
    import matplotlib.pyplot as plt
    
    from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
    
    fig, [ax_train, ax_test] = plt.subplots(1, 2, figsize=(14, 6))
    
    for i in mejores_modelos:
        RocCurveDisplay.from_estimator(mejores_modelos[i], x_test_final, y_test,ax=ax_test)
        RocCurveDisplay.from_estimator(mejores_modelos[i], x_train_final, y_train,ax=ax_train)
        #DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det, name=name)
    
    ax_train.set_title("Curvas ROC en el entrenamiento ")
    ax_test.set_title("Curvas ROC en la prueba ")
    
    ax_train.grid(linestyle="--")
    ax_test.grid(linestyle="--")
    plt.savefig(pref+'plot_roc.svg', dpi=300)
    

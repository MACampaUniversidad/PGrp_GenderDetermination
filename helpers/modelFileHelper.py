import difflib 
import pandas as pd
import numpy as np
import shared.supportedFiles as const
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_curve 
from sklearn.datasets import make_classification  
from sklearn.metrics import roc_auc_score  
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

#define a class to help us with the process

class ModelFileHelper(object):
    '''Ayuda a dar una descripcion de un fichero y a su carga '''
    def __init__(self, csvFile):
        self.__configurePandas()
        self.csvFile= pd.read_csv(csvFile) 
        self.fileName=csvFile
        self.ColumnDescriptions ={}
    def getDescription(self):
        return self.csvFile.describe()

    def setColumDescriptionsFile(self, fileFormat, filepath, columnkeyIndex=0, removeNotFoundColumns=True , excelSheet=0):
        ''' vincula las descripciones de las columnas del csv a las descripciones existentes en un fichero adicional.
        fileFormat, usar la enumeracion SupportedFiles
        filepath es el fichero diccionario.
        columnKeyIndex el índice de columna que contiene el nombre de las columnas.
        removeNotFoundColumns elimina del csv cargado en el helper aquellas columnas no encontradas en el csv diccionario.
        el metodo retorna una lista con las columnas eliminadas cuando removeNotFoundColumns es verdadero.
        el metodo retorna una lista con las columnas No encontradas en el csv de datos cuando removeNotFoundColumns es falso.
        excelSheet por defecto 0 es la hoja excel de donde queremos cargar los datos '''

        dataFrame = self.__readFileFormat(filepath,fileFormat,excelSheet)
        #leer y montar en un diccionario key-value donde value es un join del resto de las columnas que no son el columnIndex
        for index, row in dataFrame.iterrows():
            self.ColumnDescriptions[row[columnkeyIndex]] = "".join(row[columnkeyIndex:])

         #ver que columnas existen y qué columnas no   
        resultado = [] 
        for columnName in self.csvFile.columns:
            if not columnName in self.ColumnDescriptions:
                item = columnName  + " No encontrada en CSV de datos "
                if (removeNotFoundColumns):
                    #eliminar columna
                    resultado.append( item + "Accion: Eliminada")
                    self.dropColumn(columnName)
                else:
                    #no eliminar, informar.
                    resultado.append( item + "Accion: Ninguna")
        return resultado   
    def setColumnsType(self, dictionaryOfColumnNamesTypes):
        self.csvFile=self.csvFile.astype(dictionaryOfColumnNamesTypes)

    def removeUselessColumns (self, dropConstants=True, dropQualifiyingColums=True, stdThreshold=0.05, Silent=False):
        ''' Elimina columnas constantes por debajo de un valor dado para la desviacion estandar, stdThreshold. El modo 
        Silent activa o desactiva la salida del resultado de las acciones '''
        if (dropConstants):
            #mediante el analisis de la desviación típica elminamos los valores que esten por debajo de la desviacion propuesta.
            #para retirar variables proximas a ser constantes del modelo.
            columns =  self.csvFile.columns.copy()
            for column in columns:
                stdDeviation=self.csvFile[column].std(ddof=0)
                if (stdDeviation<=stdThreshold):
                    self.dropColumn(column)
                    if (Silent==False):
                        print (" Column:" + column + " removed : Standart Deviation = " + str(stdDeviation))
 
        if (dropQualifiyingColums):
           self.csvFile = self.csvFile.select_dtypes(exclude=np.object) #las cadenas de texto las interpreta como objectos.   

    def fillGapsUsingMultivariable(self, method='median', useNearestVariables=None):
        '''  method puede ser, "mean”, “median”, “most_frequent”, or “constant” , useNearestVariables numero de variables cercanas para inferir el valor, por defecto todas. '''
        
        iterativeImputer = IterativeImputer(random_state=0, initial_strategy= method,n_nearest_features=useNearestVariables )
        #el iterative imputer elimina las columnas tenemos que guardarlas para luego volver a establecerlas.
        savedColumNames= self.csvFile.columns.to_list() 
        self.csvFile = pd.DataFrame(iterativeImputer.fit_transform(self.csvFile)) 
        self.csvFile.columns=savedColumNames

    def replaceColumnTextByDictionaryValues(self , dictionary, column, naValue=''):
        ''' remplaza el contenido de las celdas de una columna que coincidan con los valores de un diccionario por su valor.'''
        #normalizar a minusculas ya que tenemos valores como Sanscrit y sanscrit, que son lo mismo a nivel de significante.
        self.csvFile[column]=self.csvFile[column].str.lower() 
        for key in dictionary:
            self.csvFile[column]= self.csvFile[column].replace(key, dictionary[key])

        self.csvFile[column]=  self.csvFile[column].fillna(naValue)  
    
    def dropColumn(self, columnName):
        ''' wrapper para eliminar una columna'''
        self.csvFile= self.csvFile.drop(columnName, axis=1)

    def getModelTypeDetail(self, fileName=None):
        ''' Retorna una estructura legible con los tipos de dato del conjunto de datos del csv cargado
             exporta la salida a filename si se proporciona un nombre de fichero '''
        if (fileName==None) :    
            return self.__translateTypestoHumanReadable(self.csvFile.dtypes)
        else:
            with open( fileName, 'w') as f:
               print(self.__translateTypestoHumanReadable(self.csvFile.dtypes), file=f)     

    def findDifferences(self, other):
        '''Retorna una lista con la comparacion de las columnas y los tipos de dos csv'''
        returnlist = list (difflib.Differ().compare(self.getModelTypeDetail().to_string().splitlines(1), other.getModelTypeDetail().to_string().splitlines(1)))
        returnlist.append("Comparativa de tamaños: ")
        returnlist.append (self.fileName +  " Filas:" + ''.join(self.__tuplaCleanUp(self.csvFile.shape[0:1])) + " Columnas:" +  ''.join(self.__tuplaCleanUp(self.csvFile.shape[1:2])))
        returnlist.append (other.fileName + " Filas:" + ''.join(self.__tuplaCleanUp(other.csvFile.shape[0:1])) + " Columnas:" +  ''.join(self.__tuplaCleanUp(other.csvFile.shape[1:2])))
        return returnlist

    def pearson(self,  A,  B):
        ''' indice de coorrelacion lineal de Pearson de la variable A con respecto a variable B. 
        Acotado entre [1 , -1] indicando |1| alta coorrelacion y en el caso de ser negativo el coeficiente, correlación inversa '''
        pearson = self.csvFile[A].corr(self.csvFile[B])
        return pearson

    def removeColPearsonCriteria(self, removeUnderValue, compareWithColumn):
        for (columnName, columnData) in self.csvFile.iteritems():
            pearsonIndex= self.pearson(columnName, compareWithColumn)
            if( abs(pearsonIndex) <= abs(removeUnderValue)  ):
                self.dropColumn(columnName)
                print("Columna eliminada: "+str(columnName)+". índice de Pearson: "+str(abs(pearsonIndex)) + " <= " + str(abs(removeUnderValue)) )
            else:
                 print("Columna NO eliminada: "+str(columnName)+". índice de Pearson: "+str(abs(pearsonIndex)) + " > = " + str(abs(removeUnderValue)) )

    def exportToCsv(self, fileName):
        self.csvFile.to_csv(fileName, index=False)
    
    def exportHarmonizatedModel(self, harmonizationMatrix, harmonizationquery, fileName):
        '''Exporta el modelo tras armonizar los valores en funcion de una matriz de armonización dada y una query'''
        harmonizated =self.csvFile 
        dataframe = pd.DataFrame(harmonizationMatrix)
        for index, trainedRow in dataframe.iterrows() :
            group =harmonizated.query(harmonizationquery) 
            for index, groupRow in  group.iterrows():
                randomVal= np.random.randint(trainedRow['Min'], trainedRow['Max'])
                harmonizated.loc[(harmonizated.PassengerId  ==  groupRow.PassengerId) , "Age"]=randomVal
        #dump to csv
        print ("volcando a archivo harmonizated_train.csv")
        harmonizated.to_csv(fileName,  index=False)  

    def nullCounts(self):
        ''' Vuelca informacion sobre el numero de valores nulos en las diferentes columnas '''
        self.csvFile.info(verbose=True, null_counts=True)

    def getNullPercents(self):
        ''' Recupera un diccionario donde las claves son los nombres y de columna y los valores un diccionario anexado
        con clave '%' para el % de valores nulos y 'description' para obtener la descripcion de la variable (diccionario) '''
        total_rows = self.csvFile.shape[0] #(rows, colums)
        result ={}
        for  column in self.csvFile.loc[:, self.csvFile.isnull().any()] :
            notnullValues= self.csvFile[column].count()
            #key -> (%,Descripcion)
            try:
                description = self.ColumnDescriptions[column]
            except KeyError:
                description= "no description available"

            result[column]={ '%' :100 * float(total_rows-notnullValues) / float(total_rows), 'description': description}
        return sorted(result.items(), key= lambda x: x[1]['%'], reverse=True ) 

    def removeColumnsHavingNulls(self, threshold, Silent=False):
        ''' Elimina las columnas que tienen un umbral de nulos por encima del proporcionado '''
        removable = [x for x in self.getNullPercents() if x[1]['%'] >= threshold ]
        for column in removable:
            self.dropColumn(column[0])
            if (Silent == False):
                print("Removed " + column[0] + " having a " + str(column[1]['%']) + " Percent of nulls"  )

    def viewUniqueColumnValues(self, column):
        return pd.unique(self.csvFile[column]).tolist()    

    def getHeatMap(self, corrMethod, dimensionX, dimensionY):
        corr=self.csvFile.corr(method=corrMethod) 
        plt.figure(figsize=(dimensionX, dimensionY))
        sea.heatmap(corr, xticklabels=True, yticklabels=True) 

    def getBestPredictionAlgorithm(self, predictColumn, identifierColumn, testModel,  Silent=False, ROC_Curve=True ):
        '''
        Se usan varios algoritmos de prediccion para determinar cuál es el mejor:
        predictColumn es la columna a ser predecida. El modelo entrenado se asume que es el contenido en la instancia del helper. Self.
        identifierColumn: normalmente es una columna que hace las veces de indexador identificando de forma unica un registro.
        testModel: El modelo de test contra el que se realiza la predicción. (Pandas Dataframe)

        El modo Silent activa la salida de texto informativo
        El modo ROC_Curve genera la curva ROC para cada algoritmo ayudando de forma visual a la determinación del mejor.
        Retorna el mejor algoritmo en base a la mejor precisión para cada uno de los algorimos empleados en la estimación que son los siguientes:
          - Regresión logística
          - Random Forest
          - Perceptrón
          - Árboles de decisión
          - Bayesiano (Naybe)
          - K-NeigthBors (Vecindad de 5)
        '''
       
        #inicializar los clasificadores
        xgb=XGBClassifier(n_estimators=500,n_jobs =16,max_depth=16)
        logReg= LogisticRegression()
        rForest= RandomForestClassifier(n_estimators=500,n_jobs =16,max_depth=16)
        pctron = Perceptron(max_iter=2000 )
        decTree = DecisionTreeClassifier(max_depth=16)
        gaussNb= GaussianNB()
        knNeighbors = KNeighborsClassifier(n_neighbors=5, algorithm='auto') #usamos auto para dejar a el que decida el mejor.
        
        #Slicing del modelo
          #el metodo drop de pandas no elimina la columna en el dataframe; retorna un dataframe como resultado de la operacion.
        xTrain = self.csvFile.drop(predictColumn, axis=1, inplace=False)
        yTrain = self.csvFile[predictColumn]

        xTest=testModel.copy() #asegurarnos que trabajamos sobre una copia no sobre el modelo por referencia.
        
        if (Silent==False):
             print ("harmonizing the train model, ensure column equivalences:") 

        #armonizamos el modelo de test eliminando aquellas columnas que no estén presentes en el modelo de entrenamiento
        for testColumn in xTest.columns:
            if testColumn in xTrain.columns:
                if (Silent==False): print (testColumn + " exist at train model.") 
            else:
                xTest.drop(testColumn,axis=1, inplace=True)
                if (Silent==False): print (testColumn + " does not exist at train model. REMOVED from test model") 

        #ordenamos las columnas para que esten en el mismo orden todas:
        xTest.sort_index(axis=1, inplace=True)
        xTrain.sort_index(axis=1,inplace=True)
        precisiones =[]      

        #medimos la capacidad explicativa de cada modelo entrenado sobre el propio xtrain. Cómo es capaz de explicar la variable dependiente.
        #if (Silent==False):
        #     print ("Procesando XGBoost...")   
        #xgb.fit(xTrain,yTrain)
        #yPred = xgb.predict(xTrain)
        #probs = xgb.predict_proba(xTrain)
        #resultado= self.__getResultado(identifierColumn,predictColumn,xTrain, yPred)
        #precisiones= [('XGBoost',  str(round (xgb.score(xTrain, yTrain)*100,2 )), resultado.copy(),probs.copy())]
        #-------
        #if (Silent==False):
        #    print ("Procesando Regresión logística...")   
        #logReg.fit(xTrain,yTrain)
        #yPred = logReg.predict(xTrain)
        #probs = logReg.predict_proba(xTrain)
        #resultado= self.__getResultado(identifierColumn,predictColumn,xTrain, yPred)
        #precisiones.append(('Regresión Logística',  str(round (logReg.score(xTrain, yTrain)*100,2 )), resultado.copy(),probs.copy()))
        #-------
        if (Silent==False):
            print ("Procesando Random Forest...")
        rForest.fit(xTrain,yTrain)
        yPred = rForest.predict(xTest)
        probs = rForest.predict_proba(xTest)
        resultado= self.__getResultado(identifierColumn,predictColumn,xTest, probs)
        precisiones.append(('Random Forest',  str(round (rForest.score(xTrain, yTrain)*100,2 )), resultado.copy(), yPred.copy()))
        #-------
        #if (Silent==False):
        #    print ("Procesando Perceptron...")   
        #pctron.fit(xTrain,yTrain)
        #yPred = pctron.predict(xTrain)
        #probs = pctron._predict_proba_lr(xTrain)
        #resultado= self.__getResultado(identifierColumn,predictColumn,xTrain, yPred)
        #precisiones.append(('Perceptron',  str(round (pctron.score(xTrain, yTrain)*100,2 )), resultado.copy(),probs.copy(),probs.copy()))
        #-------
        #if (Silent==False):
        #    print("Procesando árboles de decisión...")
        #decTree.fit(xTrain,yTrain)
        #yPred = decTree.predict(xTrain)
        #probs = decTree.predict_proba(xTrain)
        #resultado= self.__getResultado(identifierColumn,predictColumn,xTrain, yPred)
        #precisiones.append(('árboles de decisión',  str(round (decTree.score(xTrain, yTrain)*100,2 )), resultado.copy(),probs.copy()))
        #-------
        #if (Silent==False):
        #    print ("Procesando Naybe Bayes...")
        #gaussNb.fit(xTrain,yTrain)
        #yPred = gaussNb.predict(xTrain)
        #probs = gaussNb.predict_proba(xTrain)
        #resultado= self.__getResultado(identifierColumn,predictColumn,xTrain, yPred)
        #precisiones.append(('Naybe Bayes',  str(round (gaussNb.score(xTrain, yTrain)*100,2 )), resultado.copy(),probs.copy(),probs.copy()))
        #-------
        #if (Silent==False):
        #    print ("Procesando K-Neighbours...")
        #knNeighbors.fit(xTrain,yTrain)
        #yPred = knNeighbors.predict(xTrain)
        #probs = knNeighbors.predict_proba(xTrain)
        #resultado= self.__getResultado('test_id','is_female',xTrain, yPred)
        #precisiones.append(('K-Neighbours',  str(round (knNeighbors.score(xTrain, yTrain)*100,2 )), resultado.copy(),probs.copy()))
    
        #ordenar de mayor a menor usando el score de los modelos: 
        precisiones.sort(key=lambda tupla: tupla[1], reverse=True) 
        if (Silent==False):
            print ("Mejor algoritmo: " + precisiones[0][0] + " " + precisiones[0][1] + "%" )


        #precisiones contiene una tupla con los siguientes valoes : 
        # posicion [0] nombre del algoritmo
        # posicion [1] xcore o calidad del modelo predictivo
        # posicion [2] resultado de la prediccion para el modelo.
        # posicion [3] probabilidades para curva ROC
        for precision in precisiones:

            if (Silent==False) :
                print(precision[0] + " " + precision[1])
            if (ROC_Curve==True):
                probabilities = precision[2]
                predictions =  precision[3]
                self.__calculateRocAucCurve(probabilities[predictColumn], predictions, precision[0])    
           

        print("exportando el mejor de los modelos:")
        tupla = precisiones[0]
        tupla[2].to_csv( tupla[0] +"_submission.csv", index=False)
        print ("Generado " +tupla[0] + "_submission.csv" )  
    
    def __getResultado(self, identifierColumn, predictedColumn, test, predictions):
        return pd.DataFrame({identifierColumn : test[identifierColumn], predictedColumn: predictions})
    
    def __calculateRocAucCurve(self, testy, probs, modelName):
        #Nos quedamos con las probabilidades de la clase positiva únicamente
        probs = probs[:, 1]

        #Calculamos puntuación AUC y dibujamos curva ROC. El valor ideal de AUC es 1.
        auc = roc_auc_score(testy, probs)
        print('AUC calculada para el modelo '+modelName+': %.2f' % auc)
        fpr, tpr, thresholds = roc_curve(testy, probs)  
        self.__drawRocCurve(fpr, tpr, modelName)
        
        return auc

    def __drawRocCurve(self, fpr, tpr, modelName):
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('Tasa positiva falsa - FPR')
        plt.ylabel('Tasa positiva verdadera - TPR')
        plt.title('Curva ROC (Receiver Operating Characteristic) del modelo: '+modelName)
        plt.legend()
        plt.show()

    def __tuplaCleanUp(self, tupla):
        result = str(tupla).replace('(','').replace(')','').replace(',','')
        return result

    def __translateTypestoHumanReadable(self, text):
        return text.replace("int64", "Numero").replace("object", "Cadena de texto AlfaNumerica").replace("float64", "Numero (largo)")
    
    def __readFileFormat(self, filepath, format, excelSheetIndex=0):
        if (format == const.SupportedFiles.EXCEL):
            return pd.read_excel(io= filepath, sheet_name = excelSheetIndex)
        elif (format == const.SupportedFiles.CSV):
            return pd.read_csv(filepath)
        return 

    def __configurePandas(self):
        pd.set_option('display.max_rows', 3000)
        pd.set_option('display.max_columns', 3000)

    def _viewUniqueColumnValues(self, column):
        return pd.unique(self.csvFile[column]).tolist()   
        

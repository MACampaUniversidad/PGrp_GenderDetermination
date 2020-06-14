#Este fichero contendrá las constantes que usaremos a lo largo de la practica
#por conveniencia REVISAR la guia de estilos de python https://www.python.org/dev/peps/pep-0008/
#para que todos sigamos un estilo más o menos común:
import helpers.modelFileHelper as mh

#Clave para acceder al file helper del fichero Train
TRAIN_KEY = "train"
# Clave para acceder al file helper del fichero Test
TEST_KEY = "test"
#Clave para acceder al file helper del fichero Train modificado en la parte 1
TRAIN_KEY_PART_2 = "train2"
# Clave para acceder al file helper del fichero Test modificado en la parte 1
TEST_KEY_PART_2 = "test2"
#PART2 nombres de ficheros
FILENAME_TEST_PART2 = "data_sources/test_part2.csv"
FILENAME_TRAIN_PART2 = "data_sources/train_part2.csv"

# Diccionario que contiene los ficheros CSV bajo las claves. Pueden ser accedidos mediante las claves train y test 
CSV_FILES = { TRAIN_KEY : mh.ModelFileHelper("data_sources/train.csv"), 
    TEST_KEY :   mh.ModelFileHelper("data_sources/test.csv"), 
    TEST_KEY_PART_2 : mh.ModelFileHelper(FILENAME_TEST_PART2) , 
    TRAIN_KEY_PART_2 : mh.ModelFileHelper(FILENAME_TRAIN_PART2) } 

# Diccionario que contiene todos los posibles lenguajes para transcribirlos a un valor numérico.
LANGUAGES = {
'bengali':1,
'hindi':2,
'tamil':4,
'malayalam':5,
'chattisgari':6,
'telugu':7,
'marathi':8,
'nepali':9,
'rajasthani':10,
'marathi & hindi':11,
'assamese':12,
'telugu & kannada':13,
'punjabi':14,
'kannada':15,
'oriya':16,
'english':17,
'none':18,
'gujarati':19,
'kokborok':20,
'gujarati & marathi':21,
'manipuri':22,
'hindi & oriya':23,
'hindi & marathi':24,
'urdu':25,
'mewari':26,
'hindi & rajasthani':27,
'hindi & punjabi':28,
'thadou':29,
'punjabi & hindi':30,
'konkani':31,
'govan':32,
'konkani & hindi':33,
'tribal language':34,
'aadiwasi':35,
'urdu, kannada & hindi':36,
'sanskrit':37,
'hindi & urdu':38,
'bhojpuri':39,
'meitei':40,
'karbi':41,
'arbi':42,
'haryanvi':43,
'kannada & urdu':44,
'dehati':45,
'kannada & telugu':46,
'marathi & kannada':47,
'hindi & gujarati':48,
'hindi, oriya & bengali':49,
'hindi & kannada':50,
'oriya, hindi & bengali':51,
'santhali':52,
'oriya & hindi':53,
'rabha':54,
'hindi, marathi & gujarati':55,
'assamese and hindi':56,
'mundri':57,
'urdu & kannada':58,
'marwadi':59,
'urdu & hindi':60,
'jharkhandi':61,
'english, hindi & bengali':62,
'chhetriya bhasha':63,
'boro':64,
'unnao':65,
'malayalam & tamil':66,
'tamil & kannada':67,
'telugu & tamil':68,
'marathi & konkani':69,
'sowtali':70,
'gujarati & hindi':71,
'kannada & marathi':72,
'rajasthan or hindi':73,
'hindi & tamil':74,
'kannada, telugu & hindi':75,
'marathi & gujarati':76,
'samtal':77,
'gondi':78,
'kuki':79,
'bengali & hindi':80,
'kannada,marathi & hindi':81,
'hindi & bhojpuri':82,
'kannada & hindi':83,
'urdu & marathi':84,
'english, hindi & manipuri':85,
'mathli':86,
'hindi & manipuri':87
}
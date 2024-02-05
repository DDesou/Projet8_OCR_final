# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Creation of the mounts

# COMMAND ----------

# MAGIC %md
# MAGIC **data container mount**

# COMMAND ----------

storage_account_name = "projet8images"
container_name = "imagesdata"
storage_account_key = "zez1wQv/LGqyCCNn1LAKv0jTyT29ilq0/ccqsEHT1X4CyKOlHa7bOAPu7xm5xglPmnCU1UVFvRnb+AStFfOftw=="

dbutils.fs.mount(
    source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/",
    mount_point = "/mnt/projet8",
    extra_configs = {"fs.azure.account.key.projet8images.blob.core.windows.net": storage_account_key}
)

# COMMAND ----------

# MAGIC %md
# MAGIC **results container mount**

# COMMAND ----------

storage_account_name = "projet8images"
container_name = "results2"
storage_account_key = "zez1wQv/LGqyCCNn1LAKv0jTyT29ilq0/ccqsEHT1X4CyKOlHa7bOAPu7xm5xglPmnCU1UVFvRnb+AStFfOftw=="

dbutils.fs.mount(
    source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/",
    mount_point = "/mnt/results2",
    extra_configs = {"fs.azure.account.key.projet8images.blob.core.windows.net": storage_account_key}
)

# COMMAND ----------

# MAGIC %md
# MAGIC storage_account_name = "projet8images"
# MAGIC container_name = "results3"
# MAGIC storage_account_key = "zez1wQv/LGqyCCNn1LAKv0jTyT29ilq0/ccqsEHT1X4CyKOlHa7bOAPu7xm5xglPmnCU1UVFvRnb+AStFfOftw=="
# MAGIC
# MAGIC dbutils.fs.mount(
# MAGIC     source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/",
# MAGIC     mount_point = "/mnt/results3",
# MAGIC     extra_configs = {"fs.azure.account.key.projet8images.blob.core.windows.net": storage_account_key}
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC **result table container mount**

# COMMAND ----------

storage_account_name = "projet8images"
container_name = "mytables"
storage_account_key = "zez1wQv/LGqyCCNn1LAKv0jTyT29ilq0/ccqsEHT1X4CyKOlHa7bOAPu7xm5xglPmnCU1UVFvRnb+AStFfOftw=="

dbutils.fs.mount(
    source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/",
    mount_point = "/mnt/mytables",
    extra_configs = {"fs.azure.account.key.projet8images.blob.core.windows.net": storage_account_key}
)

# COMMAND ----------

# MAGIC %md
# MAGIC **show the path_data for the images**

# COMMAND ----------

dbutils.fs.ls("/mnt/projet8/fruits/fruits/fruits-360-original-size/fruits-360-original-size/Test/")

# COMMAND ----------

dbutils.fs.ls("/mnt/results2/")

# COMMAND ----------

# MAGIC %md
# MAGIC # I. Librairies & packages

# COMMAND ----------

#File system manangement
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from os import listdir

from pathlib import Path
import pandas as pd
import pyspark.pandas as ps
from PIL import Image
import numpy as np
import io

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, element_at, split
from pyspark.sql import SparkSession

#second PCA
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.functions import col, size, udf, lit
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import VectorUDT, Vectors
import matplotlib.pyplot as plt

# COMMAND ----------

os.getcwd()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Installations (if needed)

# COMMAND ----------

#!pip install Pandas pillow tensorflow pyspark pyarrow

# COMMAND ----------

#pip install matplotlib

# COMMAND ----------

# MAGIC %md
# MAGIC # II. Define paths

# COMMAND ----------

PATH_Data = "/mnt/projet8/fruits/fruits/fruits-360-original-size/fruits-360-original-size/Test/"

# COMMAND ----------

PATH_Result = '/mnt/results2/'

# COMMAND ----------

PATH_Tables = '/mnt/mytables/'

# COMMAND ----------

# MAGIC %md
# MAGIC # III. Notebook from another person

# COMMAND ----------

# MAGIC %md
# MAGIC ## III.1 Création de la SparkSession (not necessary)
# MAGIC
# MAGIC
# MAGIC
# MAGIC L’application Spark est contrôlée grâce à un processus de pilotage (driver process) appelé **SparkSession**. <br />
# MAGIC <u>Une instance de **SparkSession** est la façon dont Spark exécute les fonctions définies par l’utilisateur <br />
# MAGIC dans l’ensemble du cluster</u>. <u>Une SparkSession correspond toujours à une application Spark</u>.
# MAGIC
# MAGIC <u>Ici nous créons une session spark en spécifiant dans l'ordre</u> :
# MAGIC  1. un **nom pour l'application**, qui sera affichée dans l'interface utilisateur Web Spark "**P8**"
# MAGIC  2. que l'application doit s'exécuter **localement**. <br />
# MAGIC    Nous ne définissons pas le nombre de cœurs à utiliser (comme .master('local[4]) pour 4 cœurs à utiliser), <br />
# MAGIC    nous utiliserons donc tous les cœurs disponibles dans notre processeur.<br />
# MAGIC  3. une option de configuration supplémentaire permettant d'utiliser le **format "parquet"** <br />
# MAGIC    que nous utiliserons pour enregistrer et charger le résultat de notre travail.
# MAGIC  4. vouloir **obtenir une session spark** existante ou si aucune n'existe, en créer une nouvelle

# COMMAND ----------

# MAGIC %md
# MAGIC **not necessary anymore**
# MAGIC spark = (SparkSession
# MAGIC              .builder
# MAGIC              .appName('P8')
# MAGIC              .master('local')
# MAGIC              .config("spark.sql.parquet.writeLegacyFormat", 'true')
# MAGIC              .getOrCreate()
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC <u>Nous créons également la variable "**sc**" qui est un **SparkContext** issue de la variable **spark**</u> :

# COMMAND ----------

# MAGIC %md
# MAGIC ## III.2 SparkContext

# COMMAND ----------

sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md
# MAGIC <u>Affichage des informations de Spark en cours d'execution</u> :

# COMMAND ----------

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## III.3 Traitement des données
# MAGIC
# MAGIC <u>Dans la suite de notre flux de travail, <br />
# MAGIC nous allons successivement</u> :
# MAGIC 1. Préparer nos données
# MAGIC     1. Importer les images dans un dataframe **pandas UDF**
# MAGIC     2. Associer aux images leur **label**
# MAGIC     3. Préprocesser en **redimensionnant nos images pour <br />
# MAGIC        qu'elles soient compatibles avec notre modèle**
# MAGIC 2. Préparer notre modèle
# MAGIC     1. Importer le modèle **MobileNetV2**
# MAGIC     2. Créer un **nouveau modèle** dépourvu de la dernière couche de MobileNetV2
# MAGIC 3. Définir le processus de chargement des images et l'application <br />
# MAGIC    de leur featurisation à travers l'utilisation de pandas UDF
# MAGIC 3. Exécuter les actions d'extraction de features
# MAGIC 4. Enregistrer le résultat de nos actions
# MAGIC 5. Tester le bon fonctionnement en chargeant les données enregistrées
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### III.3.1 Chargement des données
# MAGIC
# MAGIC Les images sont chargées au format binaire, ce qui offre, <br />
# MAGIC plus de souplesse dans la façon de prétraiter les images.
# MAGIC
# MAGIC Avant de charger les images, nous spécifions que nous voulons charger <br />
# MAGIC uniquement les fichiers dont l'extension est **jpg**.
# MAGIC
# MAGIC Nous indiquons également de charger tous les objets possibles contenus <br />
# MAGIC dans les sous-dossiers du dossier communiqué.

# COMMAND ----------

images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load(PATH_Data)

# COMMAND ----------

# MAGIC %md
# MAGIC <u>Affichage des 5 premières images contenant</u> :
# MAGIC  - le path de l'image
# MAGIC  - la date et heure de sa dernière modification
# MAGIC  - sa longueur
# MAGIC  - son contenu encodé en valeur hexadécimal

# COMMAND ----------

# MAGIC %md
# MAGIC <u>Je ne conserve que le **path** de l'image et j'ajoute <br />
# MAGIC     une colonne contenant les **labels** de chaque image</u> :

# COMMAND ----------

images = images.withColumn('label', element_at(split(images['path'], '/'),-2))
print(images.printSchema())
print(images.select('path','label').show(5,False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### III.3.2 Préparation du modèle
# MAGIC
# MAGIC Je vais utiliser la technique du **transfert learning** pour extraire les features des images.<br />
# MAGIC J'ai choisi d'utiliser le modèle **MobileNetV2** pour sa rapidité d'exécution comparée <br />
# MAGIC à d'autres modèles comme *VGG16* par exemple.
# MAGIC
# MAGIC Pour en savoir plus sur la conception et le fonctionnement de MobileNetV2, <br />
# MAGIC je vous invite à lire [cet article](https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c).
# MAGIC
# MAGIC <u>Voici le schéma de son architecture globale</u> : 
# MAGIC
# MAGIC ![Architecture de MobileNetV2](img/mobilenetv2_architecture.png)
# MAGIC
# MAGIC Il existe une dernière couche qui sert à classer les images <br />
# MAGIC selon 1000 catégories que nous ne voulons pas utiliser.<br />
# MAGIC L'idée dans ce projet est de récupérer le **vecteur de caractéristiques <br />
# MAGIC de dimensions (1,1,1280)** qui servira, plus tard, au travers d'un moteur <br />
# MAGIC de classification à reconnaitre les différents fruits du jeu de données.
# MAGIC
# MAGIC Comme d'autres modèles similaires, **MobileNetV2**, lorsqu'on l'utilise <br />
# MAGIC en incluant toutes ses couches, attend obligatoirement des images <br />
# MAGIC de dimension (224,224,3). Nos images étant toutes de dimension (100,100,3), <br />
# MAGIC nous devrons simplement les **redimensionner** avant de les confier au modèle.
# MAGIC
# MAGIC <u>Dans l'odre</u> :
# MAGIC  1. Nous chargeons le modèle **MobileNetV2** avec les poids **précalculés** <br />
# MAGIC     issus d'**imagenet** et en spécifiant le format de nos images en entrée
# MAGIC  2. Nous créons un nouveau modèle avec:
# MAGIC   - <u>en entrée</u> : l'entrée du modèle MobileNetV2
# MAGIC   - <u>en sortie</u> : l'avant dernière couche du modèle MobileNetV2

# COMMAND ----------

model = MobileNetV2(weights='imagenet',
                    include_top=True,
                    input_shape=(224, 224, 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### III.3.3 Broadcast weights (added)

# COMMAND ----------

#bc_model_weights = sc.broadcast(model.get_weights())

# COMMAND ----------

new_model = Model(inputs=model.input,
                  outputs=model.layers[-2].output)

# COMMAND ----------

# MAGIC %md
# MAGIC Affichage du résumé de notre nouveau modèle où nous constatons <br />
# MAGIC que <u>nous récupérons bien en sortie un vecteur de dimension (1, 1, 1280)</u> :

# COMMAND ----------

new_model.summary()

# COMMAND ----------

brodcast_weights = sc.broadcast(new_model.get_weights())

# COMMAND ----------

# MAGIC %md
# MAGIC <u>Mettons cela sous forme de fonction</u> :

# COMMAND ----------

def model_fn():
    """
    Returns a MobileNetV2 model with top layer removed 
    and broadcasted pretrained weights.
    """
    model = MobileNetV2(weights='imagenet',
                        include_top=True,
                        input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    new_model = Model(inputs=model.input,
                  outputs=model.layers[-2].output)
    new_model.set_weights(brodcast_weights.value)
    return new_model

# COMMAND ----------

# MAGIC %md
# MAGIC ### III.3.4 Définition du processus de chargement des images et application <br/>de leur featurisation à travers l'utilisation de pandas UDF
# MAGIC
# MAGIC Ce notebook définit la logique par étapes, jusqu'à Pandas UDF.
# MAGIC
# MAGIC <u>L'empilement des appels est la suivante</u> :
# MAGIC
# MAGIC - Pandas UDF
# MAGIC   - featuriser une série d'images pd.Series
# MAGIC    - prétraiter une image

# COMMAND ----------

def preprocess(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)

def featurize_series(model, content_series):
    """
    Featurize a pd.Series of raw images using the input model.
    :return: a pd.Series of image features
    """
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    '''
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).

    :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
    '''
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches.  This amortizes the overhead of loading big models.
    model = model_fn()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)

# COMMAND ----------

# MAGIC %md
# MAGIC ### III.7.4 Exécution des actions d'extraction de features
# MAGIC
# MAGIC Les Pandas UDF, sur de grands enregistrements (par exemple, de très grandes images), <br />
# MAGIC peuvent rencontrer des erreurs de type Out Of Memory (OOM).<br />
# MAGIC Si vous rencontrez de telles erreurs dans la cellule ci-dessous, <br />
# MAGIC essayez de réduire la taille du lot Arrow via 'maxRecordsPerBatch'
# MAGIC
# MAGIC Je n'utiliserai pas cette commande dans ce projet <br />
# MAGIC et je laisse donc la commande en commentaire.

# COMMAND ----------

# spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

# COMMAND ----------

# MAGIC %md
# MAGIC Nous pouvons maintenant exécuter la featurisation sur l'ensemble de notre DataFrame Spark.<br />
# MAGIC <u>REMARQUE</u> : Cela peut prendre beaucoup de temps, tout dépend du volume de données à traiter. <br />
# MAGIC
# MAGIC Notre jeu de données de **Test** contient **22819 images**. <br />
# MAGIC Cependant, dans l'exécution en mode **local** => not anymore, <br />
# MAGIC nous <u>traiterons un ensemble réduit de **330 images** =>no </u>.

# COMMAND ----------

features_df = images.repartition(20).select(col("path"),
                                            col("label"),
                                            featurize_udf("content").alias("features")
                                           )

# COMMAND ----------

# MAGIC %md
# MAGIC <u>Rappel du PATH où seront inscrits les fichiers au format "**parquet**" <br />
# MAGIC contenant nos résultats, à savoir, un DataFrame contenant 3 colonnes</u> :
# MAGIC  1. Path des images
# MAGIC  2. Label de l'image
# MAGIC  3. Vecteur de caractéristiques de l'image

# COMMAND ----------

print(PATH_Result)

# COMMAND ----------

features_df.write.mode("overwrite").parquet(PATH_Result)

# COMMAND ----------

# MAGIC %md
# MAGIC # IV. Chargement des données enregistrées et validation du résultat
# MAGIC
# MAGIC <u>On charge les données fraichement enregistrées dans un **DataFrame Pandas**</u> :

# COMMAND ----------

PATH_Result

# COMMAND ----------

df = ps.read_parquet(PATH_Result, engine='pyarrow')

# COMMAND ----------

df.head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC <u>On valide que la dimension du vecteur de caractéristiques des images est bien de dimension 1280</u> :

# COMMAND ----------

df.loc[0,'features'].shape

# COMMAND ----------

df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Nous venons de valider le processus sur un jeu de données allégé en local <br />
# MAGIC où nous avons simulé un cluster de machines en répartissant la charge de travail <br />
# MAGIC sur différents cœurs de processeur au sein d'une même machine.
# MAGIC
# MAGIC Nous allons maintenant généraliser le processus en déployant notre solution <br />
# MAGIC sur un réel cluster de machines et nous travaillerons désormais sur la totalité <br />
# MAGIC des 22819 images de notre dossier "Test".

# COMMAND ----------

# MAGIC %md
# MAGIC # V. Dimension reduction (PCA)

# COMMAND ----------

type(df)

# COMMAND ----------

df.loc[0, 'features']

# COMMAND ----------

df.features.to_numpy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## V.1 First try : discarded

# COMMAND ----------

data = pd.DataFrame(data=list(df.features))

# COMMAND ----------

data

# COMMAND ----------

col = ['col'+str(ele) for ele in data.columns]

# COMMAND ----------

data.columns = col

# COMMAND ----------

data.to_csv(PATH_Result+'/data.csv', index=False)

# COMMAND ----------

from pyspark.ml.feature import RFormula, PCA
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

from pyspark.ml.linalg import Vectors
#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# COMMAND ----------

data = sqlContext.read.load(PATH_Result+'/data.csv', format='com.databricks.spark.csv', delimiter = ',', header='true', inferSchema='true')

# COMMAND ----------

type(data)

# COMMAND ----------

data

# COMMAND ----------

data.show()

# COMMAND ----------

#dataPCA = PCA(k=20, inputCol=str(data.columns), outputCol="pcaFeatures")
#model = dataPCA.fit(data)

# COMMAND ----------

data = RFormula(formula=" ~ {0}".format(" + ".join(data.columns))).fit(data).transform(data)

# COMMAND ----------

data

# COMMAND ----------

dataPCA = PCA(k=50, inputCol=str(data.columns), outputCol="pcaFeatures")
#model = dataPCA.fit(data)

# COMMAND ----------

dataPCA

# COMMAND ----------

dataPCA.setInputCol("features").fit(data).transform(data)

# COMMAND ----------

pcaModel = dataPCA.fit(data)

# COMMAND ----------

pcaModel.getK

# COMMAND ----------

pcaModel.explainedVariance

# COMMAND ----------

cumValues = pcaModel.explainedVariance.cumsum() # get the cumulative values

# COMMAND ----------

cumValues

# COMMAND ----------

# plot the graph 
plt.figure(figsize=(10,8))
plt.plot(range(1,51), cumValues, marker = 'o', linestyle='--')
plt.title('variance by components')
plt.xlabel('num of components')
plt.ylabel('cumulative explained variance')

# COMMAND ----------

pcaModel.setOutputCol("output")

# COMMAND ----------

data.show()

# COMMAND ----------

type(data)

# COMMAND ----------

pcaModel.transform(data).collect()[0].output

# COMMAND ----------

pcaModel.transform(data).collect()[0]

# COMMAND ----------

df_ = pcaModel.transform(data)

# COMMAND ----------

type(df_)

# COMMAND ----------

df_.show()

# COMMAND ----------

df_.select('output').collect()

# COMMAND ----------

df2_ = spark.createDataFrame(df_.select('output').collect(),["features"])

# COMMAND ----------

df2_.show()

# COMMAND ----------

type(df2_)

# COMMAND ----------

# MAGIC %md
# MAGIC from pyspark.ml.functions import vector_to_array
# MAGIC df3_ = df2_.withColumn('features', vector_to_array('features'))

# COMMAND ----------

#data.count()

# COMMAND ----------

# Save the DataFrame to a Parquet file
parquet_path = PATH_Result+'/myparquet'

# COMMAND ----------

df2_.write.parquet(parquet_path)

# COMMAND ----------

# Read the Parquet file to verify
df_read = spark.read.parquet(parquet_path)
#df_read.show()

# COMMAND ----------

type(df_read)

# COMMAND ----------

df_read.collect()[0]

# COMMAND ----------

df_read.collect()[0][0][-1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## V.2 Second PCA (ok)

# COMMAND ----------

num_elements = len(features_df.select('features').first()['features'])

# COMMAND ----------

num_elements

# COMMAND ----------

# fct utilisateur pour convertir ArrayType en VectorUDT
def array_to_vector_udf(array_col):
    return Vectors.dense(array_col)

# COMMAND ----------

#enregistrement de la fct utilisateur en tant qu'UDF
array_to_vector = udf(array_to_vector_udf, VectorUDT())
features_df = features_df.withColumn('vector_features', array_to_vector('features'))

# COMMAND ----------

pca = PCA(k=num_elements, inputCol='vector_features')
pca_model = pca.fit(features_df)

# COMMAND ----------

print(pca_model)

# COMMAND ----------

pca_model.setOutputCol('pca_features')

# COMMAND ----------

pca_exp = pca_model.explainedVariance

# COMMAND ----------

cumValues = pca_exp.cumsum() # get the cumulative values

# COMMAND ----------

cumValues[199] #selection of the 200 st components (new_features)

# COMMAND ----------

# plot the graph 
plt.figure(figsize=(10,8))
plt.plot(range(1,num_elements+1), cumValues, marker = 'o', linestyle='--')
plt.title('variance by components')
plt.xlabel('num of components')
plt.ylabel('cumulative explained variance')
plt.show()

# COMMAND ----------

pca_features = pca_model.transform(features_df)

# COMMAND ----------

#pca_features.show()

# COMMAND ----------

pca_features.select('pca_features').show()

# COMMAND ----------

# extract the 200 st components
# define an user fct to extract the 2 fist elts 
def extract_first_n_udf(features, n):
    return Vectors.dense(features[:n])

# COMMAND ----------

extract_first_n = udf(extract_first_n_udf, VectorUDT()) #enregistrer la fct utilisateur en tant q'UDF

# COMMAND ----------

#specficy the number to extract (ex:)
n_elements = 200

# COMMAND ----------

#appliuqer l'UDF pour extraire les n premiers elts
pca_features = pca_features.withColumn('pca200', extract_first_n(col('pca_features'), lit(n_elements)))

# COMMAND ----------

pca_features.columns

# COMMAND ----------

features_df = features_df.join(pca_features.select('path', 'pca200'), 'path')

# COMMAND ----------

type(features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # VI. Saving results

# COMMAND ----------

# MAGIC %md
# MAGIC ## IV.1 Defining path results + saving

# COMMAND ----------

PATH_Result

# COMMAND ----------

pca_path = PATH_Result+'/mypca'

# COMMAND ----------

# Save the DataFrame to a Parquet file
features_df.write.parquet(pca_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## IV.2 Open parquet data

# COMMAND ----------

# Read the Parquet file to verify
df_read = spark.read.parquet(pca_path)
df_read.show()

# COMMAND ----------

df_ = ps.read_parquet(pca_path, engine='pyarrow')

# COMMAND ----------

df_.head(3)

# COMMAND ----------

len(df_.loc[0, 'features']), len(df_.loc[0, 'vector_features']), len(df_.loc[0, 'pca200'])

# COMMAND ----------

type(df_)

# COMMAND ----------



# COMMAND ----------

pca_path

# COMMAND ----------

data_dir = Path(pca_path)

# COMMAND ----------

data_dir.glob('*.parquet')

# COMMAND ----------

dbutils.fs.ls(pca_path)[-1][0]

# COMMAND ----------

ps.read_parquet(dbutils.fs.ls(pca_path)[-1][0], engine='pyarrow').head(3)

# COMMAND ----------

l = []
for elt in dbutils.fs.ls(pca_path):
    if elt[0].endswith('.parquet'):
        l.append(elt[0])

# COMMAND ----------

l

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert to pandasDF

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws
from pyspark.sql.types import StringType

# create a SparkSession
spark = SparkSession.builder.master("local").appName("MyApp").getOrCreate()

# read the first Parquet file to create a schema for the DataFrame
df = spark.read.parquet(l[0])

# read each Parquet file and union it with the initial DataFrame
for parquet_file in l[1:]:
    df = df.unionAll(spark.read.parquet(parquet_file))

# convert the "ARRAY<FLOAT>" column to a string type and "STRUCT" columns to string types
df = df.withColumn('features_str', concat_ws(',', 'features'))
df = df.drop('features')
df = df.withColumn('pca200_str', col('pca200').cast(StringType()))
df = df.drop('pca200')

# remove vector_feat
df = df.drop('vector_features')

# COMMAND ----------

df.show()

# COMMAND ----------

# write the final DataFrame to a CSV file
#df.write.csv(f'{PATH_Result}/csv_file', header=True, mode='overwrite')

# COMMAND ----------

pandasDF = df.toPandas()

# COMMAND ----------

pandasDF.head(3)

# COMMAND ----------

pandasDF.shape

# COMMAND ----------

type(pandasDF)

# COMMAND ----------

PATH_Tables

# COMMAND ----------

# MAGIC %md
# MAGIC ### /!\ Saving into one csv file

# COMMAND ----------

pandasDF.to_csv('/dbfs/mnt/mytables/table.csv')

# COMMAND ----------

dbutils.fs.ls(PATH_Tables)

# COMMAND ----------

# MAGIC %md
# MAGIC ## IV.3 Open the csv data (ok!)

# COMMAND ----------

csv_path = PATH_Tables

# COMMAND ----------

dbutils.fs.ls(csv_path)[-1][0]

# COMMAND ----------

# Read the csv files to verify
df3 = ps.read_csv(dbutils.fs.ls(csv_path)[-1][0], index_col='_c0')
df3.head(3)

# COMMAND ----------

type(df3.to_pandas())

# COMMAND ----------

df3.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## IV.4 Save csv in my workspace

# COMMAND ----------

df3.to_pandas().to_csv(os.getcwd()+'/table.csv')

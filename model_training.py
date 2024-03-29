from pyspark.sql import SparkSession
import pyspark.ml.feature
from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,IDF,StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import json
import findspark

findspark.init() 

spark = SparkSession.builder.appName("LyricsClassifierApp").getOrCreate()
df = spark.read.csv("datasets\Merged_dataset.csv",header=True,inferSchema=True)
df = df.select('artist_name', 'track_name', 'release_date', 'genre', 'lyrics')

tokenizer = Tokenizer(inputCol='lyrics',outputCol='mytokens')
stopwords_remover = StopWordsRemover(inputCol='mytokens',outputCol='filtered_tokens')
vectorizer = CountVectorizer(inputCol='filtered_tokens',outputCol='rawFeatures')
idf = IDF(inputCol='rawFeatures',outputCol='vectorizedFeatures')

labelEncoder = StringIndexer(inputCol='genre',outputCol='label').fit(df)

df = labelEncoder.transform(df)

df_grouped = df.groupBy("genre", "label").count()

result_dict = {}
for row in df_grouped.rdd.collect():
    label = row["label"]
    genre = row["genre"]
    result_dict[label] = genre

with open("datasets/class_labels2.json", "w") as f:
    json.dump(result_dict, f)

(trainDF,testDF) = df.randomSplit((0.8,0.2),seed=42)

lr = LogisticRegression(featuresCol='vectorizedFeatures',labelCol='label')

pipeline = Pipeline(stages=[tokenizer,stopwords_remover,vectorizer,idf,lr])

lr_model = pipeline.fit(trainDF)

lr_model.save("/248357D/trained_models/eight_class_pipeline")

predictions = lr_model.transform(testDF)

evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')

accuracy = evaluator.evaluate(predictions)

print ('Accuracy of the model : ',accuracy)
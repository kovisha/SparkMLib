import json
from concurrent.futures import ThreadPoolExecutor, Future
from flask import Flask,render_template, url_for
from flask import request
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType

import findspark

findspark.init()

app = Flask(__name__, static_url_path='/static')

@app.route("/")
def index():
    lyrics = request.args.get("lyrics", "")
    if lyrics:
        print(lyrics)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(predicted_genre,lyrics)
        result_genre = future.result()
        image_url = url_for('static', filename='bar_chart.png')
        return render_template('output.html', image_url=image_url, predicted_genre=result_genre)
    return render_template('index.html')


def predicted_genre(lyrics):
    spark = SparkSession.builder \
        .master("local") \
        .appName("Music Classification") \
        .config("spark.executor.memory", "1gb") \
        .getOrCreate()

    sc = spark.sparkContext

    schema = StructType([StructField("lyrics", StringType(), True)])
    data = [(lyrics,), ]  # add comma to make it a tuple
    df = spark.createDataFrame(data=data, schema=schema)

    lr_model = PipelineModel.load("trained_models/seven_class_pipeline")
    predictions = lr_model.transform(df)
    probabilities = predictions.select('probability').collect()[0][0].toArray()
    with open('datasets/class_labels.json', 'r') as file:
        json_string = file.read()
    probability_dictionary = json.loads(json_string)
    probability_dictionary = {k: v for k, v in sorted(probability_dictionary.items())}
    labels = list(probability_dictionary.values())
    predicted_genre = probability_dictionary[str(predictions.select('prediction').collect()[0][0])]
    fig, ax = plt.subplots()
    ax.bar(labels, probabilities, color='#ed95ce', label='Probability of belonging to each Music Genre')
    ax.legend(loc='upper right')
    plt.savefig('static/bar_chart.png')
    plt.close()
    return predicted_genre

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)



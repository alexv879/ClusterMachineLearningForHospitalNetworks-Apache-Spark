
"""
Hospital Resource Demand Prediction
This does training regression models and creating a classification example in a cluster of computers considering a hospital network. The goal is to predict hospital resource metrics 
(e.g., Length of Stay) and to classify patients into risk groups (e.g., high vs. low LOS) to help with operational improvements.

Regression Models Included:
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor

Classification Example:
    - LOS is binarized (e.g., high LOS if above a threshold) and 
      a Decision Tree Classifier and Random Forest Classifier are trained.
      
Key metrics (features):
    - admission_count, current_occupancy, emergency_visits, seasonality_index

Update the CONFIG section as needed.
"""

import matplotlib.pyplot as plt
import pandas as pd

# Import PySpark modules
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType, DoubleType
from pyspark.sql.functions import current_timestamp, when
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#these are the configuration paramenters for the hdfs and the input from all the hospitals in the network that contribute to the information used towards the unbounded tables


# 1. CONFIGURATION PARAMETERS

CONFIG = {
    "appName": "HospitalResourcePredictionExtended",
    "hdfsInputPath": "hdfs://namenode:9000/hospitals/incoming/",  # CSV uploads directory
    "checkpointLocation": "hdfs://namenode:9000/checkpoints/hospital_stream/",
    "outputTable": "hospital_unbounded_table",
    "trainingWindowStart": "2025-03-31 22:00:00",  # Adjust as needed
    "trainingWindowEnd": "2025-03-31 23:00:00",
    "hdfsMaster": "spark://master-node-address:7077",
    "modelSavePath": "hdfs://namenode:9000/hospitals/models/latest_model",
    "losThreshold": 5.0  # Example threshold to binarize LOS for classification
}


# 2. INITIALIZE SPARK SESSION

spark = SparkSession.builder \
    .appName(CONFIG["appName"]) \
    .master(CONFIG["hdfsMaster"]) \
    .getOrCreate()

###insure that all data columns with relevant information have the same name or use the join function

# 3. DEFINE SCHEMA & INGEST STREAMING DATA WITH WATERMARKING

schema = StructType([
    StructField("hospital_id", StringType(), True),
    StructField("event_time", TimestampType(), True),
    StructField("admission_count", IntegerType(), True),
    StructField("current_occupancy", IntegerType(), True),
    StructField("emergency_visits", IntegerType(), True),
    StructField("seasonality_index", DoubleType(), True),
    StructField("length_of_stay", DoubleType(), True)  # Target variable for regression
])

# Read CSV files as a streaming DataFrame from HDFS.
streaming_df = spark.readStream \
    .option("header", True) \
    .schema(schema) \
    .csv(CONFIG["hdfsInputPath"])

# Add watermark to handle late-arriving data.
streaming_df = streaming_df.withWatermark("event_time", "10 minutes")
streaming_df = streaming_df.withColumn("ingest_time", current_timestamp())

#here is where the machine learning code should be so that the data is processed
#this is the location where the learning in the module should happen

def ML():

######################## code from the chatgpt on foreachBatch to make the data stream correctly
# Define a function to train the model incrementally using foreachBatch
def train_model_on_batch(batch_df, batch_id):
    # Preprocess the data for the model
    processed_data = vectorAssembler.transform(batch_df)

    # Fit the LogisticRegression model on this batch of data
    model = lr.fit(processed_data)

    # Use the trained model to make predictions
    predictions = model.transform(processed_data)
    
    # Optionally, save the model or predictions (this step depends on your use case)
    # Example: Save the model after every batch
    model.save(f"models/logistic_regression_model_batch_{batch_id}")

    # You could also store predictions or log them (here we just print)
    predictions.show()

###########################Consider refactoring the data based on this information and adding this to the final model.

# Write streaming data to an unbounded table.
query_stream = streaming_df.writeStream.foreachBatch(ML) \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", CONFIG["checkpointLocation"]) \
    .table(CONFIG["outputTable"])

# Uncomment below in production to keep the stream running:
# query_stream.awaitTermination()


# 4. EXTRACT TRAINING DATA FROM THE UNBOUNDED TABLE

training_window_query = f"""
    SELECT *
    FROM {CONFIG["outputTable"]}
    WHERE event_time BETWEEN '{CONFIG["trainingWindowStart"]}' AND '{CONFIG["trainingWindowEnd"]}'
"""
training_df = spark.sql(training_window_query).na.drop()


# 5. FEATURE CONSTRUCTION FOR REGRESSION

# Valuable features for predicting LOS.
feature_cols = ["admission_count", "current_occupancy", "emergency_visits", "seasonality_index"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_data = assembler.transform(training_df).select("features", "length_of_stay")

# Split the data into training and testing sets.
train_data, test_data = final_data.randomSplit([0.7, 0.3], seed=42)

#for regression models we use different models that offer certain important information to end user hospitals
# 6. REGRESSION MODELS


# 6A. Linear Regression
lr = LinearRegression(featuresCol="features", labelCol="length_of_stay")
lr_model = lr.fit(train_data)
lr_predictions = lr_model.transform(test_data)

# 6B. Decision Tree Regressor
dt = DecisionTreeRegressor(featuresCol="features", labelCol="length_of_stay")
dt_model = dt.fit(train_data)
dt_predictions = dt_model.transform(test_data)

# 6C. Random Forest Regressor
rf = RandomForestRegressor(featuresCol="features", labelCol="length_of_stay")
rf_model = rf.fit(train_data)
rf_predictions = rf_model.transform(test_data)

#this is where the model is being evaluated using RMSE
# Evaluate each model using RMSE.
reg_evaluator = RegressionEvaluator(labelCol="length_of_stay", predictionCol="prediction", metricName="rmse")
lr_rmse = reg_evaluator.evaluate(lr_predictions)
dt_rmse = reg_evaluator.evaluate(dt_predictions)
rf_rmse = reg_evaluator.evaluate(rf_predictions)

print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Decision Tree Regression RMSE: {dt_rmse}")
print(f"Random Forest Regression RMSE: {rf_rmse}")


# 7. CLASSIFICATION EXAMPLE
#here is where we use different examples for classifiers that can be effective
# Create a binary label for classification based on LOS threshold.
# For example, label = 1 if length_of_stay > threshold (high LOS), else 0.
training_df = training_df.withColumn("LOS_binary", 
                when(training_df["length_of_stay"] > CONFIG["losThreshold"], 1).otherwise(0))
# Reassemble features for classification.
classification_data = assembler.transform(training_df).select("features", "LOS_binary")
class_train, class_test = classification_data.randomSplit([0.7, 0.3], seed=42)

# Train a Decision Tree Classifier.
dt_classifier = DecisionTreeClassifier(featuresCol="features", labelCol="LOS_binary")
dt_class_model = dt_classifier.fit(class_train)
dt_class_predictions = dt_class_model.transform(class_test)

# Train a Random Forest Classifier.
rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="LOS_binary")
rf_class_model = rf_classifier.fit(class_train)
rf_class_predictions = rf_class_model.transform(class_test)

# Evaluate classifiers using accuracy.
class_evaluator = MulticlassClassificationEvaluator(labelCol="LOS_binary", predictionCol="prediction", metricName="accuracy")
dt_accuracy = class_evaluator.evaluate(dt_class_predictions)
rf_accuracy = class_evaluator.evaluate(rf_class_predictions)

print(f"Decision Tree Classifier Accuracy: {dt_accuracy}")
print(f"Random Forest Classifier Accuracy: {rf_accuracy}")


# 8. VISUALIZATION: REGRESSION RESULTS
#this is where we visualise the results of the regression 
# For demonstration, convert Linear Regression predictions to Pandas DataFrame.
predictions_pd = lr_predictions.select("length_of_stay", "prediction").toPandas()
predictions_pd["residual"] = predictions_pd["length_of_stay"] - predictions_pd["prediction"]

plt.figure(figsize=(8, 6))
plt.scatter(predictions_pd["length_of_stay"], predictions_pd["prediction"], alpha=0.6, color='blue')
plt.xlabel("Actual Length of Stay")
plt.ylabel("Predicted Length of Stay")
plt.title("Predicted vs. Actual Length of Stay (Linear Regression)")
plt.plot([predictions_pd["length_of_stay"].min(), predictions_pd["length_of_stay"].max()],
         [predictions_pd["length_of_stay"].min(), predictions_pd["length_of_stay"].max()],
         color='red', lw=2)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(predictions_pd["prediction"], predictions_pd["residual"], alpha=0.6, color='green')
plt.xlabel("Predicted Length of Stay")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot (Linear Regression)")
plt.axhline(y=0, color='red', linestyle='--')
plt.show()
#each of these plots will report on different information according to the labels

# 9. FEATURE IMPORTANCE FROM TREE-BASED MODELS
#this is where we add the results of the models and inform the end users on the predictions meaning
print("\n--- Feature Importances ---")
print("Decision Tree Regressor Feature Importances:")
for feature, importance in zip(feature_cols, dt_model.featureImportances):
    print(f"{feature}: {importance}")

print("Random Forest Regressor Feature Importances:")
for feature, importance in zip(feature_cols, rf_model.featureImportances):
    print(f"{feature}: {importance}")

#use dataframe here as well 
# 10. SAVE MODELS & REPORT INSIGHTS
#this is where the model needs to be reported to the end user/users
# Save the regression models; you can similarly save classifier models if needed.
lr_model.write().overwrite().save(CONFIG["modelSavePath"] + "/lr")
dt_model.write().overwrite().save(CONFIG["modelSavePath"] + "/dt")
rf_model.write().overwrite().save(CONFIG["modelSavePath"] + "/rf")

print("\n--- Operational Insights ---")
print("Analysis indicates that high admission counts, increased emergency visits, and seasonal trends")
print("correlate with longer patient lengths of stay. The regression models provide RMSE values as follows:")
print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Decision Tree Regression RMSE: {dt_rmse}")
print(f"Random Forest Regression RMSE: {rf_rmse}")
print("For classification (high vs. low LOS), accuracies are:")
print(f"Decision Tree Classifier Accuracy: {dt_accuracy}")
print(f"Random Forest Classifier Accuracy: {rf_accuracy}")
print("Recommendations: Adjust staffing and optimize discharge procedures during peak times to reduce LOS.")
print("----------------------------")

# Stop the Spark session.
spark.stop()
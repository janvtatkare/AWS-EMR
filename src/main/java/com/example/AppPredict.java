package com.example;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import static org.apache.spark.sql.functions.col;

public class AppPredict {

    // Clean dataset: Remove quotes and cast columns to DoubleType
    public static Dataset<Row> cleanData(Dataset<Row> df) {
        for (String colName : df.columns()) {
            String cleanColName = colName.replace("\"", ""); // Remove quotes
            df = df.withColumn(cleanColName, col(colName).cast(DataTypes.DoubleType)); // Cast to DoubleType
            if (!cleanColName.equals(colName)) {
                df = df.drop(colName); // Drop old column if renamed
            }
        }
        return df;
    }

    public static void main(String[] args) {
        // Initialize SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("test_wine_quality_prediction")
                .master("local[*]")
                .getOrCreate();

        // Set log level
        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("ERROR");

        // Path to test dataset
        String testPath = "s3a://janhavi-bucket/ValidationDataset.csv";  // Replace with your file path

        // Load and clean test data
        Dataset<Row> testData = spark.read()
                .format("csv")
                .option("header", "true")
                .option("sep", ";")
                .option("inferschema", "true")
                .load(testPath);
        testData = cleanData(testData);

        // Define feature columns
        String[] featureColumns = new String[]{
                "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                "pH", "sulphates", "alcohol"
        };

        // Feature assembler and label indexer
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol("features");

        StringIndexer indexer = new StringIndexer()
                .setInputCol("quality")
                .setOutputCol("label")
                .setHandleInvalid("skip");

        // List of model names
        String[] modelNames = {
                "LogisticRegression",
                "RandomForestClassifier",
                "DecisionTreeClassifier",
                "NaiveBayes",
                "MultilayerPerceptronClassifier"
        };

        // Iterate through models
        for (String modelName : modelNames) {
            System.out.println("Loading and testing model: " + modelName);

            // Load the model from S3
            String modelPath = "s3a://janhavi-bucket/" + modelName;
            PipelineModel model = PipelineModel.load(modelPath);

            // Transform the test dataset using the model
            Dataset<Row> predictions = model.transform(testData);

            // Evaluate the model
            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("label")
                    .setPredictionCol("prediction");

            double accuracy = evaluator.setMetricName("accuracy").evaluate(predictions);
            double f1Score = evaluator.setMetricName("f1").evaluate(predictions);

            // Output performance metrics
            System.out.printf("Model: %s | Accuracy: %.4f | F1 Score: %.4f%n",
                    modelName, accuracy, f1Score);
        }

        // Stop Spark session
        spark.stop();
    }
}

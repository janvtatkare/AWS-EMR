package com.example;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import static org.apache.spark.sql.functions.col;
public class App {
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
                .appName("wine_quality_prediction")
                .master("local[*]")
                .getOrCreate();
// Set log level
        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("ERROR");
        // Paths to datasets
        String inputPath = "s3://janhavi-bucket/TrainingDataset.csv";  // Replace with your file path
        String validPath = "s3://janhavi-bucket/ValidationDataset.csv";  // Replace with your file path
        // Load and clean training data
        Dataset<Row> trainData = spark.read()
                .format("csv")
                .option("header", "true")
                .option("sep", ";")
                .option("inferschema", "true")
                .load(inputPath);
        trainData = cleanData(trainData);
        // Load and clean validation data
        Dataset<Row> validData = spark.read()
                .format("csv")
                .option("header", "true")
                .option("sep", ";")
                .option("inferschema", "true")
                .load(validPath);
        validData = cleanData(validData);
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
        // List of classification algorithms
        List<Classifier> classifiers = new ArrayList<>();
        classifiers.add(new LogisticRegression()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setMaxIter(100)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setFamily("multinomial"));
        classifiers.add(new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setNumTrees(50));
        classifiers.add(new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features"));
        classifiers.add(new NaiveBayes()
                .setLabelCol("label")
                .setFeaturesCol("features"));
        classifiers.add(new MultilayerPerceptronClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setLayers(new int[]{11, 32, 16, 10}));  // Example layers: 11 inputs, 32 hidden, 16 hidden, 10 output classes
        // Iterate through classifiers
        for (Classifier classifier : classifiers) {
            System.out.println("Training and evaluating: " + classifier.getClass().getSimpleName());
            // Create pipeline
            Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, indexer, classifier});
            // Train model
            PipelineModel model = pipeline.fit(trainData);
            // Make predictions on validation data
            Dataset<Row> predictions = model.transform(validData);
// Evaluate model
            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("label")
                    .setPredictionCol("prediction");
            double accuracy = evaluator.setMetricName("accuracy").evaluate(predictions);
            double f1Score = evaluator.setMetricName("f1").evaluate(predictions);
            // Output performance metrics
            System.out.printf("Model: %s | Accuracy: %.4f | F1 Score: %.4f%n",
                    classifier.getClass().getSimpleName(), accuracy, f1Score);
            // Save the model
            String modelPath = "s3://janhavi-bucket/" + classifier.getClass().getSimpleName();
            try {
                model.write().overwrite().save(modelPath);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            System.out.println("Model saved to: " + modelPath);
        }
        // Stop Spark session
        spark.stop();
    }
}

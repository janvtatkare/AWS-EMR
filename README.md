# Wine Quality Prediction - Spark Application by ad2252(CS 643 859)

This project involves training a wine quality prediction machine learning model using Apache Spark on AWS EC2 instances. The model is trained in parallel on four EC2 instances and is designed for wine quality prediction based on input datasets.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setting Up the Environment](#setting-up-the-environment)
3. [Running the Model Training](#running-the-model-training)
4. [Running the Prediction Application](#running-the-prediction-application)
5. [Docker Setup](#docker-setup)
6. [File Structure](#file-structure)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Acknowledgments](#acknowledgments)

## Prerequisites

Before running the application, ensure that the following are installed:

- **Java 11 or higher**
- **Apache Maven**
- **Apache Spark 3.x**
- **AWS CLI (for accessing S3)**
- **Docker (for containerization)**

You can follow the instructions on the respective websites to install these tools.

## Setting Up the Environment

1. **Set up AWS EC2 instances**: You need four EC2 instances for training the model in parallel. Ensure that you have the following configured:
   - Spark installed on all EC2 instances.
   - AWS CLI configured for accessing S3 buckets.
   - Your datasets uploaded to S3.
   
2. **Set up the S3 bucket**: Create an S3 bucket (or use an existing one) to store your training, validation, and model files.

   - **Training Dataset**: `TrainingDataset.csv`
   - **Validation Dataset**: `ValidationDataset.csv`
   - **Model Output**: Models saved in the S3 bucket

## Running the Model Training

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/wine-quality-prediction.git
   cd wine-quality-prediction
   ```

2. **Update paths** in `App.java` to point to your S3 buckets for the training and validation datasets.

3. **Build the project**:

   Use Maven to build the project:

   ```bash
   mvn clean package
   ```

4. **Run the model training**:

   After building the project, run the training using:

   ```bash
   java -cp target/app-jar-with-dependencies.jar com.example.App
   ```

   This will train the model on the `TrainingDataset.csv`, validate the model on `ValidationDataset.csv`, and save the models to S3.

## Running the Prediction Application

1. **Update paths** in `AppPredict.java` for the test dataset and model locations.

2. **Build the project**:

   ```bash
   mvn clean package
   ```

3. **Run the prediction application**:

   After building the project, run the prediction application:

   ```bash
   java -cp target/apppredict-jar-with-dependencies.jar com.example.AppPredict
   ```

   This will load the model from S3, make predictions on the test dataset, and evaluate the model's performance (accuracy and F1 score).

## Docker Setup

### Building the Docker Image

1. **Create a Dockerfile**:

   In the project directory, create a `Dockerfile` with the following content:

   ```dockerfile
   FROM openjdk:11-jre-slim

   # Install necessary dependencies (if any)
   RUN apt-get update && apt-get install -y wget

   # Set environment variables
   ENV SPARK_VERSION 3.5.3
   ENV HADOOP_VERSION 3.2
   ENV AWS_REGION us-east-1

   # Download and install Spark
   RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
   RUN tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
   RUN mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /usr/local/spark

   # Set Spark environment variables
   ENV SPARK_HOME /usr/local/spark
   ENV PATH $SPARK_HOME/bin:$PATH

   # Copy the application JAR file into the container
   COPY target/apppredict-jar-with-dependencies.jar /app/

   # Set the entrypoint for the application
   ENTRYPOINT ["java", "-cp", "/app/apppredict-jar-with-dependencies.jar", "com.example.AppPredict"]
   ```

2. **Build the Docker image**:

   ```bash
   docker build -t wine-quality-predictor .
   ```

3. **Run the Docker container**:

   ```bash
   docker run -e AWS_ACCESS_KEY_ID=your-access-key -e AWS_SECRET_ACCESS_KEY=your-secret-key -e AWS_DEFAULT_REGION=your-region wine-quality-predictor
   ```

   This will run the prediction application inside the container.

## File Structure

```
wine-quality-prediction/
│
├── src/
│   ├── com/
│   │   ├── example/
│   │   │   ├── App.java
│   │   │   └── AppPredict.java
│   └── main/
│       └── resources/
│
├── pom.xml
├── Dockerfile
└── README.md
```

## Evaluation Metrics

The model's performance is evaluated using the following metrics:
- **Accuracy**: Measures the percentage of correct predictions.
- **F1 Score**: Harmonic mean of precision and recall.

The metrics are printed after each model evaluation in the console.

## Acknowledgments

- Apache Spark and MLlib for distributed machine learning and model training.
- AWS EC2 for cloud infrastructure.
- Docker for containerizing the prediction application.
- The dataset provided for training, validation, and testing.

---
# Thank You Janhavi Tatkare

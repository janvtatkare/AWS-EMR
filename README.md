Wine Quality Prediction
This project focuses on training a machine learning model to predict wine quality using Apache Spark on AWS EC2 instances. The model is trained in parallel on four EC2 instances and evaluates wine quality based on the provided datasets.

Table of Contents
Prerequisites
Environment Setup
Training the Model
Running the Prediction Application
Docker Integration
Project Structure
Performance Metrics
Acknowledgments
Prerequisites
Before starting, ensure you have the following installed:

Java 11 or later
Apache Maven
Apache Spark 3.x
AWS CLI (for S3 integration)
Docker (for containerized prediction application)
Follow official guides for installation instructions if required.

Environment Setup
Configure AWS EC2 Instances:

Set up four EC2 instances for parallel model training.
Install Spark on all instances.
Configure AWS CLI to enable access to S3.
Prepare the S3 Bucket:

Create or use an existing S3 bucket to store the datasets and models.
Upload the following files:
Training Dataset: TrainingDataset.csv
Validation Dataset: ValidationDataset.csv
Model Outputs: To be saved in the S3 bucket.
Training the Model
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/wine-quality-predictor.git
cd wine-quality-predictor
Update Paths:
Modify App.java to include the paths for the training and validation datasets in your S3 bucket.

Build the Project:
Use Maven to build the project:

bash
Copy code
mvn clean package
Run the Training Application:
Execute the training script:

bash
Copy code
java -cp target/app-jar-with-dependencies.jar com.example.App
This trains the model using TrainingDataset.csv, validates it on ValidationDataset.csv, and saves the models to the specified S3 bucket.

Running the Prediction Application
Update Paths:
Modify AppPredict.java to include the test dataset and model paths from S3.

Build the Project:
Rebuild the Maven project:

bash
Copy code
mvn clean package
Run the Prediction Application:
Execute the prediction application:

bash
Copy code
java -cp target/apppredict-jar-with-dependencies.jar com.example.AppPredict
This loads the model from S3, makes predictions, and evaluates its performance metrics (accuracy and F1 score).

Docker Integration
Build and Run the Docker Image
Create a Dockerfile:
Add the following Dockerfile to your project directory:

dockerfile
Copy code
FROM openjdk:11-jre-slim

RUN apt-get update && apt-get install -y wget

ENV SPARK_VERSION 3.5.3
ENV HADOOP_VERSION 3.2
ENV AWS_REGION us-east-1

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
RUN tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
RUN mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /usr/local/spark

ENV SPARK_HOME /usr/local/spark
ENV PATH $SPARK_HOME/bin:$PATH

COPY target/apppredict-jar-with-dependencies.jar /app/

ENTRYPOINT ["java", "-cp", "/app/apppredict-jar-with-dependencies.jar", "com.example.AppPredict"]
Build the Docker Image:

bash
Copy code
docker build -t wine-quality-predictor .
Run the Docker Container:

bash
Copy code
docker run -e AWS_ACCESS_KEY_ID=your-access-key \
           -e AWS_SECRET_ACCESS_KEY=your-secret-key \
           -e AWS_DEFAULT_REGION=your-region \
           wine-quality-predictor
Project Structure
css
Copy code
wine-quality-predictor/
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
Performance Metrics
The model is evaluated using:

Accuracy: The percentage of correct predictions.
F1 Score: A harmonic mean of precision and recall.
These metrics are displayed in the console after evaluation.

Acknowledgments
Apache Spark and MLlib for scalable machine learning.
AWS EC2 for cloud-based infrastructure.
Docker for containerized application deployment.
The dataset used for training and testing.
Thank you,
Janhavi Tatkare

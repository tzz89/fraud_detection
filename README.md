# Credit Card transaction Fraud Detection
In this project will be going through a sample Fraud detection project.
Below are the major steps in this project
1. Downloading data 
2. Basic Statistic EDA
3. Dimensional reduction EDA
4. Sklearn GridSearch, Crossvalidation cross various models
5. Deep learning using simple FeedForward Network
6. Exporting to onnx
7. Explainability using SHAP

## Dataset
The dataset have been taken from kaggle https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## EDA
In this section, we will going through some keypoints from EDA
### Imbalance dataset
As expected, the dataset is highly imbalanced (< 1% of transaction are fraud case). Therefore, we have to rely on metrics like PR curve for model comparision.
<img src="assets/label_distribution.png">


Neural network training
https://colab.research.google.com/drive/1f2ITJBrrAdgBOlJllKsWzYsBXjeDLbvP?usp=sharing
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
In total, there are 30 features and approximately 28k transaction and 500 fraud cases. The features are not the real features but a preprocessed feature using PCA

## EDA
In this section, we will going through some keypoints from EDA
### Imbalance dataset
As expected, the dataset is highly imbalanced (< 1% of transaction are fraud case). Therefore, we have to rely on metrics like PR curve for model comparision.
<img src="assets/label_distribution.png">

### Fraud over time
For this dataset, it seems like there are no apparent trend in the number of fraud cases and fraud amount over time
<img src="assets/fraud_over_time.png">

## Dimensionality reduction and visualization
As part of the EDA, I wanted to investigate if there are any clusters. 
### PCA
It seems like there are some cluster in the top left corner of the scatter plot but there are no obvious trend
<img src="assets/PCA.png">

### UMAP
For UMAP, it seems like there are no noticeable clusters. Another important observation that was made is the UMAP took a very long time to run
<img src="assets/UMAP.png">

### Improvements
I could have down sample the majority class to count that is similar to the number of fraud cases so clusters are that are form may be more apparent.

Neural network training
https://colab.research.google.com/drive/1f2ITJBrrAdgBOlJllKsWzYsBXjeDLbvP?usp=sharing
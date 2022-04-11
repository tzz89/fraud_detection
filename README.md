# Credit Card transaction Fraud Detection
In this project will be going through a sample Fraud detection project.
Below are the major steps in this project
1. Downloading data 
2. Basic Statistic EDA
3. Dimensional reduction EDA
4. Golden test set creation
5. Sklearn GridSearch, Crossvalidation cross various models
6. Deep learning using simple FeedForward Network
7. Exporting to onnx
8. Explainability using SHAP

## Dataset
The dataset have been taken from kaggle https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud <br>
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
For UMAP, it seems like there are no noticeable clusters. Another important observation that was made is the UMAP took a very long time to run <br>

<img src="assets/UMAP.png">

### Improvements
I could have down sample the majority class to count that is similar to the number of fraud cases so clusters are that are form may be more apparent.

## Golden test set creation
Before any further processing, the golden test set is first taken out so we can compare the performance of different models<br>
<img src="assets/train_golden_set.JPG"><br>
<img src="assets/train_test_split.JPG">


## Model training
The model training was done using Non Neutral-network approach as well as Neutral-network approach
### Sklearn/Lightgbm
- Configurable trainer using yaml.
- GridSearch and CrossValidation
- Saving of best model based on cross validation score
- Using of Sklearn pipeline to group Normalization(Robust Scaler) with model
- experiment tracking using Weights and bias
#### Results
The below image shows the average cross validation score for each type of model
<img src="assets/Best_model_scores.JPG">

#### Experiment Tracking
All the experiments are tracked using weights and bias so we can review the experiments later
<img src="assets/Logging_to_weights_bias.JPG">

### Deep learning
The deep learning model is trained on COLAB to take advantage of the GPU 
https://colab.research.google.com/drive/1f2ITJBrrAdgBOlJllKsWzYsBXjeDLbvP?usp=sharing

- logging using tensorboard
- FP 16 training uing grad scaler
- 5 fold cross validation

#### Results
The below image shows the training result and the use of tensorboard<br>
<img src="assets/deep_learning_metrics.JPG">
<img src="assets/tensorboard.JPG">

### LightGBM vs Deep learning PR curve
The below image shows the PR curve comparison between deep learning and Lightgbm
<img src="assets/PR_cure_nn_lgbm.png">

## Exporting to onnx
```python
model = FraudModel(args.n_features)
    model.load_state_dict(torch.load(args.model_weights, map_location=args.map_location))
    model.eval()

    x = torch.randn(1, args.n_features, requires_grad=True)
    x_numpy = x.detach().numpy()

    torch_output = model(x).detach().cpu().numpy()

    torch.onnx.export(model, x, args.output_fp, 
                      export_params=True, opset_version=10, 
                      do_constant_folding=True, input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input':{0 :'batchsize'}, 
                                    'output':{0: 'batchsize'}})

    # checking if the export is successful
    onnx_model = onnx.load(args.output_fp)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(args.output_fp)
    ort_inputs = {ort_session.get_inputs()[0].name: x_numpy}
    ort_outputs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(torch_output, ort_outputs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
```

### Improvements
Research and practice on how to convert tensorflow/pytorch model into tensorRT format<br>
TensorRT 8.2 contains new module Torch-TensorRT, TensorFlow-TensorRT for direct conversion however, operator avaliablity still remains a challenge<br>
source: https://developer.nvidia.com/blog/nvidia-announces-tensorrt-8-2-and-integrations-with-pytorch-and-tensorflow/


## Explainability using SHAP
As the prediction from fraud detection model will go to a case management team. It is important that we have some kind of explainability for the model prediction to gain some trust from the reviewers. Therefore, in this project, showcasing some explainability using SHAP

### Fraud sample
We can see from below image that feature 14 is the main reason for the fraud prediction
<img src="assets/shap_postive_case.JPG">

### Non-Fraud sample
We can see from below image that feature 14 is also the main reason for that sample not being a fraud case
<img src="assets/shap_negative_case.JPG">

## IMPROVEMENTS
There are many improvements that can be done in this project
1. Testing Downsampling and Upsampling of majority or minority class
2. Feature enrichment using polynomial features and feature crosses
3. Incorporating GNN features (user embeddings)
4. Perform better clustering and create different models for different clusters (eg new users/long time users)
# TNBC Classification using CNN

## Introduction

This Python script demonstrates the process of building, training, and evaluating a Convolutional Neural Network (CNN) model for Triple-Negative Breast Cancer (TNBC) classification using TensorFlow and Keras. The script utilizes the following libraries: NumPy, Pandas, TensorFlow, and scikit-learn.

## Dataset

The dataset used for TNBC classification is assumed to be stored in a CSV file named 'your_dataset.csv'. It should contain both features and labels, with features being the input data and labels representing the target variable (TNBC classification).

## Preprocessing

1. **Loading the Dataset**: The script loads the dataset using Pandas' `read_csv` function.
2. **Feature and Label Separation**: It separates the features (X) and labels (y) from the dataset.
3. **Splitting Data**: The dataset is split into training and testing sets using scikit-learn's `train_test_split` function. The testing set size is set to 20%, and a random state is specified for reproducibility.
4. **Feature Scaling**: Feature scaling is performed using scikit-learn's `StandardScaler` to normalize the feature values.

## Model Building

1. **Reshaping Input Features**: Since we're using a 1D CNN, the input features need to be reshaped to match the network's input requirements.
2. **Building the CNN Model**: The CNN model is constructed using TensorFlow and Keras. It consists of a 1D convolutional layer followed by max-pooling, flattening, and fully connected layers with ReLU activation. The output layer uses softmax activation for multi-class classification.

## Model Training

1. **Compiling the Model**: The model is compiled with sparse categorical cross-entropy loss and the Adam optimizer. Accuracy is chosen as the evaluation metric.
2. **Training the Model**: The compiled model is trained on the training set for 10 epochs with a batch size of 32. Validation data is provided to monitor the model's performance during training.

## Model Evaluation

1. **Model Evaluation**: After training, the model is evaluated on the test set using the `evaluate` method. Loss and accuracy metrics are computed and printed.

## Conclusion

This Python script provides a comprehensive demonstration of building and training a CNN model for TNBC classification. By following the documented steps, users can adapt the script to their specific dataset and classification task.

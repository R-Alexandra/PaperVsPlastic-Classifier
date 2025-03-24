# Paper Vs Plastic - Classifier
Through this project I explored Logistic Regression, k-NN, and Naive Bayes algorithms, analyzing their efficiency in handling real-world image variations, optimizing waste sorting automation

## Overview
This project focuses on binary image classification to distinguish between paper and plastic objects. The ability to differentiate between these materials is essential for applications in recycling and waste sorting. 

## Algorithms Explored
Three supervised learning algorithms were tested for classification:
- **Logistic Regression**: A simple and efficient model for binary classification.
- **k-Nearest Neighbors (k-NN)**: An instance-based learning algorithm.
- **Naïve Bayes**: A probabilistic model based on Bayes' theorem.

## Dataset
The dataset was compiled from multiple sources, including publicly available datasets and manually collected images. Images were labeled and structured into three main folders:
- **Training Set**: 560 images (280 paper, 280 plastic)
- **Validation Set**: 100 images (50 paper, 50 plastic)
- **Testing Set**: 80 images (40 paper, 40 plastic)

## Preprocessing
- Images were resized to **300x300 pixels**.
- Feature extraction included:
  - **Color Histogram (HSV)**: Extracted color distribution features.
  - **Local Binary Pattern (LBP)**: Extracted texture-based features.
- Standard scaling was applied for k-NN, while Naïve Bayes was used without scaling.

## Libraries Used
The following libraries were used for data processing, feature extraction, model training, and evaluation:

### Logistic Regression
- `os`: File and directory manipulation
- `cv2`: Image reading and processing
- `numpy`: Matrix-based data manipulation
- `sklearn.linear_model`: LogisticRegression (logistic regression model)
- `skimage.feature`: local_binary_pattern (texture feature extraction)
- `skimage.color`: Image color space conversion (BGR to HSV, BGR to grayscale)
- `skimage.io`: Image reading and resizing
- `pandas`: Dataframe manipulation and saving
- `sklearn.metrics`: classification_report, accuracy_score, confusion_matrix
- `seaborn`: Confusion matrix visualization
- `matplotlib.pyplot`: Plotting graphs
- `matplotlib.gridspec`: Organizing image visualization

### k-Nearest Neighbors (k-NN)
- `os`: File and directory manipulation
- `cv2 (OpenCV)`: Image reading and processing (resizing, color conversion)
- `numpy`: Matrix-based data manipulation
- `pandas`: Creating and saving results in CSV format
- `sklearn.neighbors`: KNeighborsClassifier (k-NN model training)
- `sklearn.metrics`:
  - `accuracy_score`: Accuracy calculation
  - `classification_report`: Model performance report
  - `confusion_matrix`: Confusion matrix creation
- `skimage.feature`: local_binary_pattern (LBP texture feature extraction)
- `sklearn.preprocessing`:
  - `LabelEncoder`: Encoding text labels into numerical values
  - `StandardScaler`: Feature normalization
- `matplotlib.pyplot`: Plotting graphs (classified images, confusion matrix)
- `seaborn`: Confusion matrix visualization

### Naïve Bayes
- `os`: File and directory manipulation
- `cv2 (OpenCV)`: Image reading and processing (resizing, color conversion)
- `numpy`: Matrix-based data manipulation
- `pandas`: Creating and saving results in CSV format
- `sklearn.naive_bayes`: MultinomialNB (Naïve Bayes classifier training)
- `sklearn.metrics`:
  - `accuracy_score`: Accuracy calculation
  - `classification_report`: Model performance report
  - `confusion_matrix`: Confusion matrix creation
- `skimage.feature`: local_binary_pattern (LBP texture feature extraction)
- `sklearn.preprocessing`: LabelEncoder (encoding text labels into numerical values)
- `matplotlib.pyplot`: Plotting graphs (classified images, confusion matrix)
- `seaborn`: Confusion matrix visualization

## Model Evaluation
Performance was assessed using accuracy, precision, recall, and F1-score. 

### Logistic Regression
- **Training Time**: 19 seconds
- **Validation Accuracy**: 80%
- **Testing Accuracy**: 78.75%

### k-Nearest Neighbors (k-NN)
- **Training Time**: 20 seconds
- **Validation Accuracy**: 75%
- **Testing Accuracy**: 75%

### Naïve Bayes
- **Training Time**: 17 seconds
- **Validation Accuracy**: 75%
- **Testing Accuracy**: 76%

## Conclusion
- **Logistic Regression** performed best, balancing efficiency and accuracy.
- **k-NN** provided competitive results but required careful tuning of parameters.
- **Naïve Bayes** was the fastest but relied on feature independence assumptions.

The project demonstrates that simple machine learning techniques can effectively classify materials based on image features, which is useful in automated waste management systems.

## Usage
1. Ensure the dataset is structured correctly in `Train/`, `Valid/`, and `Test/` directories.
2. Run the preprocessing script to extract features.
3. Train the desired model and evaluate performance.

For further improvements, deep learning approaches like Convolutional Neural Networks (CNNs) could be explored for enhanced accuracy.

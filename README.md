# House-Prices-Prediction-using-TFDF
# House Price Prediction using TensorFlow Decision Forests 🏠

A machine learning project using TensorFlow Decision Forests to predict house prices with Random Forest regression.

## 📊 Workflow Overview

```mermaid
graph TD
    A[Data Loading] -->|Preprocessing| B[Data Cleaning]
    B --> C[Feature Engineering]
    C --> D[Train/Test Split]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    F --> G[Predictions]
    G --> H[Submission]
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style D fill:#fff3e0
    style E fill:#fff3e0
    style F fill:#f9fbe7
    style G fill:#f1f8e9
    style H fill:#f1f8e9
```

## 🛠️ Requirements

```mermaid
graph LR
    A[TensorFlow] --> B[TF Decision Forests]
    C[Pandas] --> D[Data Processing]
    E[Seaborn] --> F[Visualization]
    G[Matplotlib] --> F
    H[NumPy] --> D
    
    style A fill:#f9f9f9
    style B fill:#f9f9f9
    style C fill:#f9f9f9
    style D fill:#f5f5f5
    style E fill:#f9f9f9
    style F fill:#f5f5f5
    style G fill:#f9f9f9
    style H fill:#f9f9f9
```

## 📝 Implementation Steps

### 1. Environment Setup

```python
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)
```

### 2. Data Pipeline

```mermaid
flowchart LR
    A[Raw Data] -->|Load| B[DataFrame]
    B -->|Clean| C[Remove ID]
    C -->|Analyze| D[Statistics]
    D -->|Split| E[Train/Test Sets]
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#e3f2fd
    style D fill:#e3f2fd
    style E fill:#e3f2fd
```

#### Key Data Processing Steps:
```python
# Remove ID column
dataset_df = dataset_df.drop('Id', axis=1)

# Data splitting
def split_dataset(dataset, test_ratio=0.30):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
```

### 3. Model Architecture

```mermaid
flowchart TD
    A[Input Data] --> B[Random Forest]
    B --> C{Training Process}
    C -->|Monitor| D[RMSE]
    C -->|Track| E[Feature Importance]
    C -->|Evaluate| F[Model Performance]
    
    style A fill:#fff3e0
    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#fff3e0
    style F fill:#fff3e0
```

#### Model Implementation:
```python
# Create and train model
rf = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
rf.compile(metrics=["mse"])
rf.fit(x=train_ds)

# Model inspection
inspector = rf.make_inspector()
logs = inspector.training_logs()
```

### 4. Visualization Pipeline

```mermaid
flowchart LR
    A[Sale Price Distribution] --> E[Visualizations]
    B[Feature Histograms] --> E
    C[Training Progress] --> E
    D[Feature Importance] --> E
    
    style A fill:#f3e5f5
    style B fill:#f3e5f5
    style C fill:#f3e5f5
    style D fill:#f3e5f5
    style E fill:#f3e5f5
```

### 5. Prediction & Submission

```mermaid
flowchart LR
    A[Test Data] -->|Preprocess| B[Model Input]
    B -->|Predict| C[Price Predictions]
    C -->|Format| D[Submission File]
    
    style A fill:#e8f5e9
    style B fill:#e8f5e9
    style C fill:#e8f5e9
    style D fill:#e8f5e9
```

#### Submission Code:
```python
# Generate predictions
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    test_data,
    task=tfdf.keras.Task.REGRESSION)

preds = rf.predict(test_ds)

# Create submission file
sample_submission_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
sample_submission_df['SalePrice'] = rf.predict(test_ds)
sample_submission_df.to_csv('/kaggle/working/submission.csv', index=False)
```

## 📈 Key Features

```mermaid
mindmap
    root((Model Features))
        Automatic preprocessing
            Missing value handling
            Feature scaling
        Model analysis
            Feature importance
            Training logs
        Visualization
            Distribution plots
            Feature correlations
        Evaluation
            MSE metrics
            RMSE tracking
```

## 🔍 Model Performance Monitoring

The following metrics are tracked during training:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Feature importance rankings
- Training progress logs

## 📊 Sample Visualizations

When you run the code, you'll see visualizations for:
1. Sale price distribution
2. Numerical feature histograms
3. Training progress (RMSE vs number of trees)
4. Feature importance rankings

## 📝 Notes

- All visualizations are generated using matplotlib and seaborn
- The model automatically handles missing values
- Feature importance analysis helps identify key price predictors
- Training progress can be monitored in real-time

## 🚀 Usage

1. Clone the repository
2. Install requirements
3. Run the notebooks in order
4. Check the generated visualizations
5. View the submission file

---
*This README is enhanced with Mermaid diagrams. Make sure your GitHub markdown viewer supports Mermaid syntax.*

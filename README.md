# ML Projects Practice Repository

This repository contains a collection of machine learning projects I've worked on for fun and learning. These projects come from various sources including Coursera courses, Kaggle competitions, and personal exploration.

## Projects

### 1. [Loan Approval Prediction](./loan_approval_prediction/)
- **Objective**: Predict loan approval status using applicant features
- **Data Source**: Kaggle
- **Models**: Logistic Regression, XGBoost
- **Best Accuracy**: 83.17% (XGBoost)

### 2. [Autism Prediction](./autism_prediction/)
- **Objective**: Predict Autism Spectrum Disorder (ASD) based on behavioral and demographic features
- **Data Source**: Kaggle
- **Models**: SVC, Logistic Regression, LightGBM, XGBoost
- **Best ROC-AUC**: 0.893 (SVC with SMOTE)

### 3. [Customer Segmentation](./customer_segmentation/)
- **Objective**: Segment customers into distinct groups using K-Means clustering
- **Data Source**: Marketing Analytics Dataset
- **Method**: Unsupervised learning with K-Means
- **Optimal Clusters**: 4 (determined by Elbow method and Silhouette score)

### 4. [Doge Price Prediction](./doge_price_prediction/)
- **Objective**: Predict Dogecoin closing prices using historical price features
- **Data Source**: Yahoo Finance / Kaggle
- **Models**: Random Forest Regressor, Linear Regression
- **Best R² Score**: 99.61% (Linear Regression with scaled features)

### 5. [PyTorch Binary Classifier Tutorial](./pytorch_binary_classifier/)
- **Objective**: Learn to build a binary classifier using PyTorch
- **Source**: Educational Tutorial
- **Framework**: PyTorch
- **Dataset**: Synthetic data (for demonstration)

### 6. [LLM Pipelines](./LLM/)
- **Source**: Transformers Packt Course
- **Objective**: Explore different Hugging Face Transformers pipelines
- **Notebooks**:
  - `01_sentiment_analysis.ipynb` - Sentiment analysis on text
  - `02_feature_extraction.ipynb` - Extract embeddings for semantic search
  - `03_text_generation.ipynb` - Generate text with language models
  - `04_fill_mask.ipynb` - Masked language modeling
  - `05_named_entity_recognition.ipynb` - Identify entities in text

## Repository Structure

```
ml-problems/
├── README.md
├── requirements.txt
├── loan_approval_prediction/
│   ├── loan_approval_prediction.ipynb
│   └── LoanApprovalPrediction.csv
├── autism_prediction/
│   ├── autism_prediction.ipynb
│   └── autism_prediction.csv
├── customer_segmentation/
│   ├── customer_segmentation.ipynb
│   └── customer_segmentation.csv
├── doge_price_prediction/
│   ├── doge_price_prediction.ipynb
│   └── DOGE-USD.csv
├── pytorch_binary_classifier/
│   └── pytorch_binary_classifier_tutorial.py
├── LLM/
│   ├── 01_sentiment_analysis.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_text_generation.ipynb
│   ├── 04_fill_mask.ipynb
│   ├── 05_named_entity_recognition.ipynb
│   ├── Tweets.csv
│   └── bbc_text_cls.csv
└── others/
    └── (practice and exploratory files)
```

## Getting Started

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Navigate to any project folder and run the notebooks

## Requirements

See [requirements.txt](./requirements.txt) for the full list of dependencies.

## Note

These are personal practice projects completed for learning purposes. They represent my journey exploring different machine learning techniques and datasets from various sources including Coursera courses, Kaggle competitions, and personal curiosity-driven exploration.


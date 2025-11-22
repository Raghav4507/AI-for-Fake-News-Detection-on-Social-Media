# 📰 Overview

This project focuses on detecting fake news on social media using advanced NLP and deep learning models, specifically BERT and BiLSTM. Fake news spreads rapidly across online platforms, influencing public opinion and creating misinformation. Our goal is to build a robust AI system capable of identifying fake or real news reliably.

This repository contains the complete workflow including dataset preprocessing, model training, evaluation, visualization, and prediction.

# 🎯 Objectives

Build a supervised learning system for detecting fake news.

Compare the performance of BERT and BiLSTM architectures.

Work with binary classification (real/fake) as well as 6-class classification datasets.

Evaluate accuracy, loss, generalization, and prediction capability.

# 🧠 Models Used
## 1. BERT (Bidirectional Encoder Representations from Transformers)

Understands context from both directions.

Pretrained on large corpora.

Achieved >99% accuracy on 2-class dataset.

Overfits on 6-class dataset → requires regularization and tuning.

## 2. BiLSTM (Bidirectional Long Short-Term Memory)

Reads text from both forward and backward directions.

Works well with sequential patterns.

Reaches ~99% accuracy on 2-class dataset with faster training.

# 📚 Datasets
## 1. 2-Class Fake News Dataset (Hugging Face)

Labels: Real, Fake

Files: train.tsv, test.tsv, valid.tsv

Size: 44,267 samples

Features: title, text, subject, date, label

Clean and well-structured without missing values.

## 2. 6-Class Politifact Dataset (Kaggle)

Labels: true, mostly-true, half-true, barely-true, false, pants-on-fire

Size: 12,836 samples

High-quality, manually verified data.

# 🛠️ Tech Stack

Programming: Python

Libraries:

Pandas, NumPy, NLTK

Scikit-learn

PyTorch (BiLSTM)

HuggingFace Transformers (BERT)

Visualization: Matplotlib, Seaborn, WordCloud

# 🧹 Preprocessing Steps

Remove special characters & stopwords

Tokenization (model-specific)

Word embeddings (for BiLSTM)

Padding and attention masks (for BERT)

Train-validation-test split

Handling imbalanced classes

# 🚀 Training & Evaluation
## BERT – 2 Class

3 epochs (~20 min each)

Achieved 99%+ accuracy

Fast convergence due to pretrained weights

## BiLSTM – 2 Class

5 epochs (~1 min each)

Achieved 99%+ accuracy

## BERT – 6 Class

Training accuracy → 99.7%

Validation accuracy → 25%

Severe overfitting detected

## BiLSTM – 6 Class

Training accuracy → 63%

Validation accuracy → 21–26%

Requires regularization and class balancing

# 📈 Results Summary
## Model	Dataset	Accuracy	Notes
BERT	2-Class	≈99%	Best performer
BiLSTM	2-Class	≈99%	Slightly lower but fast
BERT	6-Class	~24% (val)	Overfits
BiLSTM	6-Class	~25% (val)	Overfits, stable

## Visualizations show:

BERT converges faster

BiLSTM improves steadily

Both models struggle with 6-class generalization

# 🔍 Sample Predictions

Input Example:

“The earth is flat and NASA is hiding it.”

Model Output:

Fake / Pants-on-Fire (depending on dataset)

Both models show strong prediction consistency on unseen inputs.

# 🧭 Future Improvements

Add multilingual support

Apply early stopping and regularization

Deploy as a browser plugin or web API

Improve 6-class dataset performance

Continuous dataset updates to track new misinformation trends

# 📝 Summary

This project demonstrates how transformer-based and RNN-based deep learning models can be used to identify fake news effectively. With extremely high accuracy on binary classification and valuable insights from multi-class classification experiments, this work lays a strong foundation for real-world fake news detection systems.

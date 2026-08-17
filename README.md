# 🤖 MLPilot: Autonomous AI/ML Engineer Agent

## Overview

MLPilot is an agentic machine learning system designed to automate the end-to-end machine learning workflow through a modular, stateful agent architecture.

Built using LangGraph, MLPilot coordinates specialized agents responsible for dataset ingestion, data analysis, preprocessing, train-test splitting, model selection, model training, evaluation, and automated reporting.

The long-term goal of MLPilot is to function as an autonomous AI/ML Engineer that can analyze a dataset, determine the appropriate machine learning strategy, experiment with suitable models, evaluate their performance, and identify the best-performing solution with minimal human intervention.

---

## Features

### 📂 Automated Dataset Loading

* Load datasets directly from CSV files.
* Validate whether the dataset can be loaded successfully.
* Detect empty or invalid datasets.
* Store the loaded dataset in the shared LangGraph state.
* Automatically extract dataset dimensions.

### 🔍 Intelligent Dataset Analysis

* Analyze dataset structure automatically.
* Identify numerical and categorical features.
* Detect columns containing missing values.
* Calculate dataset dimensions.
* Pass analysis results to subsequent agents through shared state.

### 🧠 Automated Problem Detection

* Determine whether the machine learning task is classification or regression.
* Analyze target-column characteristics.
* Automatically identify the appropriate ML problem type.

### ⚙️ Automated Data Preprocessing

* Handle missing numerical values.
* Handle missing categorical values.
* Encode categorical features.
* Scale numerical features.
* Build reusable preprocessing pipelines.

### ✂️ Train-Test Split

* Automatically separate training and testing data.
* Support stratified splitting for classification tasks.
* Prevent test-set information from influencing model training.

### 🤖 Intelligent Model Selection

* Analyze dataset characteristics.
* Identify suitable machine learning algorithms.
* Select multiple candidate models for experimentation.
* Compare candidate models based on actual validation performance.

### 🏋️ Automated Model Training

* Train selected machine learning models automatically.
* Integrate preprocessing and model training into a unified pipeline.
* Support multiple classification and regression algorithms.

### 📊 Model Evaluation

* Evaluate trained models using appropriate performance metrics.
* Compare candidate models using cross-validation.
* Select the best-performing model based on objective evaluation results.
* Evaluate the final model on an unseen test dataset.

### 📑 Automated ML Reporting

Generate structured reports containing:

* Dataset information
* Detected problem type
* Selected models
* Cross-validation performance
* Final test performance
* Best-performing model

---

## System Architecture

```text
User Dataset
     │
     ▼
Data Upload Agent
     │
     ▼
Data Analysis Agent
     │
     ▼
Problem Detection Agent
     │
     ▼
Preprocessing Agent
     │
     ▼
Train-Test Split Agent
     │
     ▼
Model Selection Agent
     │
     ▼
Model Experimentation
     │
     ├──────────────┬──────────────┐
     ▼              ▼              ▼
 Model A         Model B         Model C
     │              │              │
     └──────────────┼──────────────┘
                    ▼
             Evaluation Agent
                    │
                    ▼
             Best Model Selection
                    │
                    ▼
              Final Training
                    │
                    ▼
              Report Generation

# 🌍 New-Worlds — AI-Powered Exoplanet Intelligence Platform

**New-Worlds** is an AI-powered platform that combines Machine Learning, Deep Learning, Quantum Machine Learning and Large Language Models to support exoplanet analysis and telescope observation scheduling.

The project integrates multiple AI approaches through a microservices architecture and provides an interactive dashboard for predictions, model comparisons and observation scheduling.

---

## 🚀 Project Overview

The platform explores four major AI technologies:

- **Machine Learning (ML):** Exoplanet classification experiments using Logistic Regression, Random Forest and XGBoost.
- **Deep Learning (DL):** Transit signal classification using CNN and LSTM architectures.
- **Quantum Machine Learning (QML):** Telescope observation scheduling experiments using QAOA and VQE.
- **Large Language Models (LLM):** AI-powered interpretation of benchmark results and telescope scheduling assistance.

## 🏗️ System Architecture

```text
             React + Vite Frontend
                      |
                FastAPI Backend
                      |
       +--------------+--------------+
       |              |              |
   ML Service     DL Service     QML Service
       |              |              |
   Classification  CNN / LSTM    QAOA / VQE
                      |
                 LLM Service
                      |
              Groq GPT-OSS-20B
```

The application follows a microservices architecture. Each service handles a separate responsibility and communicates through APIs.

## 🧠 Technology Stack

| Category | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn, Random Forest, XGBoost |
| Deep Learning | CNN, LSTM, PyTorch |
| Quantum Computing | PennyLane, QAOA, VQE |
| LLM Integration | Groq API, GPT-OSS-20B |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Containerisation | Docker, Docker Compose |
| Version Control | Git, GitHub |

## 📁 Project Structure

```text
New-Worlds/
│
├── backend/
│   └── main.py
│
├── frontend/
│   └── src/
│
├── services/
│   ├── ml-service/
│   ├── dl-service/
│   ├── qml-service/
│   └── llm-benchmark-service/
│
├── data/
│   └── processed/
│
├── notebooks/
├── tests/
├── k8s/
│
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

## 🤖 Machine Learning

The ML component compares multiple classification algorithms:

- Logistic Regression
- Random Forest
- XGBoost
- Dummy Classifier baseline

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Balanced Accuracy

### Recorded Experimental Results

| Model | Accuracy | F1-score |
|---|---:|---:|
| Dummy Baseline | 98.72% | 0.00% |
| Logistic Regression | 95.49% | 36.54% |
| Random Forest | 99.63% | 85.90% |
| XGBoost | 99.61% | 85.89% |

The Dummy Baseline demonstrates why accuracy alone can be misleading when working with imbalanced datasets.

## 🧬 Deep Learning

The DL service explores transit signal classification using:

- Convolutional Neural Networks (CNN)
- Long Short-Term Memory (LSTM)

Recorded test results:

| Metric | CNN | LSTM |
|---|---:|---:|
| Accuracy | 69.75% | 63.95% |
| Precision | 67.24% | 63.26% |
| Recall | 99.26% | 98.94% |
| F1-score | 80.17% | 77.18% |

## ⚛️ Quantum Machine Learning

The QML component explores telescope observation scheduling using:

- Quantum Approximate Optimization Algorithm (QAOA)
- Variational Quantum Eigensolver (VQE)
- Classical exhaustive-search comparison

The experiments use PennyLane's classical quantum simulator.

| Algorithm | Mean Priority | Optimality Gap |
|---|---:|---:|
| Classical | 158.75 | 0% |
| QAOA | 158.75 | 0% |
| VQE | 158.75 | 0% |

These are small experimental scheduling scenarios. No quantum advantage has been established.

## 🧠 LLM Integration

The LLM service integrates Groq's GPT-OSS-20B model to:

- Interpret ML, DL and QML benchmark results.
- Generate model comparisons.
- Provide recommendations.
- Summarise experimental limitations.
- Assist with telescope observation scheduling.

The LLM interprets experimental outputs; it does not train or independently validate the underlying models.

## 📊 Experiment Tracking and Data Versioning

**MLflow**

Used for tracking machine learning experiments, including parameters, evaluation metrics and artifacts.

**DVC**

Initialised for data and pipeline version management. Google Drive was selected as the intended remote storage backend; remote configuration and data synchronisation remain pending.

## 🐳 Running the Project

### Prerequisites

- Python
- Docker Desktop
- Git
- Node.js (if running the frontend separately)

### Clone the repository

```bash
git clone https://github.com/jayanth-3345/New-Worlds.git

cd New-Worlds
```

### Start the application

Make sure Docker Desktop is running.

```bash
docker compose up -d --build
```

### Check running services

```bash
docker compose ps
```

### Access the dashboard

http://localhost:5173

### Backend API documentation

http://localhost:8000/docs

## 🔌 Service Ports

| Service | Port |
|---|---:|
| Frontend | 5173 |
| Backend | 8000 |
| ML Service | 8002 |
| DL Service | 8003 |
| QML Service | 8004 |
| LLM Service | 8005 |

## ⚠️ Research Limitations

- Model performance depends on the quality and characteristics of the datasets.
- High classification accuracy does not necessarily indicate strong minority-class performance.
- Deep learning models may produce false positives.
- QML experiments use classical simulation rather than physical quantum hardware.
- The experiments do not establish quantum advantage.
- Computational predictions cannot prove the existence of life or planetary habitability.

## 🎯 Project Objective

New-Worlds aims to explore how different AI paradigms can be integrated into an extensible platform for exoplanet analysis and telescope observation scheduling.

---

**Project:** New-Worlds  
**Repository:** [github.com/jayanth-3345/New-Worlds](https://github.com/jayanth-3345/New-Worlds)  
**Domain:** Artificial Intelligence · Astronomy · Exoplanet Analysis

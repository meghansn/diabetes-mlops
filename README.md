# Diabetes Progression Predictor: End-to-End MLOps Microservice

A production-ready machine learning microservice that predicts diabetes disease progression one year out based on baseline clinical variables. 

This project demonstrates a complete MLOps deployment pipeline—moving from a raw Scikit-Learn script to a highly validated, containerized FastAPI web application.

---

## 🏗️ Architecture Overview

The pipeline is structured across four distinct operational layers:

1. **The Model Factory (`train.py`):** Trains a regularized Ridge Regression model on 442 medical patient profiles, optimizing for an $R^2$ baseline metric. The finalized mathematical weights are serialized into a binary artifact (`model.joblib`).
2. **The Validation & Serving Layer (`app.py`):** A high-performance **FastAPI** web server. It uses **Pydantic** schemas to act as a strict data gatekeeper, ensuring all incoming JSON inferences match clinical data types perfectly before hitting the model.
3. **The Blueprint (`Dockerfile`):** A containerization recipe that builds an isolated, minimal Linux-based Python 3.13 environment.
4. **The Container (`Docker Image`):** Unifies the serving layer and serialized model weights into a platform-agnostic, cloud-ready microservice.

> 💡 **The MLOps Analogy:** Think of a standard Python script like a **Word Document (.docx)**—it looks perfect on your machine, but the formatting, paths, and dependencies break the moment you open it on another system. Containerizing this application with Docker turns it into a **PDF**—an un-mess-withable, isolated artifact that renders and runs identically on a local MacBook Air, an AWS instance, or Google Cloud Vertex AI.

---

## 🛠️ Local Deployment Guide

### Prerequisites
* Python 3.13
* Docker Desktop

### 1. Model Training & Serialization
To execute the training pipeline and generate the core `model.joblib` artifact:
```bash
# Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run training script (Baseline Test Set R² ~ 0.4192)
python train.py
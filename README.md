# Medical Insurance Cost Prediction - MLOps Project

This project is an end-to-end **MLOps pipeline** for predicting medical insurance costs based on user attributes such as age, BMI, region, smoking status, etc. The system includes data preprocessing, model training (Random Forest & Linear Regression), Flask-based API deployment, CI/CD automation with GitHub Actions, Docker containerization, and optional VM/Kubernetes deployment.

---

## Features

- Machine learning model (Random Forest Regressor)
- Flask web API with HTML frontend
- Retraining pipeline (`retrain.py`) with Linear Regression
- Docker containerization
- CI/CD via GitHub Actions
- Deployment-ready for Google VM, Docker, or Kubernetes

---

## Project Components

- `train.py`: Trains a Random Forest Regressor and saves the model
- `retrain.py`: Retrains using Linear Regression for fallback/update
- `main.py`: Flask app exposing web UI and API
- `test_app.py`: Unit tests for the Flask API
- `requirements.txt`: Project dependencies
- `.github/workflows/`: CI, CD, and retrain GitHub Actions
- `Dockerfile`: Container definition for building and deploying the app

---

## How to Run

### Option 1: Run Locally

```bash
pip install -r requirements.txt
python app/train.py
python app/main.py

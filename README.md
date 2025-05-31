# Insurance Cost Prediction - MLOps Project

This project predicts **medical insurance costs** based on user attributes such as age, BMI, smoking status, and region. It follows an end-to-end **MLOps workflow** using:

- Machine Learning: `RandomForestRegressor`
- Dockerized Flask API
- CI/CD with GitHub Actions
- Kubernetes Deployment (Minikube)
- Model Retraining Pipeline

---

## Features

- ML model to predict insurance charges
- REST API with Flask
- Dockerized application
- CI/CD GitHub Actions to:
  - Build & push image to DockerHub
  - Deploy to Minikube cluster
  - Retrain model on new data

---

## ML Model Info

- Algorithm: `RandomForestRegressor` from `scikit-learn`
- Trained on: `medical_insurance.csv`
- Inputs:
  - Age
  - Sex
  - BMI
  - Children
  - Smoker
  - Region
- Output:
  - Insurance Charges (`charges`)

---

## Docker Setup

###  Build Image
```bash
docker build -t insurance-cost-app .

## Automated Loan Approval & Credit Risk Model

## Overview

This project is an end-to-end Automated Loan Approval and Credit Risk Prediction system designed using real-world industry practices. It demonstrates how a machine learning model can be trained, tested, containerized, and deployed using CI/CD pipelines.

The goal of this project is to assess a customer’s credit risk and support automated loan approval decisions with a scalable and production-ready architecture.

## Key Features

1. End-to-end Machine Learning pipeline
2. Data preprocessing and feature engineering
3. Model training and evaluation
4. Automated testing using Pytest
5. Dockerized application for portability
6. CI/CD pipeline using GitHub Actions
7. Docker image pushed automatically to Docker Hub
8. Clean, modular, and production-ready project structure

## Tech Stack
1. Programming Language: Python
2. Machine Learning: Scikit-learn
3. Testing: Pytest
4. MLOps & CI/CD: GitHub Actions
5. Containerization: Docker
6. Version Control: Git & GitHub
7. Deployment Ready: Docker Hub

## CI/CD Workflow (Industry Style)
1. This project follows a complete CI/CD lifecycle:
2. Continuous Integration (CI)
3. Triggered on every push to main

## Steps:
1. Set up Python environment
2. Install dependencies
3. Run automated tests (Pytest)
4. Fail fast if tests break
5. Continuous Deployment (CD)
6. Builds Docker image automatically
7. Logs into Docker Hub using GitHub Secrets
8. Pushes the latest Docker image to Docker Hub

✅ Result: Every successful code push produces a tested and deployable Docker image.

## Docker Image
- The application is containerized and available on Docker Hub:

- docker pull rammahto1/credit-risk1:latest

## Run the container
- docker pull rammahto1/credit-risk1:latest

## How to Run Locally

1. How to Run Locally

git clone https://github.com/RamMahto1/Automated_Loan_Approval_Risk_Model.git
cd Automated_Loan_Approval_Risk_Model

2. Create virtual environment 
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

3. Install dependencies
- pip install -r requirements.txt

4. Run tests
 - pytest

5. Run application
- python app.py

## Learning Outcome
- Ram Mahto
- Aspiring Data Scientist | MLOps Enthusiast


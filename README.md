# Automated Loan Approval & Credit Risk Model

## Overview

This project is an end-to-end Automated Loan Approval and Credit Risk Prediction system built using real-world, industry-aligned practices. It demonstrates how a machine learning model can be trained, tested, containerized, and deployed using a complete CI/CD pipeline.

The objective of this project is to assess customer credit risk and support automated loan approval decisions using a scalable, production-ready architecture.

---

## Key Features

- End-to-end machine learning pipeline
- Data preprocessing and feature engineering
- Model training and evaluation
- Automated testing using Pytest
- Dockerized application for portability
- CI/CD pipeline implemented using GitHub Actions
- Automatic Docker image build and push to Docker Hub
- Clean, modular, and industry-ready project structure

---

## Tech Stack

- Programming Language: Python  
- Machine Learning: Scikit-learn  
- Testing: Pytest  
- CI/CD & MLOps: GitHub Actions  
- Containerization: Docker  
- Version Control: Git & GitHub  
- Deployment: Docker Hub  

---

## CI/CD Workflow (Industry Standard)

This project follows a complete CI/CD lifecycle.

### Continuous Integration (CI)
Triggered on every push to the `main` branch:
1. Set up Python environment
2. Install dependencies
3. Run automated unit tests using Pytest
4. Fail fast if any test fails

### Continuous Deployment (CD)
After successful CI:
1. Build Docker image
2. Authenticate to Docker Hub using GitHub Secrets
3. Push the latest Docker image to Docker Hub

## Result: Every successful code push produces a tested and deployable Docker image.

---

## Docker Image

The application is containerized and available on Docker Hub:


docker pull rammahto1/credit-risk1:latest

# Run the Docker Container
docker run -p 5000:5000 rammahto1/credit-risk1:latest

# How to Run locally
1. Clone the Repositry
- git clone https://github.com/RamMahto1/Automated_Loan_Approval_Risk_Model.git
- cd Automated_Loan_Approval_Risk_Model

2. Create and Activate Virtual Enviroment
- Python -m venv venv
- Activate the enviroment

Window
- venv/scripts/activate
Linux/macOS
- source venv/bin/activate

3. Install Dependencies
- pip install -r requirement.txt

4. Run Test
- pytest

5. Run the Application
 - python app.py

## Learning Outcomes
- Through this project, I gained hands-on experience with:
- Building end-to-end machine learning pipelines
- Writing industry-standard unit tests
- Implementing CI/CD pipelines using GitHub Actions
- Containerizing ML applications using Docker
- Applying production-ready MLOps practices



# Author 
- Ram Mahto
- Aspiring Data Scientist/MLOps Enthusiast


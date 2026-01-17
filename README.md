# 1. Software Characteristics

## a. Short Name

**MSFT Predictor**

## b. Full Name

**Microsoft Stock Price Prediction System**

## c. Summary Description and Objectives

The Microsoft Stock Price Prediction System is a machine learning-based application designed to forecast the next trading day's closing price of Microsoft (MSFT) stock. The system integrates historical market data, technical indicators, and multiple regression models to generate predictions presented through an interactive web interface.

The primary objectives of the project are to demonstrate the end-to-end machine learning workflow, including data acquisition, feature engineering, model training, evaluation, and deployment. The system is intended for educational and demonstrative purposes only and does not constitute financial advice.

---

# 2. Copyright

## a. Authors

* Project Author(s): *Olha Babicheva*, *Agata Jabłońska*, *Anna Saldat*.

## b. Licensing Terms

The software developed by the group is distributed under the **MIT License**.

---

# 3. Requirements Specification

## Priority Legend

| Priority         | Meaning                                                                             |
| ---------------- | ----------------------------------------------------------------------------------- |
| **1 - Mandatory**| Mandatory requirement. The system is not considered correct or complete without it. |
| **2 - Useful**   | Important but not critical requirement. Improves functionality or usability.        |
| **3 - Optional** | Nice-to-have requirement. Not essential for core operation.                         |

---

## a. Grouped Requirements List

### Module: Data Preparation

| ID     | Name                | Description                                           | Priority | Category       | Status |
| ------ | ------------------- | ----------------------------------------------------- | -------- | -------------- | ------ |
| FR-01  | Data Download       | Fetch historical MSFT stock data from Yahoo Finance   | 1        | Functional     | Met    |
| FR-02  | Feature Engineering | Calculate 10 technical indicators                     | 1        | Functional     | Met    |
| FR-03  | Data Accuracy       | Ensure correctness and consistency of downloaded data | 1        | Functional     | Met    |

---

### Module: Model Training

| ID     | Name             | Description                                       | Priority | Category       | Status |
| ------ | ---------------- | ------------------------------------------------- | -------- | -------------- | ------ |
| FR-04  | Model Training   | Train regression models (LR, RF, SVR, MLP)        | 1        | Functional     | Met    |
| FR-05  | Model Evaluation | Evaluate models using standard regression metrics | 2        | Functional     | Met    |
| FR-06  | Performance      | Training completes within reasonable time (< 1h)  | 2        | Functional     | Met    |

---

### Module: Web Application

| ID     | Name                   | Description                                                       | Priority | Category       | Status      |
| ------ | ---------------------- | ----------------------------------------------------------------- | -------- | -------------- | ----------- |
| FR-07  | Prediction Display     | Display next-day stock price prediction                           | 1        | Functional     | Met         |
| FR-08  | Live Data Fetching     | Fetch latest market data for predictions                          | 1        | Functional     | Met         |
| NFR-01 | Usability              | Provide a clear and intuitive user interface (display: market status, volatility range, predicted next close) | 2        | Non-functional | Met         |
| FR-09  | Multilingual Interface | Support additional interface languages (Polish, Spanish, Chinese) | 3        | Functional     | Not met     |

---

### Notes on Compliance

* All **Priority 1 (Required)** requirements were implemented and verified.
* All **Priority 2 (Useful)** requirements were implemented and tested successfully.
* There is one **Priority 3 (Optional)** requirement defined for this version of the system (FR-09: Multilingual Interface), which was not implemented.

---

# 4. System / Software Architecture

## a. Development Architecture (Technology Stack)

| Technology         | Purpose                                  | Version       |
| ------------------ | ---------------------------------------- | ------------- |
| Python             | Primary programming language             | **>= 3.10**   |
| Visual Studio Code | Source code editing and debugging        | **>= 1.108**  |
| GitHub             | Version control and collaboration        | Not specified |
| GitHub Actions     | CI/CD automation                         | Not specified |
| pandas             | Data preparation and feature engineering | **>= 2.3.2**  |
| numpy              | Numerical and statistical computations   | **>= 2.1.2**  |
| scikit-learn       | Machine learning model development       | **>= 1.7.2**  |
| yfinance           | Market data access during development    | **>= 0.2.66** |
| streamlit          | Development and testing of web interface | **>= 1.52.2** |
| joblib             | Model serialization and persistence      | **>= 1.5.2**  |



## b. Runtime Architecture (Execution Environment)

| Technology   | Purpose                                    | Version       |
| ------------ | ------------------------------------------ | ------------- |
| Python       | Application runtime environment            | **>= 3.10**   |
| pandas       | Data manipulation and time-series analysis | **>= 2.3.2**  |
| numpy        | Numerical computations                     | **>= 2.1.2**  |
| scikit-learn | Machine learning model execution           | **>= 1.7.2**  |
| yfinance     | Retrieval of financial market data         | **>= 0.2.66** |
| streamlit    | Web-based user interface and visualization | **>= 1.52.2** |
| joblib       | Loading trained models                     | **>= 1.5.2**  |

## c. Presentation of Used Technologies

The system leverages Python as the primary development language, supported by data science libraries such as pandas and NumPy for analysis and preprocessing. Machine learning functionality is implemented using scikit-learn, while trained models and scaler are serialized with joblib. The user interface is built using Streamlit, enabling deployment of an interactive web-based dashboard. GitHub is used for version control and collaborative development, with Visual Studio Code serving as the main development environment.

---

# 5. Testing

## a. Test Scenarios

1. **Data Download Test**

   * Verify that historical MSFT stock data is correctly downloaded from Yahoo Finance.

2. **Feature Engineering Test**

   * Confirm that technical indicators are computed without errors and stored correctly.

3. **Model Training Test**

   * Ensure that all regression models train successfully and produce output files.

4. **Web Application Test**

   * Validate that the Streamlit application launches correctly and displays predictions.

## b. Test Execution Report

All defined test scenarios were executed successfully. Data preparation scripts completed without errors, models were trained and saved correctly, and the Streamlit dashboard displayed valid predictions based on the trained models. No critical defects were identified during testing. Minor usability improvements were noted for future iterations.

---

# 6. CI/CD and Automation (GitHub Actions)

## a. Workflow Description

The project includes an automated Continuous Integration (CI) workflow defined in the `main.yaml` file. This workflow ensures that data preparation and model training steps are automatically executed on every relevant change to the repository, improving code reliability and reproducibility.

The workflow is triggered on:

* Push events to the `main` branch
* Pull requests targeting the `main` branch
* Manual execution via `workflow_dispatch`

## b. Workflow Configuration Overview

**Workflow name:** ML Workflow

**Permissions:**

* Full write permissions (`write-all`) to allow artifact handling and repository interactions if extended in the future.

## c. Automated Job Description

### Job: `build`

| Step | Name                 | Description                                                          |
| ---- | -------------------- | -------------------------------------------------------------------- |
| 1    | Checkout Repository  | Fetches the repository contents, including Git LFS files             |
| 2    | Install Packages     | Upgrades `pip` and installs all dependencies from `requirements.txt` |
| 3    | Run Data Preparation | Executes `data_preparation.py` to download and preprocess stock data |
| 4    | Run Model Training   | Executes `training.py` to train machine learning models              |

**Execution Environment:**

* Operating System: `ubuntu-latest`
* Python environment provided by GitHub-hosted runners

## d. Role in System Architecture

The `main.yaml` workflow is part of the **development and quality assurance architecture**. It automates critical pipeline steps and ensures that:

* The project remains buildable after each change
* Data preparation and training scripts execute without errors
* Core machine learning functionality is continuously validated

## e. Relation to Testing

The CI workflow complements manual test scenarios by automatically executing integration-level tests:

* Verifies end-to-end execution of the data and training pipeline
* Detects runtime errors early during development
* Supports consistent experimentation and collaboration across contributors

---

# 7. Running the Project on a Local Computer

This section describes the procedure required to run the Microsoft Stock Price Prediction System on a local machine.
All steps must be executed **in the specified order** to ensure correct operation of the system.

## 7.1. Prerequisites

Before running the project, ensure that the following requirements are met:

* A computer with Windows or Linux
* Python **version ≥ 3.10**
* Internet connection (required for downloading market data)
* Command-line interface (Terminal, PowerShell, or equivalent)

## 7.2. Installing Dependencies

Navigate to the project's root directory and install all required Python packages using the provided dependency list:

```bash
pip install -r requirements.txt
```

This command installs all libraries necessary for data processing, model training, and running the web application.

## 7.3. Data Preparation

Run the data preparation script to download historical Microsoft (MSFT) stock data and generate the required datasets:

```bash
python data_preparation.py
```

This step performs:

* Retrieval of historical stock prices from Yahoo Finance
* Calculation of technical indicators
* Creation of training and testing datasets

## 7.4. Model Training

After the data preparation step is complete, train the machine learning models by executing:

```bash
python training.py
```

This process:

* Trains multiple regression models
* Evaluates model performance
* Saves the trained models into a file named `models_bundle.joblib`

The generated model file is required for running the web application.

## 7.5. Launching the Web Application

Once the trained model file is available, start the interactive Streamlit dashboard with the following command:

```bash
streamlit run app.py
```

After execution, the application will open automatically in a web browser, allowing users to view live market data and next-day stock price predictions.

<img width="100%" height="100%" alt="image" src="https://github.com/user-attachments/assets/512ea433-2041-4f5d-b208-7bf80d6f6f1d" />

## 7.6. Verification of Correct Execution

The system is considered to be running correctly if:

* No errors occur during script execution
* The Streamlit dashboard loads successfully
* Stock price predictions are displayed in the user interface

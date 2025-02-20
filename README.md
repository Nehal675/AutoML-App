# AutoML Web App with PyCaret 

## Overview

This project is a Streamlit-based web application designed to simplify machine learning experimentation using PyCaret. It provides a user-friendly interface for loading datasets, preprocessing data, running automated machine learning (AutoML) workflows, comparing models, and exporting the best-performing model for future use.

## Features

• Upload CSV datasets for quick analysis.
•Automated data preprocessing, including handling missing values, removing constant and high-cardinality columns.
•Preprocessing step and ensuring categorical consistency.
•Exploratory Data Analysis (EDA) with summary statistics and visualizations.
•AutoML with PyCaret, supporting both classification and regression tasks.
•Model comparison to find the most effective algorithm.
•Download trained model for deployment.


## Installation

1. Clone the Repository:
   git clone https://github.com/Nehal675/AutoML-App.git
   cd AutoML-App
2. Install Dependencies:
   pip install -r requirements.txt

## Usage

Run the Streamlit App:
   streamlit run app.py

## Workflow

1. Upload a CSV dataset.
2. Data Cleaning to handle missing values and remove unnecessary columns .
3. Data Preprocessing to ensure categorical consistency.
4. Select the target variable for classification or regression.
5. Perform EDA to explore the dataset.
6. Train models .
7. Download the best model, saved in the models directory, for later use.

## Project Structure

📂 AutoML-App/
│── 📜 app.py                 # Main Streamlit App (Handles UI)
│── 📜 pipeline.py            # ML Pipeline (Preprocessing, Model Training, EDA & Visualization)
│── 📜 utils.py               # Utility Functions (File Handling)
│── 📜 requirements.txt       # Dependencies
│── 📜 README.md              # Project Documentation
├── 📂 models/                # Directory for saved trained models
│       ├── best_model.pkl    # Best model saved after training


## Dependencies

- Python 
- Streamlit
- Pandas, NumPy
- Seaborn, Matplotlib
- PyCaret
- Scikit-learn

## Contributing

Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request.

## License

This project is open-source under the MIT License.

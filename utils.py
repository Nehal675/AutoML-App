import pandas as pd

def load_data(uploaded_file):
    """Loads and returns a CSV file as a DataFrame."""
    return pd.read_csv(uploaded_file)

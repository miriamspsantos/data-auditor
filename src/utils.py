import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, na_values="?")
    df.columns.str.strip()
    return df
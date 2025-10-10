import pandas as pd

def loadfile(path:str):
    df = pd.read_csv(path)
    df.dropna(subset=["Summary"])
    df["Summary"] = df["Summary"].astype(str)

    return df
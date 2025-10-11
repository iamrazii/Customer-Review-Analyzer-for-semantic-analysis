import pandas as pd

def loadfile(path:str):
    df = pd.read_csv(path)
    df.dropna(subset=["Summary"])
    df["Summary"] = df["Summary"].astype(str)

    df_reduced = (
    df.groupby('product_name', group_keys=False)
      .apply(lambda x: x.loc[x.sample(frac=0.005, random_state=42).index])
      .reset_index(drop=True)
    )




    return df_reduced
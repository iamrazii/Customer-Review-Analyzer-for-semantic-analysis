import pandas as pd

def loadfile(path:str):
    df = pd.read_csv(path)
    df.dropna(subset=["Summary"])
    df["Summary"] = df["Summary"].astype(str)

    unique_products = df["product_name"].unique()
    product_id_map = {name: idx + 100 for idx, name in enumerate(unique_products)}

    # Step 2 — map those IDs back into the dataframe
    df["product_id"] = df["product_name"].map(product_id_map)
    
    df_reduced = (
    df.groupby('product_name', group_keys=False)
      .sample(frac=0.005, random_state=42)
      .reset_index(drop=True)
    )



    return df_reduced
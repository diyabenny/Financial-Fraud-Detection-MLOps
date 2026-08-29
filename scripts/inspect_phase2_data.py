import pandas as pd


files = {
    "APRIL": "data/incoming/april/converted_april_file.csv",
    "MAY": "data/incoming/may/converted_may_file.csv",
}


for name, path in files.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumn names:")
    print(df.columns.tolist())

    if "TX_DATETIME" in df.columns:
        df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

        print("\nDate range:")
        print("Start:", df["TX_DATETIME"].min())
        print("End:", df["TX_DATETIME"].max())

    if "TX_TIME_DAYS" in df.columns:
        print("\nTX_TIME_DAYS range:")
        print(df["TX_TIME_DAYS"].min(), "to", df["TX_TIME_DAYS"].max())

    if "TX_FRAUD" in df.columns:
        print("\nFraud distribution:")
        print(df["TX_FRAUD"].value_counts())
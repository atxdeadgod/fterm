from backend.data_provider import DataManager
import pandas as pd

dm = DataManager()
print("Testing get_fed_news()...")
try:
    df = dm.get_fed_news()
    if df.empty:
        print("Result: EMPTY DataFrame")
    else:
        print(f"Result: Found {len(df)} rows.")
        print(df.head())
except Exception as e:
    print(f"Result: ERROR - {e}")

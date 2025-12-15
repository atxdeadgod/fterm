from backend.data_provider import DataManager
import pandas as pd

dm = DataManager()
print("Testing get_analyst_revisions('IBM')...")
try:
    df = dm.get_analyst_revisions('IBM')
    if not df.empty:
        print("Columns:", df.columns.tolist())
        if 'date' in df.columns:
            print("SUCCESS: 'date' column present.")
        else:
            print("FAILURE: 'date' column MISSING.")
    else:
        print("DataFrame is empty.")
except Exception as e:
    print(f"Error: {e}")

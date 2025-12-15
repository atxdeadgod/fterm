from backend.data_provider import DataManager
import pandas as pd

dm = DataManager()
print("Listing tables in 'ibes' library...")
try:
    db = dm._get_conn()
    tables = db.list_tables(library='ibes')
    pt_tables = [t for t in tables if 'ptg' in t or 'rec' in t]
    print("Found Ratings/Target tables:")
    print(pt_tables)
except Exception as e:
    print(f"Error: {e}")

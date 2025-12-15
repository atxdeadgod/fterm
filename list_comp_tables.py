from backend.data_provider import DataManager
import pandas as pd

dm = DataManager()
print("Listing tables in 'comp' library...")
try:
    db = dm._get_conn()
    tables = db.list_tables(library='comp')
    seg_tables = [t for t in tables if 'seg' in t]
    print("Found segment tables:")
    print(seg_tables)
except Exception as e:
    print(f"Error: {e}")

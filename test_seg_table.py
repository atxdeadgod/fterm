from backend.data_provider import DataManager
import pandas as pd

dm = DataManager()
print("Testing 'comp.wrds_segmerged'...")
try:
    db = dm._get_conn()
    gvkey = '006066' # IBM
    q = f"""
        select *
        from comp.wrds_segmerged
        where gvkey = '{gvkey}'
        order by datadate desc
        limit 1
    """
    df = db.raw_sql(q)
    if not df.empty:
        print("Success! Columns:")
        print(df.columns.tolist())
    else:
        print("Table exists but returned empty for IBM.")
except Exception as e:
    print(f"Error: {e}")

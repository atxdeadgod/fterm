from backend.data_provider import DataManager
import pandas as pd

dm = DataManager()
print("Testing IBES schemas...")
try:
    db = dm._get_conn()
    ticker = 'IBM'
    
    # 1. Recommendations Summary
    print("\n--- ibes.recdsum ---")
    q1 = f"select * from ibes.recdsum where ticker='{ticker}' order by statpers desc limit 1"
    df1 = db.raw_sql(q1)
    if not df1.empty:
        print(df1.columns.tolist())
        print(df1.iloc[0].to_dict())
        
    # 2. Price Target Summary
    print("\n--- ibes.ptgsum ---")
    q2 = f"select * from ibes.ptgsum where ticker='{ticker}' order by statpers desc limit 1"
    df2 = db.raw_sql(q2)
    if not df2.empty:
        print(df2.columns.tolist())
        print(df2.iloc[0].to_dict())

except Exception as e:
    print(f"Error: {e}")

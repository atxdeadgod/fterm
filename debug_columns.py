from backend.data_provider import DataManager
import pandas as pd

dm = DataManager()
ticker = 'GOOGL'

print(f"--- FUndamentals ({ticker}) ---")
fund_df = dm.get_ratios(ticker)
print(fund_df.columns.tolist() if not fund_df.empty else "Empty")

print(f"--- Insider ({ticker}) ---")
insider_df = dm.get_insider_transactions(ticker)
print(insider_df.columns.tolist() if not insider_df.empty else "Empty")

print(f"--- Loans ({ticker}) ---")
loan_df = dm.get_corporate_loans(ticker)
print(loan_df.columns.tolist() if not loan_df.empty else "Empty")


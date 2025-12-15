import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from backend.data_provider import DataManager
import statsmodels.api as sm
from datetime import datetime, timedelta

# Page Config
st.set_page_config(
    page_title="RGTERM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize DataManager
@st.cache_resource
def get_manager():
    return DataManager()

dm = get_manager()

# Sidebar
st.sidebar.title("RGTERM")

# Ticker Selection
@st.cache_data
def load_tickers():
    df = dm.get_ticker_list()
    if df.empty:
        return []
    return df['label'].tolist()

ticker_options = load_tickers()

if ticker_options:
    # Default index
    default_ix = 0
    # Try to find IBM
    res = [i for i, s in enumerate(ticker_options) if s.startswith('IBM |')]
    if res:
        default_ix = res[0]

    selected_label = st.sidebar.selectbox(
        "Search Ticker", 
        options=ticker_options, 
        index=default_ix,
        help="Type to search..."
    )
    if selected_label:
        ticker = selected_label.split('|')[0].strip()
    else:
        ticker = "IBM"
else:
    # Fallback if list fails to load
    st.sidebar.warning("Search unavailable (Network/Cache issue). Using manual entry.")
    ticker = st.sidebar.text_input("Enter Ticker", value="IBM").upper()

# Date Range Picker
today = datetime.today()
start_default = today - timedelta(days=365*2)
date_range = st.sidebar.date_input(
    "Analysis Period",
    value=(start_default, today),
    max_value=today
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = start_default, today

if st.sidebar.button("Load Data"):
    st.session_state['ticker'] = ticker

if 'ticker' not in st.session_state:
    st.session_state['ticker'] = "IBM"

current_ticker = st.session_state['ticker']

st.title(f"{current_ticker}")

# Fetch Data
with st.spinner(f"Fetching data for {current_ticker}..."):
    try:
        # Pass start/end date to get_prices to optimize? 
        # For caching reasons, provider defaults to long history, we filter here.
        prices_df = dm.get_prices(current_ticker)
        fund_df = dm.get_ratios(current_ticker)
        est_df = dm.get_estimates(current_ticker)
        factors_df = dm.get_factors(start_date='2000-01-01') # Get long history for factors
        
        availability = dm.get_data_availability(current_ticker)
        
        if availability:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Latest Data Available**")
            if 'prices' in availability and availability['prices']:
               st.sidebar.info(f"Prices: {availability['prices'][:10]}")
            if 'fundamentals' in availability and availability['fundamentals']:
               st.sidebar.info(f"Financials: {availability['fundamentals'][:10]}")
            
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        prices_df = pd.DataFrame()
        fund_df = pd.DataFrame()
        est_df = pd.DataFrame()
        factors_df = pd.DataFrame()
        availability = {}

# Filter Logic
prices_filtered = pd.DataFrame()
if not prices_df.empty and 'date' in prices_df.columns:
    mask_p = (prices_df['date'] >= pd.to_datetime(start_date)) & (prices_df['date'] <= pd.to_datetime(end_date))
    prices_filtered = prices_df.loc[mask_p].copy()

# Standardize Fundamentals Date & Columns (Global Fix)
if not fund_df.empty:
    if 'date' not in fund_df.columns and 'datadate' in fund_df.columns:
        fund_df['date'] = pd.to_datetime(fund_df['datadate'])
    if 'sale' not in fund_df.columns and 'saleq' in fund_df.columns:
        fund_df['sale'] = fund_df['saleq']
    if 'ni' not in fund_df.columns and 'niq' in fund_df.columns:
        fund_df['ni'] = fund_df['niq']

factors_filtered = pd.DataFrame()
if not factors_df.empty and 'date' in factors_df.columns:
    mask_f = (factors_df['date'] >= pd.to_datetime(start_date)) & (factors_df['date'] <= pd.to_datetime(end_date))
    factors_filtered = factors_df.loc[mask_f].copy()

# Advanced Analysis: Regressions
alpha = 0.0
beta_mkt = 1.0
beta_smb = 0.0
beta_hml = 0.0
r2 = 0.0
excess_ret_cum = pd.DataFrame()
rolling_beta = pd.DataFrame()

if not prices_filtered.empty and not factors_filtered.empty:
    # Prepare Data for Regression
    reg_df = pd.merge(prices_filtered, factors_filtered, on='date', how='inner')
    
    if not reg_df.empty:
        # Stock Excess Return = ret - rf
        # Ensure all columns are float
        cols_to_check = ['ret', 'rf', 'mktrf', 'smb', 'hml']
        for c in cols_to_check:
             if c in reg_df.columns:
                 reg_df[c] = pd.to_numeric(reg_df[c], errors='coerce')
        
        reg_df.dropna(subset=cols_to_check, inplace=True)

        if not reg_df.empty:
            reg_df['stock_ex_ret'] = reg_df['ret'] - reg_df['rf']
            
            # Regression: stock_ex_ret ~ mktrf + smb + hml
            # Explicitly cast to float to satisfy statsmodels
            X = reg_df[['mktrf', 'smb', 'hml']].astype(float)
            X = sm.add_constant(X)
            y = reg_df['stock_ex_ret'].astype(float)
            
            try:
                model = sm.OLS(y, X).fit()
                
                alpha = model.params['const']
                beta_mkt = model.params['mktrf']
                beta_smb = model.params['smb']
                beta_hml = model.params['hml']
                r2 = model.rsquared
            except Exception as e:
                st.error(f"Regression error: {e}")

            # Cumulative Returns Calculation for Chart
            reg_df['cum_ret_stock'] = (1 + reg_df['ret']).cumprod() - 1
            reg_df['cum_ret_mkt'] = (1 + reg_df['mktrf'] + reg_df['rf']).cumprod() - 1 
            excess_ret_cum = reg_df
            
            # Rolling Beta Analysis
            window_size = st.sidebar.slider("Rolling Window (Days)", 20, 252, 60)
            
            # Simple rolling covariance calculation for Market Beta
            # Beta = Cov(Stock, Mkt) / Var(Mkt)
            # using 'ret' (Total Return) vs 'mktrf'+'rf' (Mkt Total Return)
            reg_df['mkt_total'] = reg_df['mktrf'] + reg_df['rf']
            
            roll_cov = reg_df['ret'].rolling(window_size).cov(reg_df['mkt_total'])
            roll_var = reg_df['mkt_total'].rolling(window_size).var()
            reg_df['rolling_beta'] = roll_cov / roll_var
            rolling_beta = reg_df.dropna(subset=['rolling_beta'])

# Metrics Calculation
current_price = 0.0
total_return = 0.0
volatility = 0.0

if not prices_filtered.empty:
    current_price = prices_filtered.iloc[-1]['prc'] # Display Raw Price
    
    if len(prices_filtered) > 1:
        # Use Adjusted Close for Return Calculation to handle Splits
        start_p_adj = prices_filtered.iloc[0]['adj_close']
        end_p_adj = prices_filtered.iloc[-1]['adj_close']
        
        total_return = (end_p_adj / start_p_adj) - 1 if start_p_adj != 0 else 0.0
        
        if 'ret' in prices_filtered.columns:
             volatility = prices_filtered['ret'].std() * (252**0.5)
        else:
             volatility = prices_filtered['adj_close'].pct_change().std() * (252**0.5)

# Layout
col1, col2 = st.columns([3, 1])

# Navigation (Right Column)
with col2:
    st.subheader("Navigation")
    nav_options = [
        "Price & Performance", "Factor Analysis", "Peer Comparison", 
        "Fundamentals & Valuation", "Estimates", "Ownership & Shorts", 
        "Debt & Credit", "News & Sentiment", "Governance", "Derivatives", 
        "Corporate Bonds", "Macro & Fed News"
    ]
    selection = st.radio("Go to Module", nav_options, label_visibility="collapsed")
    
    st.markdown("---")
    st.subheader("Snapshot")
    if not fund_df.empty and not prices_df.empty:
        last_price = prices_df.iloc[-1]['prc']
        shares = fund_df.iloc[0]['cshoq'] if 'cshoq' in fund_df.columns else 0
        mkt_cap = last_price * shares
        st.metric("Last Price", f"${last_price:,.2f}")
        st.metric("Mkt Cap", f"${mkt_cap:,.2f} M")
        st.metric("Total Return", f"{total_return:.2%}")
        st.metric("Ann. Volatility", f"{volatility:.2%}")

# Main Content (Left Column)
with col1:
    if selection == "Price & Performance":
        if not prices_filtered.empty:
            col_chart, col_metrics = st.columns([3, 1])
            with col_chart:
                 st.subheader("Price History (Split-Adjusted)")
                 fig = go.Figure()
                 fig.add_trace(go.Scatter(x=prices_filtered['date'], y=prices_filtered['adj_close'], mode='lines', name='Price'))
                 fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0))
                 st.plotly_chart(fig, use_container_width=True)
                 
                 if not excess_ret_cum.empty:
                    st.subheader("Performance vs Market")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=excess_ret_cum['date'], y=excess_ret_cum['cum_ret_stock'], mode='lines', name=f'{current_ticker} Cum Ret'))
                    fig2.add_trace(go.Scatter(x=excess_ret_cum['date'], y=excess_ret_cum['cum_ret_mkt'], mode='lines', name='Market Cum Ret', line=dict(dash='dash')))
                    fig2.update_layout(yaxis_tickformat='.0%', height=300, margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig2, use_container_width=True)

            with col_metrics:
                st.metric("Latest Price", f"${current_price:.2f}")
                st.metric("Total Return", f"{total_return:.2%}")
                st.metric("Annualized Volatility", f"{volatility:.2%}")
        else:
            st.warning("No Price Data for selected range")

    elif selection == "Factor Analysis":
        st.subheader("Advanced Factor Analysis (FF5 + Momentum + Sector)")
        
        # Update Factors to FF6
        # Start date for factors might need to align if we switch to long history
        # We will assume factors_df passed in is now FF6 or we need to re-fetch
        # Actually, let's fetch extended factors specifically here or update the main fetch
        
        with st.spinner("Calculating Advanced Factors..."):
            ff6_df = dm.get_extended_factors(start_date='2010-01-01')
            sector_df = dm.get_sector_index(current_ticker, start_date='2010-01-01')
        
        if not prices_filtered.empty and not ff6_df.empty:
            # Merge Prices + FF6
            reg_df_adv = pd.merge(prices_filtered, ff6_df, on='date', how='inner')
            
            if not sector_df.empty:
                 reg_df_adv = pd.merge(reg_df_adv, sector_df, on='date', how='left')  
            
            if not reg_df_adv.empty:
                # 1. FF6 Regression
                # ret - rf = a + b*Mkt + s*SMB + h*HML + r*RMW + c*CMA + m*MOM
                reg_df_adv['stock_ex_ret'] = reg_df_adv['ret'] - reg_df_adv['rf']
                
                features = ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'mom']
                # Ensure existence (mom might be missing if DB fetch failed)
                features = [f for f in features if f in reg_df_adv.columns]
                
                # Clean NaNs
                reg_df_adv = reg_df_adv.dropna(subset=['stock_ex_ret'] + features)
                
                try:
                    X = reg_df_adv[features].astype(float)
                    X = sm.add_constant(X)
                    y = reg_df_adv['stock_ex_ret'].astype(float)
                    
                    model = sm.OLS(y, X).fit()
                    
                    # Display Coefficients
                    st.markdown("#### Fama-French 6-Factor Model")
                    
                    # Layout: 3 columns of metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Alpha (Daily)", f"{model.params['const']:.5f}")
                    c1.metric("Market Beta", f"{model.params.get('mktrf', 0):.2f}")
                    c2.metric("Size (SMB)", f"{model.params.get('smb', 0):.2f}")
                    c2.metric("Value (HML)", f"{model.params.get('hml', 0):.2f}")
                    c3.metric("Quality (RMW)", f"{model.params.get('rmw', 0):.2f}")
                    c3.metric("Inv (CMA)", f"{model.params.get('cma', 0):.2f}")
                    
                    st.metric("Momentum (MOM)", f"{model.params.get('mom', 0):.2f}")
                    
                    st.caption(f"R-Squared: {model.rsquared:.2%} | Observations: {len(reg_df_adv)}")
                    
                    # 2. Sector Beta
                    if 'sector_ret' in reg_df_adv.columns:
                        st.markdown("---")
                        st.markdown("#### Sector Analysis")
                        
                        # Regression: Stock Ret ~ Sector Ret
                        # Logic: strictly correlated to sector index?
                        # Or relative to market? Let's just do single factor beta vs sector
                        
                        reg_sec = reg_df_adv.dropna(subset=['ret', 'sector_ret'])
                        if not reg_sec.empty:
                            X_sec = sm.add_constant(reg_sec['sector_ret'].astype(float))
                            y_sec = reg_sec['ret'].astype(float)
                            model_sec = sm.OLS(y_sec, X_sec).fit()
                            
                            sec_beta = model_sec.params['sector_ret']
                            sec_alpha = model_sec.params['const']
                            
                            c1, c2 = st.columns(2)
                            c1.metric("Sector Beta", f"{sec_beta:.2f}", help="Sensitivity to Peer Group Index")
                            c2.metric("Sector Alpha", f"{sec_alpha:.5f}")
                        
                except Exception as e:
                    st.error(f"Error in Advanced Regression: {e}")

                # Rolling Analysis (Updated for User Choice?)
                # Keep the existing rolling market beta? Or add choices?
                # User asked for "rolling factor analysis for different factors"
                
                st.markdown("---")
                st.markdown("#### Rolling Factor Betas")
                
                roll_factor = st.selectbox("Select Factor for Rolling Analysis", features, index=0)
                window_size = st.sidebar.slider("Rolling Window", 20, 252, 60, key='roll_window_adv')
                
                # Rolling Regression is expensive. 
                # Rolling Covariance / Variance approximation for single factor sensitivity?
                # Beta_F = Cov(R, F) / Var(F)
                # This assumes univariate regression. For multivariate, it's complex.
                # Only "Market Beta" is reliably approximated by Univariate for visualization usually.
                # However, providing univariate rolling beta for 'smb' implies "Sensitivity to SMB ignoring other factors".
                # Let's do that for interactivity/speed.
                
                roll_cov = reg_df_adv['stock_ex_ret'].rolling(window_size).cov(reg_df_adv[roll_factor])
                roll_var = reg_df_adv[roll_factor].rolling(window_size).var()
                rolling_beta_f = roll_cov / roll_var
                
                fig_rb = go.Figure()
                fig_rb.add_trace(go.Scatter(x=reg_df_adv['date'], y=rolling_beta_f, mode='lines', name=f'Rolling {roll_factor} Beta'))
                fig_rb.add_hline(y=0.0, line_dash="dash", line_color="black")
                fig_rb.update_layout(height=300, title=f"Rolling {window_size}-Day Beta to {roll_factor}")
                st.plotly_chart(fig_rb, use_container_width=True)

        else:
            st.warning("Insufficient data for Advanced Factors")

    elif selection == "Peer Comparison":
        st.subheader("Peer Group Analysis (SIC)")
        
        peers = dm.get_sic_peers(current_ticker)
        if peers:
            st.write(f"Found {len(peers)} peers based on industry code.")
            
            # Fetch bulk data
            # Include target ticker in comparison
            compare_list = [current_ticker] + peers
            bulk_df = dm.get_bulk_fundamentals(compare_list)
            
            if not bulk_df.empty:
                # Add Price data? Doing it simple for now, just fundamentals
                # Need latest price for P/E calculation
                # We can't fetch prices for 50 tickers loop easily in current setup without delay.
                # Just comparing Size (Assets) and Profitability (ROE) for now
                
                # Calculate ROE = niq / (atq - ltq)
                bulk_df['Equity'] = bulk_df['atq'] - bulk_df['ltq']
                bulk_df['ROE'] = bulk_df['niq'] / bulk_df['Equity']
                
                # Display table (Safe)
                if 'datadate' in bulk_df.columns and 'date' not in bulk_df.columns:
                    bulk_df['date'] = pd.to_datetime(bulk_df['datadate'])

                display_cols = ['tic', 'date', 'atq', 'niq', 'ROE', 'EPS_Ann']
                actual_cols = [c for c in display_cols if c in bulk_df.columns]
                st.dataframe(bulk_df[actual_cols].sort_values('atq', ascending=False).head(20))
                
                # Highlight Target Rank
                target_roe = bulk_df[bulk_df['tic'] == current_ticker]['ROE'].values
                if len(target_roe) > 0:
                    rank = (bulk_df['ROE'] > target_roe[0]).sum() + 1
                    total = len(bulk_df)
                    st.metric("ROE Rank", f"#{rank} of {total}")
            else:
                st.warning("Could not fetch peer fundamentals.")
        else:
            st.info("No peers found or SIC code missing.")

    elif selection == "Fundamentals & Valuation":
        if not fund_df.empty:
            st.subheader("Deep Fundamentals & Valuation")
            
            # 1. Valuation Model
            val_df = dm.get_valuation_metrics(current_ticker)
            if not val_df.empty:
                st.markdown("### Valuation Model (Historical P/E Bands)")
                current_pe = val_df.iloc[-1]['PE']
                avg_pe = val_df.iloc[-1]['AvgPE']
                
                c1, c2 = st.columns(2)
                c1.metric("Current P/E", f"{current_pe:.2f}")
                c2.metric("5Y Avg P/E", f"{avg_pe:.2f}")
                
                # Plot
                fig_val = go.Figure()
                fig_val.add_trace(go.Scatter(x=val_df['date'], y=val_df['prc'], mode='lines', name='Price', line=dict(color='black', width=2)))
                fig_val.add_trace(go.Scatter(x=val_df['date'], y=val_df['FairPrice'], mode='lines', name='Fair Value', line=dict(dash='dash', color='blue')))
                fig_val.add_trace(go.Scatter(x=val_df['date'], y=val_df['UpperBand'], mode='lines', name='+1 StdDev', line=dict(width=0), showlegend=False))
                fig_val.add_trace(go.Scatter(x=val_df['date'], y=val_df['LowerBand'], mode='lines', name='-1 StdDev', fill='tonexty', fillcolor='rgba(0,0,255,0.1)', line=dict(width=0), showlegend=True))
                
                fig_val.update_layout(height=400, title=f"Price vs Historical Fair Value (Avg P/E: {avg_pe:.1f}x)")
                st.plotly_chart(fig_val, use_container_width=True)
            else:
                st.info("Insufficient data for Valuation Model (Need positive earnings history)")
                
            st.markdown("---")
            
            # 2. Dupont Analysis
            dupont_df = dm.get_dupont_data(current_ticker)
            if not dupont_df.empty:
                st.markdown("### Dupont Analysis (ROE Decomposition)")
                st.markdown("ROE = Net Margin × Asset Turnover × Financial Leverage")
                
                # Stacked Bar Chart? Or just 3 lines? 
                # Stacked bar doesn't work well because they are Multiplicative, not Additive.
                # Normalized lines/area might be better.
                
                fig_d = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_d.add_trace(go.Scatter(x=dupont_df['datadate'], y=dupont_df['NetMargin'], name='Net Margin', line=dict(color='green')), secondary_y=False)
                fig_d.add_trace(go.Scatter(x=dupont_df['datadate'], y=dupont_df['AssetTurnover'], name='Asset Turnover', line=dict(color='blue')), secondary_y=False)
                fig_d.add_trace(go.Scatter(x=dupont_df['datadate'], y=dupont_df['Leverage'], name='Fin. Leverage', line=dict(color='orange')), secondary_y=True)
                
                fig_d.update_layout(title="Drivers of ROE", height=400)
                fig_d.update_yaxes(title_text="Margin / Turnover", secondary_y=False)
                fig_d.update_yaxes(title_text="Leverage (Equity Mult.)", secondary_y=True)
                
                st.plotly_chart(fig_d, use_container_width=True)
            
            st.subheader("Raw Data")
            st.dataframe(fund_df)
            
            # Show Ratios
            if 'BVPS' in fund_df.columns:
                # Latest Price for P/B
                last_price = prices_df.iloc[-1]['prc'] if not prices_df.empty else 0
                latest_bvps = fund_df.iloc[0]['BVPS'] if pd.notnull(fund_df.iloc[0]['BVPS']) else 0
                
                if latest_bvps > 0:
                   pb_ratio = last_price / latest_bvps
                   st.metric("P/B Ratio", f"{pb_ratio:.2f}x")
        else:
            st.info("No Fundamental Data Found")

    elif selection == "Estimates":
        st.subheader("Analyst Estimates (LSEG/IBES)")
        
        # 1. Historical Revisions Chart
        revisions_df = dm.get_analyst_revisions(current_ticker)
        if not revisions_df.empty:
             st.markdown("##### Consensus EPS Revisions (Next FY)")
             
             # Calculate 3M Trend
             latest_mean = revisions_df.iloc[-1]['meanest']
             
             # Look back ~3 months (approx 90 days or 3 rows if monthly)
             # statpers is approx monthly.
             if len(revisions_df) >= 4:
                 prev_mean = revisions_df.iloc[-4]['meanest']
                 chg = (latest_mean - prev_mean) / abs(prev_mean) if prev_mean != 0 else 0
                 st.metric("Mean Estimate (Next FY)", f"${latest_mean:.2f}", f"{chg:.1%} (3M Trend)")
             else:
                 st.metric("Mean Estimate (Next FY)", f"${latest_mean:.2f}")

             fig_rev = go.Figure()
             
             # Forecast Range (High/Low)
             fig_rev.add_trace(go.Scatter(
                 x=revisions_df['statpers'], y=revisions_df['highest'],
                 mode='lines', line=dict(width=0), showlegend=False, name='High'
             ))
             fig_rev.add_trace(go.Scatter(
                 x=revisions_df['statpers'], y=revisions_df['lowest'],
                 mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,255,0.1)', showlegend=True, name='Range (High-Low)'
             ))
             
             # Mean Estimate
             fig_rev.add_trace(go.Scatter(
                 x=revisions_df['statpers'], y=revisions_df['meanest'],
                 mode='lines', line=dict(color='blue', width=2), name='Mean Estimate'
             ))
             
             fig_rev.update_layout(title=f"Analyst Consensus Trend ({current_ticker})", height=350, yaxis_title="EPS Estimate ($)")
             st.plotly_chart(fig_rev, use_container_width=True)
             
             st.markdown("---")

        if not est_df.empty:
            st.subheader("Detailed Estimates Snapshot")
            st.dataframe(est_df.head(20))
        else:
            st.info("No Detailed Estimates Data Found")

    elif selection == "Ownership & Shorts":
        st.subheader("Ownership & Short Interest")
        
        col_short, col_inst = st.columns(2)
        
        with col_short:
            st.markdown("##### Short Interest")
            short_df = dm.get_short_interest(current_ticker)
            if not short_df.empty:
                # Latest
                latest_short = short_df.iloc[-1]
                st.metric("Shares Short", f"{latest_short['shortint']:,.0f}", f"Date: {latest_short['datadate'].strftime('%Y-%m-%d')}")
                
                fig_short = go.Figure()
                fig_short.add_trace(go.Scatter(x=short_df['datadate'], y=short_df['shortint'], fill='tozeroy', name='Short Interest'))
                fig_short.update_layout(title="Short Interest (Shares)", height=300)
                st.plotly_chart(fig_short, use_container_width=True)
            else:
                st.info("No Short Interest data available.")

        with col_inst:
            st.markdown("##### Institutional Holdings (13F)")
            inst_df = dm.get_institutional_holdings(current_ticker)
            if not inst_df.empty:
                # Fix Date
                d_col = 'fdate' if 'fdate' in inst_df.columns else 'rdate'
                s_col = 'total_shares' if 'total_shares' in inst_df.columns else 'shares'
                
                if d_col in inst_df.columns:
                     latest_inst = inst_df.sort_values(d_col).iloc[-1]
                     # Check columns exist
                     ts = latest_inst[s_col] if s_col in latest_inst else 0
                     ni = latest_inst['num_institutions'] if 'num_institutions' in latest_inst else 0
                     st.metric("Total Inst. Shares", f"{ts:,.0f}", f"Institutions: {ni}")
                    
                     fig_inst = go.Figure()
                     fig_inst.add_trace(go.Scatter(x=inst_df[d_col], y=inst_df[s_col] if s_col in inst_df.columns else 0, name='Shares Held'))
                     fig_inst.update_layout(title="Institutional Shares Held", height=300)
                     st.plotly_chart(fig_inst, use_container_width=True)
                else:
                     st.dataframe(inst_df.head())
            else:
                st.info("No Institutional data available.")
        
        st.markdown("---")
        st.markdown("##### Insider Transactions")
        insider_df = dm.get_insider_transactions(current_ticker)
        if not insider_df.empty:
            # Fix PersonID
            pid = 'personid' if 'personid' in insider_df.columns else 'personID'
            
            # Chart: Net Buy/Sell Value
            # Aggregate by day for chart
            daily_insider = insider_df.groupby('trandate')['signed_value'].sum().reset_index()
            
            fig_insider = go.Figure()
            colors = ['green' if v > 0 else 'red' for v in daily_insider['signed_value']]
            fig_insider.add_trace(go.Bar(
                x=daily_insider['trandate'], 
                y=daily_insider['signed_value'],
                marker_color=colors,
                name='Net Transaction Value'
            ))
            fig_insider.update_layout(title="Net Insider Buying/Selling ($)", height=300)
            st.plotly_chart(fig_insider, use_container_width=True)
            
            # Table
            cols_i = ['trandate', pid, 'rolecode1', 'acqdisp', 'shares', 'tprice', 'value']
            actual_i = [c for c in cols_i if c in insider_df.columns]
            st.dataframe(
                insider_df[actual_i].head(50),
                use_container_width=True
            )
        else:
            st.info("No Insider Transaction data available.")

    elif selection == "Debt & Credit":
        st.subheader("Corporate Debt & Credit (DealScan)")
        loan_df = dm.get_corporate_loans(current_ticker)
        
        if not loan_df.empty:
            # Metrics
            total_borrowed = loan_df['dealamount'].sum()
            recent_deals = len(loan_df[loan_df['dealactivedate'] >= (datetime.now() - timedelta(days=365*3))])
            
            m1, m2 = st.columns(2)
            m1.metric("Total Deal Volume (All History)", f"${total_borrowed:,.0f} M")
            m2.metric("Deals (Last 3 Years)", recent_deals)
            
            # Chart
            d_col = 'dealactivedate' if 'dealactivedate' in loan_df.columns else 'tranche_start_date'
            p_col = 'dealpurpose' if 'dealpurpose' in loan_df.columns else 'primary_purpose'
            
            if d_col in loan_df.columns:
                fig_loan = go.Figure()
                fig_loan.add_trace(go.Bar(
                    x=loan_df[d_col], 
                    y=loan_df['dealamount'],
                    text=loan_df[p_col] if p_col in loan_df.columns else '',
                    name='Deal Amount'
                ))
            fig_loan.update_layout(title="Capital Raising History (Syndicated Loans)", height=400, yaxis_title="Amount (Millions)")
            st.plotly_chart(fig_loan, use_container_width=True)
            
            st.markdown("##### Detailed Loan Dictionary")
            st.dataframe(loan_df[['dealactivedate', 'dealamount', 'currency', 'dealpurpose', 'dealstatus', 'company']], use_container_width=True)
        else:
            st.info("No DealScan Corporate Loan data found for this entity.")

    elif selection == "News & Sentiment":
        st.subheader("News Analytics & Sentiment (RavenPack)")
        news_df = dm.get_news_sentiment(current_ticker)
        
        if not news_df.empty:
            # 1. Sentiment Trend (Daily Average)
            # Ensure correct date time
            if 'timestamp_utc' in news_df.columns:
                 news_df['date'] = pd.to_datetime(news_df['timestamp_utc']).dt.date
            elif 'date' not in news_df.columns:
                 # Fallback
                 news_df['date'] = datetime.today().date()
            daily_sentiment = news_df.groupby('date')['event_sentiment_score'].mean().reset_index()
            
            # Metric: Today's Sentiment
            # ESS range: 0-100 (50 neutral/positive boundary depending on version, usually 50 is neutral)
            latest_sent = daily_sentiment.iloc[-1]['event_sentiment_score']
            latest_date = daily_sentiment.iloc[-1]['date']
            
            # Color logic
            color_delta = "normal"
            if latest_sent > 50: color_delta = "normal" # Streamlit handles green for positive delta manually if needed
            
            st.metric("Latest Sentiment Score (0-100)", f"{latest_sent:.1f}", f"Date: {latest_date}")
            
            # Chart
            fig_sent = go.Figure()
            fig_sent.add_trace(go.Bar(
                x=daily_sentiment['date'], 
                y=daily_sentiment['event_sentiment_score'],
                marker_color=['green' if v > 50 else 'red' for v in daily_sentiment['event_sentiment_score']],
                name='Daily Sentiment'
            ))
            fig_sent.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Neutral (50)")
            fig_sent.update_layout(title="Daily News Sentiment Trend", height=350, yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_sent, use_container_width=True)
            
            st.markdown("---")
            st.markdown("##### Real-time News Feed")
            
            # Display News Feed with color coding
            # Streamlit dataframe doesn't support row coloring easily, so we'll iterate
            # Or just show a nice table
            
            # Filter cols
            display_news = news_df[['timestamp_utc', 'headline', 'event_sentiment_score', 'topic', 'group']].head(20)
            
            # Custom formatting
            st.dataframe(
                display_news.style.map(
                    lambda x: 'color: green' if (pd.notna(x) and x > 50) else 'color: red', subset=['event_sentiment_score']
                ),
                use_container_width=True
            )
            
        else:
            st.info("No RavenPack News data found (checking 2024-2025 history).")

    elif selection == "Governance":
        st.subheader("Governance & Shareholder Rights (RiskMetrics)")
        gov_df = dm.get_governance_profile(current_ticker)
        
        if not gov_df.empty:
            gov = gov_df.iloc[0]
            st.caption(f"Data Year: {gov['year']}")
            
            # Helper to safely check YES
            def check_yes(val):
                if pd.isna(val): return False
                return str(val).upper() == 'YES'

            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown("**Board Structure**")
                # Classified Board
                is_staggered = "YES" if check_yes(gov['cboard']) else "NO"
                color = "red" if is_staggered == "YES" else "green"
                st.markdown(f"Classified Board: :{color}[**{is_staggered}**]")
            
            with c2:
                st.markdown("**Defenses**")
                # Poison Pill
                has_pill = "YES" if check_yes(gov['ppill']) else "NO"
                color_p = "red" if has_pill == "YES" else "green"
                st.markdown(f"Poison Pill: :{color_p}[**{has_pill}**]")
                
                # Golden Parachute
                has_gp = "YES" if check_yes(gov['gparachute']) else "NO"
                st.markdown(f"Golden Parachute: **{has_gp}**")
            
            with c3:
                st.markdown("**Voting Rights**")
                # Dual Class
                is_dual = "YES" if check_yes(gov['dualclass']) else "NO"
                color_d = "red" if is_dual == "YES" else "green"
                st.markdown(f"Dual Class: :{color_d}[**{is_dual}**]")
                
                # Confidential Voting
                conf_vote = gov['confvote'] if pd.notna(gov['confvote']) else "N/A"
                st.markdown(f"Confidential Voting: **{conf_vote}**")
                
            st.markdown("---")
            st.info("Metrics provided by ISS/RiskMetrics. 'YES' in Board/Defenses often indicates lower shareholder rights index.")
            
        else:
            st.info("No Governance data found.")

    elif selection == "Derivatives":
        st.subheader("Derivatives & Volatility (OptionMetrics)")
        vol_df = dm.get_volatility_surface(current_ticker)
        
        if not vol_df.empty:
            st.caption(f"Surface Date: {vol_df['date'].max().strftime('%Y-%m-%d')}")
            
            # Volatility Smile Chart (IV vs Delta)
            # Filter for specific maturities: e.g. 30 days, 91 days
            # Plot Calls (Delta > 0) vs Puts (Delta < 0) or just map X axis
            
            # We want to show IV vs Delta for a few maturities.
            # Delta ranges from -90 to 90 (approx). 
            # We usually plot 'delta' on X and 'impl_vol' on Y.
            
            fig_smile = go.Figure()
            
            days_to_plot = [30, 91, 182]
            colors = ['cyan', 'magenta', 'yellow']
            
            for i, d in enumerate(days_to_plot):
                subset = vol_df[vol_df['days'] == d].sort_values('delta')
                if not subset.empty:
                    fig_smile.add_trace(go.Scatter(
                        x=subset['delta'], 
                        y=subset['impl_volatility'], 
                        mode='lines+markers', 
                        name=f'{d} Days',
                        line=dict(color=colors[i % len(colors)])
                    ))
            
            fig_smile.update_layout(
                title="Volatility Smile (IV vs Delta)",
                xaxis_title="Delta (Put < 0 < Call)",
                yaxis_title="Implied Volatility",
                height=400
            )
            st.plotly_chart(fig_smile, use_container_width=True)
            
            # Term Structure (ATM Volatility vs Days)
            # ATM is roughly Delta 50 (Call) or Delta -50 (Put).
            # Let's approximate ATM as closest to delta 50.
            
            # Filter for Calls around 50 delta
            atm_subset = vol_df[
                (vol_df['cp_flag'] == 'C') & 
                (vol_df['delta'].between(45, 55))
            ].sort_values('days')
            
            if not atm_subset.empty:
                 st.subheader("Term Structure (ATM Volatility)")
                 fig_term = go.Figure()
                 fig_term.add_trace(go.Scatter(
                     x=atm_subset['days'],
                     y=atm_subset['impl_volatility'],
                     mode='lines+markers',
                     name='ATM IV (Call 50d)'
                 ))
                 fig_term.update_layout(xaxis_title="Days to Maturity", yaxis_title="Implied Volatility", height=300)
                 st.plotly_chart(fig_term, use_container_width=True)
            
            with st.expander("Raw Volatility Surface Data"):
                st.dataframe(vol_df)
            
            # --- Heatmap / Contour: Volatility Surface ---
            st.subheader("Volatility Surface Heatmap")
            
            # Prepare data for Contour
            # Pivot to create a grid: Index=Days, Columns=Delta, Values=Implied Vol
            # We might need to round Delta to nearest integer to group them effectively
            
            heatmap_df = vol_df.copy()
            heatmap_df['delta_rounded'] = heatmap_df['delta'].round(0)
            
            # Create pivot table
            surf_pivot = heatmap_df.pivot_table(
                index='days', 
                columns='delta_rounded', 
                values='impl_volatility', 
                aggfunc='mean'
            )
            
            if not surf_pivot.empty:
                fig_surf = go.Figure(data=go.Contour(
                    z=surf_pivot.values,
                    x=surf_pivot.columns, # Delta
                    y=surf_pivot.index,   # Days
                    colorscale='Hot',
                    colorbar=dict(title='Implied Volatility'),
                    contours=dict(
                        coloring='heatmap',
                        showlabels=True, # Show IV numbers on lines
                        labelfont=dict(size=10, color='white')
                    )
                ))
                
                fig_surf.update_layout(
                    title="Volatility Surface (X=Delta, Y=Maturity, Color=IV)",
                    xaxis_title="Delta (Moneyness)",
                    yaxis_title="Days to Maturity",
                    height=500
                )
                
                st.plotly_chart(fig_surf, use_container_width=True)
                st.caption("Detailed view of the Implied Volatility Surface. Higher temperatures indicate more expensive options (higher IV).")
                
        else:
             st.info("No OptionMetrics Volatility Surface data found.")
    
    elif selection == "Corporate Bonds":
        st.subheader("Corporate Bonds (TRACE)")
        bond_df = dm.get_bond_transactions(current_ticker)
        
        if not bond_df.empty:
            st.caption(f"Last Transaction: {bond_df['datetime'].max()}")
            
            # Scatter Plot of Yields over time
            # Color by Volume or Bond Symbol
            
            fig_yld = go.Figure()
            
            # Top 10 most active bonds by volume
            if 'bond_sym_id' in bond_df.columns:
                top_bonds = bond_df.groupby('bond_sym_id')['entrd_vol_qt'].sum().nlargest(10).index.tolist()
                plot_df = bond_df[bond_df['bond_sym_id'].isin(top_bonds)].copy()
                
                for b_sym in top_bonds:
                    subset = plot_df[plot_df['bond_sym_id'] == b_sym].sort_values('datetime')
                    fig_yld.add_trace(go.Scatter(
                        x=subset['datetime'], 
                        y=subset['yld_pt'],
                        mode='markers',
                        name=b_sym,
                        marker=dict(size=5, opacity=0.7)
                    ))
            else:
                fig_yld.add_trace(go.Scatter(
                    x=bond_df['datetime'], 
                    y=bond_df['yld_pt'], 
                    mode='markers',
                    marker=dict(color='orange', size=5, opacity=0.7)
                ))
                
            fig_yld.update_layout(
                title="Bond Yield History (Top 10 Active Issues)",
                xaxis_title="Transaction Time",
                yaxis_title="Yield (%)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_yld, use_container_width=True)
            
            # Price History
            st.subheader("Bond Price History")
            fig_prc = go.Figure()
             # Top 10 most active bonds by volume (reuse)
            if 'bond_sym_id' in bond_df.columns:
                 for b_sym in top_bonds:
                    subset = plot_df[plot_df['bond_sym_id'] == b_sym].sort_values('datetime')
                    fig_prc.add_trace(go.Scatter(
                        x=subset['datetime'], 
                        y=subset['rptd_pr'],
                        mode='lines+markers',
                        name=b_sym,
                        marker=dict(size=4)
                    ))
            else:
                 fig_prc.add_trace(go.Scatter(
                    x=bond_df['datetime'], 
                    y=bond_df['rptd_pr'], 
                    mode='markers',
                    name='Price'
                ))
            
            fig_prc.update_layout(height=350, yaxis_title="Price ($)")
            st.plotly_chart(fig_prc, use_container_width=True)

            with st.expander("Recent Bond Transactions"):
                st.dataframe(bond_df[['datetime', 'bond_sym_id', 'cusip_id', 'rptd_pr', 'yld_pt', 'entrd_vol_qt']].head(100))
                
        else:
             st.info("No TRACE Corporate Bond data found.")

    elif selection == "Macro & Fed News":
        st.subheader("Macroeconomic Cockpit (Federal Reserve)")
        
        macro_df = dm.get_macro_data(start_date=(datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d'))
        
        if not macro_df.empty:
            # 1. Yield Curve Inversion (10Y - 2Y)
            st.markdown("#### Yield Curve (10Y - 2Y Spread)")
            st.caption("Negative values (Inversion) often predict recessions.")
            
            fig_yc = go.Figure()
            fig_yc.add_trace(go.Scatter(
                x=macro_df['date'], 
                y=macro_df['t10y2y'],
                fill='tozeroy',
                name='10Y-2Y Spread',
                line=dict(color='gray')
            ))
            fig_yc.add_hline(y=0, line_dash="dash", line_color="red")
            fig_yc.update_layout(height=400, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_yc, use_container_width=True)
            
            # 2. Key Rates
            st.markdown("#### Key Interest Rates")
            fig_rates = go.Figure()
            fig_rates.add_trace(go.Scatter(x=macro_df['date'], y=macro_df['dff'], name='Fed Funds Rate'))
            fig_rates.add_trace(go.Scatter(x=macro_df['date'], y=macro_df['dgs10'], name='10Y Treasury'))
            fig_rates.add_trace(go.Scatter(x=macro_df['date'], y=macro_df['bamlh0a0hym2ey'], name='High Yield Index', line=dict(dash='dot')))
            fig_rates.update_layout(height=400, margin=dict(t=0, b=0), hovermode="x unified")
            st.plotly_chart(fig_rates, use_container_width=True)
            
        else:
            st.info("No Macro Data (FRB) found.")

# Removed duplicate Snapshot (Nav moved to right column)

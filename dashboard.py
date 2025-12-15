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
    page_title="WRDS Financial Terminal Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize DataManager
@st.cache_resource
def get_manager():
    return DataManager()

dm = get_manager()

# Sidebar
st.sidebar.title("Financial Terminal Pro")

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

st.title(f"Make Money: {current_ticker}")

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
mask_p = (prices_df['date'] >= pd.to_datetime(start_date)) & (prices_df['date'] <= pd.to_datetime(end_date))
prices_filtered = prices_df.loc[mask_p].copy()

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
    current_price = prices_filtered.iloc[-1]['prc']
    if len(prices_filtered) > 1:
        start_p = prices_filtered.iloc[0]['prc']
        total_return = (current_price / start_p) - 1 if start_p != 0 else 0.0
        
        if 'ret' in prices_filtered.columns:
             volatility = prices_filtered['ret'].std() * (252**0.5)
        else:
             volatility = prices_filtered['prc'].pct_change().std() * (252**0.5)

# Layout
col1, col2 = st.columns([3, 1])

with col1:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Price & Performance", "Factor Analysis", "Peer Comparison", "Fundamentals & Valuation", "Estimates", "Ownership & Shorts"])

    with tab1:
        if not prices_filtered.empty:
            col_chart, col_metrics = st.columns([3, 1])
            with col_chart:
                 st.subheader("Price History")
                 fig = go.Figure()
                 fig.add_trace(go.Scatter(x=prices_filtered['date'], y=prices_filtered['prc'], mode='lines', name='Price'))
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

    with tab2:
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

    with tab3:
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
                
                # Display table
                display_cols = ['tic', 'datadate', 'atq', 'niq', 'ROE', 'EPS_Ann']
                st.dataframe(bulk_df[display_cols].sort_values('atq', ascending=False).head(20))
                
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

    with tab4:
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

    with tab5:
        if not est_df.empty:
            st.subheader("Analyst Estimates")
            st.dataframe(est_df)
        else:
            st.info("No Estimates Data Found")

    with tab6:
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
                latest_inst = inst_df.iloc[-1]
                st.metric("Total Inst. Shares", f"{latest_inst['total_shares']:,.0f}", f"Institutions: {latest_inst['num_institutions']}")
                
                fig_inst = go.Figure()
                fig_inst.add_trace(go.Scatter(x=inst_df['fdate'], y=inst_df['total_shares'], name='Shares Held'))
                fig_inst.update_layout(title="Institutional Shares Held", height=300)
                st.plotly_chart(fig_inst, use_container_width=True)
            else:
                st.info("No Institutional data available.")
        
        st.markdown("---")
        st.markdown("##### Insider Transactions")
        insider_df = dm.get_insider_transactions(current_ticker)
        if not insider_df.empty:
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
            st.dataframe(
                insider_df[['trandate', 'personid', 'rolecode1', 'acqdisp', 'shares', 'tprice', 'value']].head(50),
                use_container_width=True
            )
        else:
            st.info("No Insider Transaction data available.")

with col2:
    st.subheader("Snapshot")
    if not fund_df.empty and not prices_df.empty:
        last_price = prices_df.iloc[-1]['prc']
        shares = fund_df.iloc[0]['cshoq'] if 'cshoq' in fund_df.columns else 0
        mkt_cap = last_price * shares
        
        st.metric("Last Price", f"${last_price:,.2f}")
        st.metric("Mkt Cap", f"${mkt_cap:,.2f} M")
        
    st.markdown("---")
    st.markdown("**Data Status**")
    st.success("Connected to Local Cache")

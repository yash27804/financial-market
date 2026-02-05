import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Beta Calculator", layout="wide")

st.title("📊 Advanced Beta Calculator")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Enter Stock Ticker (e.g., INFY, RELIANCE):").upper()
with col2:
   
    tperiod = st.number_input("Time Period (Years):", min_value=1, value=1)

button = st.button("Calculate Beta")

if button:
    if not ticker:
        st.error("Please enter a ticker symbol.")
    else:
        try:
            symbol = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
            benchmark = "^NSEI"
            
            with st.spinner('Fetching data...'):
                stock_data = yf.download(symbol, period=f"{int(tperiod)}y", interval='1wk', auto_adjust=True)
                nifty_data = yf.download(benchmark, period=f"{int(tperiod)}y", interval='1wk', auto_adjust=True)

            
            if stock_data.empty:
                st.error(f"No data found for {symbol}. Check the ticker.")
                st.stop()

            
            df = pd.DataFrame()
            
           
            stock_close = stock_data['Close'] if 'Close' in stock_data.columns else stock_data.iloc[:, 0]
            nifty_close = nifty_data['Close'] if 'Close' in nifty_data.columns else nifty_data.iloc[:, 0]

            df['Stock_Price'] = stock_close
            df['Market_Price'] = nifty_close
            
            
            df['Stock_Return'] = df['Stock_Price'].pct_change() * 100
            df['Market_Return'] = df['Market_Price'].pct_change() * 100
            
            
            df.dropna(inplace=True)

           
            beta = np.polyfit(df['Market_Return'], df['Stock_Return'], 1)[0]

           
            st.success(f"Beta Calculated Successfully!")
            
           
            m1, m2, m3 = st.columns(3)
            m1.metric("Beta Value", f"{beta:.2f}")
            m1.caption(f"Sensitivity to Nifty")
            
            m2.metric("Correlation", f"{df['Stock_Return'].corr(df['Market_Return']):.2f}")
            m2.caption("Correlation Coeff (R)")
            
            # Interpretation
            if beta > 1:
                verdict = "High Volatility (Aggressive)"
                color = "red"
            elif beta < 1 and beta > 0:
                verdict = "Low Volatility (Defensive)"
                color = "green"
            else:
                verdict = "Inverse Correlation"
                color = "blue"
            
            st.subheader(f"Verdict: :{color}[{verdict}]")
            st.write(f"If Nifty moves by **1%**, {ticker} is expected to move by **{beta:.2f}%**.")

            # --- VISUALIZATION (Regression Plot) ---
            st.markdown("---")
            st.subheader("Regression Plot (Visualizing Beta)")
            
            fig = px.scatter(
                df, 
                x="Market_Return", 
                y="Stock_Return", 
                trendline="ols", # Ordinary Least Squares regression line
                title=f"Weekly Returns: {ticker} vs Nifty 50",
                labels={"Market_Return": "Nifty 50 Returns (%)", "Stock_Return": f"{ticker} Returns (%)"},
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("View Raw Data"):
                st.dataframe(df)

        except Exception as e:
            st.error(f"An error occurred: {e}")
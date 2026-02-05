import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import talib as ta
import numpy as np
from plotly.subplots import make_subplots
if 'cname' not in st.session_state:
    st.session_state.cname=''

if 'data' not in st.session_state:
    st.session_state.data=''

@st.cache_data(ttl=24*3600)
def fetch_data(cname):
    data=yf.download(f"{cname}.NS",period='3mo',interval='1d')
    data.columns=data.columns.get_level_values(0)
    data.index=pd.to_datetime(data.index,'%Y-%m-%d')
    data.index=data.index.sort_values(ascending=True)
    data['Obv']=ta.OBV(data['Close'],data['Volume'])
    st.dataframe(data)
    return data
@st.cache_data(ttl=24*3600)
def plot_chart(cname, cdata, buy):
    cdata['Signal'] = buy
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1, 
        row_heights=[0.7, 0.3],
        subplot_titles=(f"Price Chart: {cname}", 'On-Balance Volume (OBV)')
    )

    # 1. Price Candlesticks
    fig.add_trace(go.Candlestick(
        x=cdata.index,
        open=cdata['Open'],
        high=cdata['High'],
        low=cdata['Low'],
        close=cdata['Close'],
        name='Price'
    ), row=1, col=1)

    # 2. OBV Line 
    fig.add_trace(go.Scatter(
        x=cdata.index,
        y=cdata['Obv'],
        line=dict(color='blue', width=1.5),
        name='Obv'
    ), row=2, col=1)

   
    buy_signals = cdata[cdata['Signal'] == 'Buy']
   

    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            # Plot marker slightly below the 'Low' price for better visibility
            y=buy_signals['Low'] * 0.98, 
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='#00ff00', 
                line=dict(width=1, color='white')
            ),
            name='Buy Signal'
        ), row=1, col=1)

    # Layout Customization
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        yaxis=dict(side='right'), 
        yaxis2=dict(side='right')
    )
    
    st.plotly_chart(fig, use_container_width=True) 
@st.cache_data(ttl=24*3600)
def generate_signal(cdata):
    price=cdata.iloc[:,0]
    obv=cdata.iloc[:,-1]
    low=cdata.iloc[:,2]
    high=cdata.iloc[:,1]
    print(low)
    combined=pd.concat([price,obv,low],axis=1)
    # print(combined)
    # print(combined['Close'].size)
    combined.dropna(inplace=True)
    i=0
    while(i<combined['Close'].size):
        if(combined['Low'][i+1]<combined['Low'][i]):
            if(combined['Close'][i+1]>combined['Close'][i]):
                if(combined['Obv'][i+1]>combined['Obv'][i]):
                   combined.loc[combined.index[i+1],'Signal']='Buy'

        i+=1    
        if(i==combined['Close'].size-1):
            
            # st.dataframe(combined)
            return combined['Signal']
            break   

cname=st.text_input("Enter Stock Ticker:")
sbutton=st.button("Submit")
if sbutton:
    if cname:
        st.session_state.cname=cname
        data=fetch_data(st.session_state.cname)
        if(data.empty == False):
          st.session_state.data=data
          buy_signal=generate_signal(st.session_state.data)
          st.session_state.buy=buy_signal
          plot_chart(st.session_state.cname,st.session_state.data,st.session_state.buy)
          
    elif cname is None:
        st.error("Ticker is Empty")

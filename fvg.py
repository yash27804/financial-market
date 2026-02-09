import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
if 'cname' not in st.session_state:
    st.session_state.cname=''
if 'data' not in st.session_state:
    st.session_state.data=''

@st.cache_data(ttl=24*3600)
def fetch_data(cname):
    data=yf.download(f"{cname}.NS",period='1y',interval='1d')
    data.index=pd.to_datetime(data.index,'%Y-%m-%d')
    data.index=data.index.sort_values(ascending=True)
    data.columns=data.columns.get_level_values(0)
    data[['Open', 'Close', 'High', 'Low']] = data[['Open', 'Close', 'High', 'Low']].round(2)
    data.dropna(inplace=True)
    data=data[data['Volume']!=0]
    st.dataframe(data)
    return data    

@st.cache_data(ttl=24*3600)
def calc_fvg(data):
    df=data.copy()
    df['fvg_up_lvl']=pd.NA
    df['fvg_low_lvl']=pd.NA
    df['fvg_mid_lvl']=pd.NA
    df['fvg_type']=pd.NA

    limit=len(df)-1
    i=1
    while i < limit:
        cday=df.iloc[i,df.columns.get_loc('Close')]
        preday_close=df.iloc[i-1,df.columns.get_loc('Close')]
        preday_high=df.iloc[i-1,df.columns.get_loc('High')]
        preday_low=df.iloc[i-1,df.columns.get_loc('Low')]
        nexday_low=df.iloc[i+1,df.columns.get_loc('Low')]
        nexday_close=df.iloc[i+1,df.columns.get_loc('Close')]
        nexday_high=df.iloc[i+1,df.columns.get_loc('High')]
        if(nexday_close>cday and cday>preday_close):
            if(nexday_low>preday_high):
                df.iloc[i+1,df.columns.get_loc('fvg_up_lvl')]=nexday_low
                df.iloc[i+1,df.columns.get_loc('fvg_low_lvl')]=preday_high
                df.iloc[i+1,df.columns.get_loc('fvg_mid_lvl')]=(nexday_low+preday_high)/2
                df.iloc[i+1,df.columns.get_loc('fvg_type')]='Bull'
        elif(cday<preday_close and nexday_close<cday):
            if(nexday_high<preday_low):
                df.iloc[i+1,df.columns.get_loc('fvg_up_lvl')]=preday_low
                df.iloc[i+1,df.columns.get_loc('fvg_low_lvl')]=nexday_high
                df.iloc[i+1,df.columns.get_loc('fvg_mid_lvl')]=(preday_low+nexday_high)/2
                df.iloc[i+1,df.columns.get_loc('fvg_type')]='Bear'
        i+=1
    return df
@st.cache(ttl=24*3600)
def plot_chart(data,cname,fvg):
    st.subheader(f"Chart of {cname}")
    fig=make_subplots(rows=1,cols= 1)
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ),row=1,col=1)

    fvg_rows=fvg[fvg['fvg_type'].notna()]
    for index,row in fvg_rows.iterrows():
        color="green" if 'Bull' in str(row['fvg_type']) else 'red'
        loc_index=data.index.get_loc(index)
        start_date=data.index[max(0,loc_index-2)]
        fig.add_shape(
            type='rect',
            x0=start_date,
            y0=row['fvg_low_lvl'],
            x1=index,
            y1=row['fvg_up_lvl'],
            fillcolor=color,
            opacity=0.5,
            line=dict(width=0),
            layer="below"

        )
        fig.add_shape(
            type='line',
            x0=start_date,
            y0=row['fvg_mid_lvl'],
            x1=index,
            y1=row['fvg_mid_lvl'],
            line=dict
            (
                color=color,
                width=1,
                dash='dot'
            ),
            layer='below'
        )

    fig.update_layout(height=600,
                      xaxis_rangeslider_visible=True,
                      template='seaborn',
                      yaxis=dict(
                          fixedrange=False,
                          side='right'
                      ),
                      dragmode='pan'
                      )
    st.plotly_chart(fig,use_container_width=True,config={'scrollZoom':True})



cname=st.text_input("Enter Ticker Name:")
st.session_state.cname=cname
sbtn=st.button("Submit")
if sbtn:
    if cname:
        data=fetch_data(st.session_state.cname)
        if(data is not None):
            st.session_state.data=data
            fvg=calc_fvg(st.session_state.data)
            st.session_state.fvg=fvg
            plot_chart(st.session_state.data,st.session_state.cname,st.session_state.fvg)
        else:
            st.error("Unable to fetch Data")
      
    elif cname is None:
        st.error("Ticker Field is Empty")
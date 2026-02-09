import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')

if 'data' not in st.session_state:
    st.session_state.data = None
if 'ticker' not in st.session_state:
    st.session_state.ticker = ''
if 'smc' not in st.session_state:
    st.session_state.smc = None

keys = [
    'inducement_price', 'inducement_date',
    'll_price', 'll_date', 'll_type',
    'bos_price', 'bos_date',
    'choch_price', 'choch_date'
]
for key in keys:
    if key not in st.session_state:
        st.session_state[key] = None

@st.cache_data(ttl=24*3600)
def fetch_data(ticker,timeframe):
    period=''
    if(timeframe=='1h' or timeframe=='4h'):
        period='1y'
    if(timeframe=='1d' or timeframe=='1wk'):
        period='2y'
    data=yf.download(f"{ticker}.NS",interval=timeframe,period=period,auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.index=data.index.sort_values(ascending=True)
    
    data.index=pd.to_datetime(data.index).tz_localize(None) 
    data=data[data['Volume']!=0]
    data[['Open','Close','High','Low']]=data[['Open','Close','High','Low']].round(2)
    st.dataframe(data)
    return data

@st.cache_data(ttl=24*3600)
def find_smc(data):
    df = data.copy()
    
    df['is_inside_bar'] = (df['Low'] > df['Low'].shift(1)) & (df['High'] < df['High'].shift(1))
    df['mother_high'] = df['High'].where(~df['is_inside_bar']).ffill()
    df['swept_prev_high'] = df['High'] > df['mother_high'].shift(1)
    df['is_break_candle'] = df['Close'] < df['Low'].shift(1).rolling(window=3).min()
    df['any_recent_sweep'] = df['swept_prev_high'].rolling(window=5).max() == 1.0
    df['signal'] = df['is_break_candle'] & df['any_recent_sweep']
    
    df=confirmed_low_weak_low(df)
    temp_min=df['confirmed_ll'].ffill()
    df['active_ll']=temp_min.cummin()
    df['is_bos'] = (df['Close'] < df['active_ll']) & (df['active_ll'].notna())
    
    df=choch(df)
    return df

@st.cache_data(ttl=24*3600)
def confirmed_low_weak_low(df):
    data = df.copy()
    data['inducement_high'] = data['High'].rolling(window=5).max().where(data['signal']).ffill()
    
    starts = data[data['signal'] == True]
    data['inducement_sweep'] = (data['High'].where(~data['signal']) > data['inducement_high'].shift(1)).fillna(False)
    ends = data[data['inducement_sweep'] == True]

    last_sweep_date = pd.Timestamp.min
    data['confirmed_ll'] = np.nan
    data['weak_ll'] = np.nan
    data['major_confirmed_low']=np.nan
    last_min_confirmed_low=data['High'].max()
    for start_date, start_row in starts.iterrows():
        if start_date < last_sweep_date:
            continue
            
        loc = data.index.get_loc(start_date)
        start_search = max(0, loc - 4)
        window = data.iloc[start_search : loc + 1]
        ind_date = window['High'].idxmax()
        ind_val = window['High'].max()
     
        valid_ends = ends[(ends.index > start_date) & (ends['High'] > ind_val)]
        
        found_valid_sweep = False

        if not valid_ends.empty:
            for end_date, end_row in valid_ends.iterrows():
                
                newer_signals = starts[(starts.index > start_date) & (starts.index < end_date)]
                
                if not newer_signals.empty:
                    break 
                
                analysis_range = data.loc[ind_date : end_date]
                if not analysis_range.empty:
                    range_min = analysis_range['Low'].min()
                    range_min_date = analysis_range['Low'].idxmin()
                    data.loc[range_min_date,'confirmed_ll']=range_min 
                    last_sweep_date = end_date
                    found_valid_sweep = True
                    if(range_min < last_min_confirmed_low):
                        data.loc[range_min_date,'major_confirmed_low']=range_min
                        last_min_confirmed_low=range_min
                    break 

        if not found_valid_sweep:
            
            next_signals = starts[starts.index > start_date]
            if next_signals.empty and ind_date > last_sweep_date:
                active_range = data.loc[ind_date:]
                if not active_range.empty:
                    current_min = active_range['Low'].min()
                    weak_date = active_range['Low'].idxmin()
                    data.loc[weak_date, 'weak_ll'] = current_min

    return data

@st.cache_data(ttl=24*3600)
def filter_ind(df):
    inducement_price = None
    inducement_date = None
    
    if 'choch' in df.columns:
        choch_rows = df[df['choch'].notna()]
        if not choch_rows.empty:
            last_choch_price = choch_rows['choch'].iloc[-1]
            last_choch_date = choch_rows.index[-1]
            
            subsequent_data = df.loc[last_choch_date:]
            choch_broken = subsequent_data[subsequent_data['Close'] > last_choch_price]
            
            if not choch_broken.empty:
                first_break_date = choch_broken.index[0]
                bos_after_break = df[(df['is_bos'] == True) & (df.index > first_break_date)]
                
                if bos_after_break.empty:
                    return None, None

    df_filtered = df[df['signal'] == True]
    
    if not df_filtered.empty:
        last_inducement = df_filtered.index[-1]
        try:
            loc = df.index.get_loc(last_inducement)
            start_loc = max(0, loc - 4)
            candle_range = df.iloc[start_loc : loc + 1]
            inducement_price = candle_range['High'].max()
            inducement_date = candle_range['High'].idxmax()
         
        except KeyError:
            pass
    
    return inducement_price, inducement_date

@st.cache_data(ttl=24*3600)
def choch(df):
    data=df.copy()
    data['choch'] = np.nan 
    bos_filtered=df[df['is_bos']==True]
    if bos_filtered.empty:
        return data
        
    last_bos_date=bos_filtered.index[-1]
    confirmed_date_filter=df[df['confirmed_ll'].notna()]
    
    if confirmed_date_filter.empty:
        return data

    confirmed_date_filter['is_before_bos']=confirmed_date_filter.index<last_bos_date
    confirmed_date_filter=confirmed_date_filter[confirmed_date_filter['is_before_bos']==True]
    
    if confirmed_date_filter.empty:
        return data
        
    start_index=confirmed_date_filter.index[-1]
    candle_high_price=df.loc[start_index:last_bos_date]['High'].max()
    candle_high_date=df.loc[start_index:last_bos_date]['High'].idxmax()
    if pd.notna(candle_high_price) and pd.notna(candle_high_date):
        data.loc[candle_high_date,'choch']=candle_high_price
    return data

@st.cache_data(ttl=24*3600)
def filter_choch(df):
    if 'choch' not in df.columns:
        return None, None
        
    filtered_data=df[df['choch'].notnull()]
    if filtered_data.empty:
        return None, None
    choch_price=filtered_data['choch'].iloc[-1]
    if pd.isna(choch_price):
        return None,None
    choch_date=filtered_data.index[-1]
    if pd.isna(choch_date):
        return None,None
    
    return choch_price, choch_date
    
@st.cache_data(ttl=24*3600)
def filter_bos(df):
    filtered_bos=df[df['is_bos']==True]
    filtered_bos=filtered_bos.drop_duplicates(subset=['active_ll'],keep='first')
    confirmed_df = df[df['major_confirmed_low'].notna()]
    filter_bos_price = None
    filtered_bos_date = None
    
    if not filtered_bos.empty:
        prior_confirmed = confirmed_df[confirmed_df.index < filtered_bos.index[-1]]
        if not prior_confirmed.empty:
            filtered_bos_date=prior_confirmed.iloc[-1].name
            filter_bos_price=filtered_bos['active_ll'].iloc[-1]
            
    return filter_bos_price,filtered_bos_date

@st.cache_data(ttl=24*3600)
def filter_ll(df):
    confirmed_df = df[df['major_confirmed_low'].notna()]
    if confirmed_df.empty:
        weak_df = df[df['weak_ll'].notna()]
        if not weak_df.empty:
            return weak_df['weak_ll'].iloc[-1], weak_df.index[-1], "Weak"
        return None, None, None

    last_confirmed_ll = confirmed_df['major_confirmed_low'].iloc[-1]
    last_confirmed_date = confirmed_df.index[-1]

    bos_df = df[df['is_bos'] == True]
    bos_df=bos_df.drop_duplicates(subset=['active_ll'])
    is_broken = False
    last_bos_date = pd.Timestamp.min

    if not bos_df.empty:
        last_bos_date = bos_df.index[-1]
        if last_bos_date > last_confirmed_date:
            is_broken = True

    if is_broken:
        weak_df = df[(df['weak_ll'].notna()) & (df.index >= last_bos_date)]
        if not weak_df.empty:
            recent_data = df.loc[last_bos_date:]
            if not recent_data.empty:
                current_min = recent_data['Low'].min()
                current_min_date = recent_data['Low'].idxmin()
            
                if current_min < last_confirmed_ll:
                     return current_min, current_min_date, "Weak"

    return last_confirmed_ll, last_confirmed_date, "Confirmed"

@st.cache_data(ttl=24*3600)
def plot_smc(data, ticker, inducement_price, inducement_date, ll_price, ll_date, ll_type, bos_price, bos_date,choch_price,choch_date,timeframe):
    st.subheader(f"CandleStick Chart of {ticker}")
    
    if timeframe in ['1h','4h']:
        date_fmt = '%d %b %y, %H:%M'
    else:
        date_fmt = '%d %b %y'
        
    data['DateStr'] = data.index.strftime(date_fmt)
    
    def get_x_coord(raw_date):
        if raw_date is None: return None
        try:
            if isinstance(raw_date, str): return raw_date
            return raw_date.strftime(date_fmt)
        except:
            return None

    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(go.Candlestick(
        x=data['DateStr'],
        open=data['Open'], 
        high=data['High'], 
        low=data['Low'], 
        close=data['Close'], 
        name='Price'
    ))

    if inducement_date and inducement_price:
        x_val = get_x_coord(inducement_date)
        if x_val in data['DateStr'].values:
            
            future_data = data.loc[inducement_date:]
            if len(future_data) > 1:
                future_highs = future_data.iloc[1:]['High']
                is_swept = future_highs.max() > inducement_price
            else:
                is_swept = False
                
            label_text = f"IND (SWEPT): {inducement_price}" if is_swept else f"IND: {inducement_price}"

            fig.add_shape(type='line',
                          x0=x_val, 
                          x1=data['DateStr'].iloc[-1], 
                          y0=inducement_price, 
                          y1=inducement_price,
                          line=dict(color="Red", width=2, dash="dash"))
            
            fig.add_annotation(x=data['DateStr'].iloc[-1], 
                               y=inducement_price, 
                               text=label_text,
                               showarrow=False, 
                               xanchor="left",
                               xshift=10,
                               yshift=10, 
                               font=dict(color="Red", size=17))

    if ll_price and ll_date:
        x_val = get_x_coord(ll_date)
        color = "Green" if ll_type == "Confirmed" else "Orange"
        dash = "solid" if ll_type == "Confirmed" else "dot"
        
        if x_val in data['DateStr'].values:
            fig.add_shape(type='line', 
                          x0=x_val, 
                          x1=data['DateStr'].iloc[-1], 
                          y0=ll_price, 
                          y1=ll_price,
                          line=dict(color=color, width=2, dash=dash))
            
            fig.add_annotation(x=data['DateStr'].iloc[-1], 
                               y=ll_price, 
                               text=f"LL: {ll_price} ({ll_type})",
                               showarrow=False, 
                               xanchor="left",
                               xshift=10,
                               yshift=-10, 
                               font=dict(color=color, size=17))

    if choch_price and choch_date:
        x_val = get_x_coord(choch_date)
        if x_val in data['DateStr'].values:
            fig.add_shape(
                type='line',
                x0=x_val,
                y0=choch_price,
                x1=data['DateStr'].iloc[-1],
                y1=choch_price,
                line=dict(color='Brown',dash="solid",width=3),
            )
            fig.add_annotation(
                x=data['DateStr'].iloc[-1],
                y=choch_price,
                yanchor="top",
                yshift=20,
                text=f"CHocH Price:{choch_price}",
                xanchor="left",
                xshift=-10,
                showarrow=False,
                font=dict(color="Brown",size=17)
            )

    if bos_price and bos_date:
        try:
            bos_idx = data.index.get_loc(bos_date)
            x_start = data['DateStr'].iloc[bos_idx]
            
            fig.add_shape(type='line', 
                          x0=x_start, 
                          x1=data['DateStr'].iloc[-1], 
                          y0=bos_price, 
                          y1=bos_price,
                          line=dict(color="Blue", width=2, dash="solid"))
            
            fig.add_annotation(x=data['DateStr'].iloc[-1], 
                               y=bos_price, 
                               text=f"BOS:{bos_price}",
                               showarrow=False, 
                               xanchor="left", 
                               xshift=10, 
                               yshift=10, 
                               font=dict(color="Blue", size=17))
        except:
            pass

    fig.update_layout(height=700, 
                      template='seaborn', 
                      xaxis_rangeslider_visible=False,
                      yaxis=dict(fixedrange=False, side='right'), 
                      dragmode='pan', 
                      hovermode='x unified',
                      margin=dict(l=10, r=150, t=10, b=10)) 
    
    fig.update_xaxes(type="category",
                     categoryorder='trace', 
                     tickangle=-45,
                     nticks=20,
                     tickfont=dict(size=10))
    st.plotly_chart(fig, config={'scrollZoom': True}, use_container_width=True)

ticker=st.text_input("Enter ticker of Stock:")
tf=['1h','4h','1d','1wk']
timeframe=st.selectbox("Select TimeFrame",tf)
sbtn=st.button("Submit")
if(sbtn):
    if ticker and timeframe:
        data=fetch_data(ticker,timeframe)
        if(data is not None):
            st.session_state.ticker=ticker
            st.session_state.data=data
            st.session_state.timeframe=timeframe
            smc=find_smc(st.session_state.data)
            if(smc is not None):
                st.session_state.smc=smc
                
                ind,ind_date=filter_ind(st.session_state.smc)
                if(ind is not None and ind_date is not None):
                    st.session_state.inducement_price=ind
                    st.session_state.inducement_date=ind_date
                else:
                    st.session_state.inducement_price=None
                    st.session_state.inducement_date=None
                
                ll_price,ll_date,ll_type=filter_ll(st.session_state.smc)
                if ll_date and ll_price:
                    st.session_state.ll_price=ll_price
                    st.session_state.ll_date=ll_date
                    st.session_state.ll_type=ll_type
                else:
                    st.session_state.ll_price=None
                    st.session_state.ll_date=None
                    st.session_state.ll_type=None
                
                bos_price,bos_date=filter_bos(st.session_state.smc)
                if bos_price and bos_date:
                    st.session_state.bos_price=bos_price
                    st.session_state.bos_date=bos_date
                else:
                    st.session_state.bos_price=None
                    st.session_state.bos_date=None
                
                choch_price,choch_date=filter_choch(st.session_state.smc)
                if choch_price and choch_date:
                    st.session_state.choch_price=choch_price
                    st.session_state.choch_date=choch_date
                else:
                    st.session_state.choch_price=None
                    st.session_state.choch_date=None

                plot_smc(st.session_state.data,st.session_state.ticker,
                         st.session_state.inducement_price,st.session_state.inducement_date,
                         st.session_state.ll_price,st.session_state.ll_date,st.session_state.ll_type,
                         st.session_state.bos_price,st.session_state.bos_date,
                         st.session_state.choch_price,
                         st.session_state.choch_date,
                         st.session_state.timeframe)
            else:
                st.error("Invalid Ticker")

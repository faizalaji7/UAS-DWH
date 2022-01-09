import pandas as pd
import numpy as np
import streamlit as st
#st.set_page_config(layout="wide")
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import ipywidgets as wd
from IPython.display import display, clear_output
from ipywidgets import interactive, interactive
from datetime import datetime, date, time

st.title('Sales Forecasting Product Using Arima Method and Streamlit Library')

#Cleaning the csv before the analysis
df=pd.read_excel('SalesData.xlsx')
# df=pd.read_csv('dim_pelanggan.csv')

df=df.iloc[5:,:] #Selecting only the relevent rows
df.reset_index(drop=True, inplace=True) #reset the index

df.columns = df.iloc[0] #Renaming the columns

df=df.iloc[1:,] #
df['Amount']=df['Amount'].astype(float).round(3)
df=df[df['Amount']>0]
df.reset_index(drop=True, inplace=True) #reset the index


if st.checkbox('Preview Dataset'):
    data=df
if st.button('First 10 rows (Head)'):
    st.write(data.head(10))
if st.button('Last 10 rows (Tail)'):
    st.write(data.tail(10))

# Selecting the Country nad category for the anlysis
st.title('Input Data for Analysis')

Select_Country=st.selectbox('Select a Country', ('All Country', 'Australia', 'Austria', 'Belgium', 'Canada', 'Denmark', 'Finland', 'France', 'Germany', 'Hong Kong', 'Ireland', 'Italy', 'Japan','New Zealand', 'Norway', 'Philippines', 'Poland', 'Portugal', 'Russia', 'Singapore', 'South Africa', 'Spain', 'Sweden', 'Switzerland', 'UK', 'USA'))
Select_Category=st.selectbox('Select a Status Order', ('All Status', 'Cancelled', 'Disputed', 'In Process', 'On Hold', 'Resolved', 'Shipped'))

#Creating a function to get data based on selected Country and Status

def section(cntry, stats):
    cntry=str(cntry)
    stats=str(stats)
    if (cntry=='All Country') & (stats=='All Status'):
        sec=pd.DataFrame(df[df['Country'].isnull()!=True])
        sec.reset_index(drop=True, inplace=True)
        return sec
    if (cntry=='All Country') & (stats!='All Status'):
        sec=pd.DataFrame(df[df['Status']==stats])
        sec.reset_index(drop=True, inplace=True)
        return sec
    if (cntry=='All Country') & (stats!='All Status'):
        sec=pd.DataFrame(df[df['Country']==cntry])
        sec.reset_index(drop=True, inplace=True)
        return sec
    else:
        sec=pd.DataFrame(df[(df['Country']==cntry) & (df['Status']==stats)])
        sec.reset_index(drop=True, inplace=True)
        return sec


df_new=section(Select_Country,Select_Category)
data=df_new

st.title('{} & {} Sales Data'.format(Select_Country,Select_Category))
st.write(data.head(10))

st.title('Visualizing the Sales data for the selection')

df2=df_new[['Customer', 'Amount', 'Week', 'Country', 'Status']].groupby('Week').sum().reset_index()
df2[['Year','Week']]=df2['Week'].str.split('-', expand=True)
df2['Date'] = pd.to_datetime(df2.Week.astype(str)+df2.Year.astype(str).add('-1') ,format='%V%G-%u')
df2['Date'] = pd.to_datetime(df2['Date'])
df3=df2[['Date','Amount']]

#Visualizing the Weekly Sales data
fig=px.line(df3, x=df3['Date'], y=df3["Amount"])
fig.update_layout(showlegend=True, height=500, width=1000,title_text="Weekly Sales Data")


# Plot!
st.plotly_chart(fig, use_container_width=True)

#Visualizing the Monthly Sales data
y=df3.set_index('Date')
y=pd.DataFrame(y['Amount'].resample('M').sum())
y=y[y['Amount']>0]

fig1=px.line(y, x=y.index, y="Amount")
fig1.update_layout(showlegend=True, height=500, width=1000,title_text="Monthly Sales Data")


# Plot!
st.plotly_chart(fig1, use_container_width=True)

#Model
k=y.copy()
k=k[:-1]

#Auto Arima
import pmdarima as pm
model = pm.auto_arima(k['Amount'], start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)


st.title('Forecasting')

# Forecast
n_periods = 15
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(k.index[-1], periods = n_periods, freq='M')


# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

fig3=make_subplots(rows=1,cols=1, shared_xaxes=True)

fig3.append_trace(go.Scatter(x=k.index, y=k.Amount, name='Actual Sales'), row=1, col=1)
fig3.append_trace(go.Scatter(x=fc_series.index,y=fc_series,name='Forecasted Sales'), row=1, col=1)

# Update axis properties
fig3.update_yaxes(title_text='Sales', row=1, col=1)
fig3.update_xaxes(title_text='Years', row=1, col=1)

fig3.update_layout(showlegend=True,height=500, width=1500,title_text="Sales vs Forecast")


# Plot!
st.plotly_chart(fig3, use_container_width=True)

#Forecasting numbers

st.title('Next 12 month forecasts')

forecast=pd.DataFrame(fc_series)
forecast.reset_index(inplace=True)
forecast.columns=[['Date', 'Sales Forecast']]

fmt = "%d-%m-%Y"
styler = forecast.style.format(
    {
        "Date": lambda t: t.datetime.date,
    }
)
st.table(styler)

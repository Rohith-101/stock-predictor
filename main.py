import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import urllib.request

# App configuration
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="Stock Price Prediction App", layout="wide")
st.title("Stock Price Prediction App")


@st.cache_data
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        # Add a User-Agent header to mimic a browser
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req)
        table = pd.read_html(html)
        df = table[0]
        return df['Symbol'].sort_values().tolist()
    except Exception as e:
        st.error(f"Error fetching S&P 500 tickers: {e}")
        return []  # Return an empty list in case of error

with st.sidebar:
    st.header("About")
    st.write(
        "This app predicts stock prices using Facebook Prophet and displays technical indicators."
    )
    stocks = get_sp500_tickers()
    selected_stock = st.selectbox("Select stock for prediction", stocks)
    n_years = st.slider("Years of prediction:", 1, 4)
    target = st.selectbox("Forecast target", ["Close", "Open", "Volume"])

period = n_years * 365

@st.cache_data
def load_data(ticker):
    """Download and clean stock data from Yahoo Finance."""
    data = yf.download(ticker, START, TODAY, auto_adjust=True)
    data.reset_index(inplace=True)
    # Handle yfinance's column naming for single tickers
    cols = data.columns.tolist()
    if len(cols) >= 7 and all(c == ticker or c == 'Date' for c in cols[1:6]):
        data.columns = ['Index', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    elif len(cols) >= 6 and all(c == ticker or c == 'Date' for c in cols[1:5]):
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    elif (
        len(cols) >= 6
        and all(selected_stock in c for c in cols[1:6])
    ):
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    return data

data = load_data(selected_stock)

# Show company info
try:
    info = yf.Ticker(selected_stock).info
    st.sidebar.markdown(f"**Company:** {info.get('longName', 'N/A')}")
    st.sidebar.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
    st.sidebar.image(info.get("logo_url", ""), width=100)
except Exception:
    pass

# Technical indicators
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

st.subheader("Recent Stock Data")
st.write(data.tail())

def plot_raw_data():
    """Plot the historical Open and Close prices with technical indicators."""
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name="Open", line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name="SMA 50", line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], name="SMA 200", line=dict(color='red', dash='dot')))
    fig.update_layout(
        title="Historical Stock Prices & Technical Indicators",
        xaxis_rangeslider_visible=True,
        yaxis_title="Price",
        height=350  # Reduce the plot height here
    )
    st.plotly_chart(fig, width='stretch')

plot_raw_data()

# Prophet forecasting
if 'Date' in data.columns and target in data.columns:
    df_train = data[['Date', target]].copy()
    df_train = df_train.rename(columns={"Date": "ds", target: "y"})
    df_train['ds'] = pd.to_datetime(df_train['ds'])
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')

    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=period)
    try:
        # Potentially problematic code
        forecast = model.predict(future)
        st.write("Forecast successful!")
    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")

    st.subheader("Forecast Data")
    st.write(forecast.tail())

    # Download button
    csv = forecast.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")

    st.subheader("Forecast Plot")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1, width='stretch')  # Forecast plot stays full width

    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.write(fig2)
else:
    st.error("Data does not have the required columns for Prophet training.")
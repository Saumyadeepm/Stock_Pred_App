## Importing Libraries ##
import pandas as pd
import numpy as np
import streamlit as st

# Importing Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf

# For time stamps
from datetime import datetime

# For Creation & Evaluation of Model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model

# To Avoid Unnecessary Warning prompts
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
from yahoo_fin import stock_info as si
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
import pandas_ta as ta

from ta.volatility import BollingerBands
from ta.volatility import KeltnerChannel
import plotly.graph_objects as go
from plotly.graph_objs import Line

from plotly.subplots import make_subplots
import json
import locale
#locale.setlocale(locale.LC_ALL, 'en_US')

from GoogleNews import GoogleNews
googlenews = GoogleNews()
googlenews = GoogleNews(lang='en', region='IN')
googlenews = GoogleNews(encode='utf-8')
#---------------------------------------------------------------#
def main():
    st.title("Stock Closing Price Forcasting App")
    st.sidebar.title('Sidebar')
    st.sidebar.subheader("Options")
if __name__ == '__main__':
    main()
ticker = "https://raw.githubusercontent.com/Saumyadeepm/Dataset_For_Mini_Project/main/nasdaq_ticker_screener.csv"
tickers = pd.read_csv(ticker)

#Extracing Symbol column From Ticker Symbol Dataframe
t_symb = tickers['Symbol']
t_symb = t_symb.tolist()


#After Conversion it into List
print(type(t_symb))
t_symb = tuple(t_symb)






stocks = t_symb


selected_stock = st.selectbox('Select the Stock', stocks)

n_years = st.slider('Years of prediction:', 1, 30)

#Data Collection
def getStockDataset():
    global inp_stock  # Declaring stock_input as a Global Variable
    global end  # Declaring end date as a Global Variable
    global start  # Declaring start date as a Global Variable
    global s_date  # Declaring number of years of historical data to be fetched
    #        as a Global Variable

    ## Enter the name of the stock & Retrieving it
    inp_stock = selected_stock
    s_date = int(n_years)
    stock_m = [inp_stock]

    ## Setting up start and end date of The Selected Stock
    stock_m = [inp_stock]

    end = datetime.now()
    start = datetime(end.year - s_date, end.month, end.day)

    ## Downloading the Stock & Storing it in the variable Stock
    for df in stock_m:
        df = yf.download(df, start, end)
    global stock  # Declaring Stock as a Global Variable
    stock = pd.DataFrame(df)


getStockDataset()

if st.sidebar.checkbox("Display Data", False):
    st.subheader("{} Stock ".format(inp_stock))
    st.write(stock)


## Retrieving basic company information
yf_symbol = yf.Ticker(inp_stock)
# Print company info
company_info = yf_symbol.info
#company_info_obj = json.loads(employee_string)


if st.sidebar.checkbox("Display Information about {} Company".format(company_info["longName"]), False):
    st.subheader("Basic Information About {}".format(company_info["longName"]))
    st.write("Company Name: ", company_info["longName"])
    st.write()
    st.write("-" * 75)
    st.markdown("""About the Company : 
    
    """)
    st.write("             {}".format(company_info["longBusinessSummary"]))

    st.write()
    st.write("*** " * 15)
    st.write("Website: ", company_info["website"])
    st.write("Industry: ", company_info["industry"])
    st.write("Total revenue: ", locale.format_string("%d", company_info["totalRevenue"], grouping=True))
    st.write("Gross profits: ", locale.format_string("%d", company_info["grossProfits"], grouping=True))
    st.write("Revenue growth: ", company_info["revenueGrowth"])
    st.write("Earnings growth: ", company_info["earningsGrowth"])
    st.write("Profit margins: ", company_info["profitMargins"])
    st.write("Debt to equity: ", company_info["debtToEquity"])
    st.write("Return on equity: ", company_info["returnOnEquity"])
    st.write("Quick ratio: ", company_info["quickRatio"])
    st.write("Current price: ", company_info["currentPrice"])
    st.write("52-Week high: ", company_info["fiftyTwoWeekHigh"])
    st.write("52-Week low: ", company_info["fiftyTwoWeekLow"])
    st.write("-" * 50)

##### Visualizing Market Index
def get_biz_days_delta_date(start_date_str, delta_days):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = start_date + (delta_days * US_BUSINESS_DAY)
    end_date_str = datetime.strftime(end_date, "%Y-%m-%d")
    return end_date_str


def load_price_data(symbol, start_date, end_date):
    # Download data
    try:
        df = si.get_data(symbol, start_date=start_date, end_date=end_date, index_as_date=False)
        return df
    except:
        print('Error loading stock data for ' + symbol)
        return None

#  Plot market graph
def plot_market_graph(market_symbol, in_df):
    df = in_df.copy()

    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=['Close'])
    #  Plot close price
    fig.add_trace(go.Line(x=df.index, y=df['close'], line=dict(color="blue", width=1), name="Close"), row=1, col=1)

    fig.update_layout(
        title={'text': market_symbol, 'x': 0.5},
        autosize=True, )
    fig.update_yaxes(range=[0, 1000000000], secondary_y=True)
    fig.update_yaxes(visible=False, secondary_y=True)  # hide range slider
    #  Show graph
    st.plotly_chart(fig, use_container_width=True)

today = datetime.today()
today_str = today.strftime("%Y-%m-%d")
#  Get last year's worth of data
past_date_str = get_biz_days_delta_date(today_str, -251)
m = '^'+inp_stock
market_symbol = '^NDX'
market_df = load_price_data(market_symbol, past_date_str,today_str)


# Visualizing stock graph and technical indicators
def plot_stock_graph(symbol, in_df):
    df = in_df.copy()
    #  Calculate strategy indicators
    df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
    df['RSI'] = ta.rsi(df['close'], length=14)
    indicator_bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_mid'] = indicator_bb.bollinger_mavg()
    df['BB_high'] = indicator_bb.bollinger_hband()
    df['BB_low'] = indicator_bb.bollinger_lband()
    indicator_keltner = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=20)
    df['Keltner_mid'] = indicator_keltner.keltner_channel_mband()
    df['Keltner_high'] = indicator_keltner.keltner_channel_hband()
    df['Keltner_low'] = indicator_keltner.keltner_channel_lband()

    fig = make_subplots(rows=5, cols=1,
                        subplot_titles=['Close', 'MACD', 'RSI', 'Bollinger Bands', 'Keltner Channels', ])
    #  Plot close price
    fig.add_trace(go.Line(x=df.index, y=df['close'], line=dict(color="blue", width=1), name="Close"), row=1, col=1)
    # Plot MACD
    fig.add_trace(go.Line(x=df.index, y=df['MACD_12_26_9'], line=dict(color="#99b3ff", width=1), name="MACD"), row=2,
                  col=1)
    fig.add_trace(go.Line(x=df.index, y=df['MACDs_12_26_9'], line=dict(color="#ebab34", width=1), name="MACD"), row=2,
                  col=1)
    # Plot RSI
    fig.add_trace(go.Line(x=df.index, y=df['RSI'], line=dict(color="#99b3ff", width=1), name="RSI"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="#ebab34", width=1), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="#ebab34", width=1), row=3, col=1)
    # Plot Bollinger
    fig.add_trace(go.Line(x=df.index, y=df['close'], line=dict(color="blue", width=1), name="Close"), row=4, col=1)
    fig.add_trace(go.Line(x=df.index, y=df['BB_high'], line=dict(color="#ebab34", width=1), name="BB High"), row=4,
                  col=1)
    fig.add_trace(go.Line(x=df.index, y=df['BB_mid'], line=dict(color="#fac655", width=1), name="BB Mid"), row=4, col=1)
    fig.add_trace(go.Line(x=df.index, y=df['BB_low'], line=dict(color="#ebab34", width=1), name="BB Low"), row=4, col=1)
    # Plot Keltner
    fig.add_trace(go.Line(x=df.index, y=df['close'], line=dict(color="blue", width=1), name="Close"), row=5, col=1)
    fig.add_trace(go.Line(x=df.index, y=df['Keltner_high'], line=dict(color="#ebab34", width=1), name="Keltner High"),
                  row=5, col=1)
    fig.add_trace(go.Line(x=df.index, y=df['Keltner_mid'], line=dict(color="#fac655", width=1), name="Keltner Mid"),
                  row=5, col=1)
    fig.add_trace(go.Line(x=df.index, y=df['Keltner_low'], line=dict(color="#ebab34", width=1), name="Keltner Low"),
                  row=5, col=1)
    fig.update_layout(
        title={'text': symbol, 'x': 0.5},
        autosize=False, width=800, height=1600)
    fig.update_yaxes(range=[0, 1000000000], secondary_y=True)
    fig.update_yaxes(visible=False, secondary_y=True)  # hide range slider
    #  Show graph
    st.plotly_chart(fig, use_container_width=True)


df_info = ['NDX Closing Price', 'Technical Indicators', 'Descriptive Analysis', 'Adjusted Closing Price', 'Sales Volume',
           'Moving Average', 'Average Daily Return', 'Avg Daily Return Using Hist']


st.sidebar.subheader("Display Visualizations/ Analysis")
sdbar = st.sidebar.multiselect("Select:", df_info)

def plot_g():

    # Plot NDX Closing Price
    if 'NDX Closing Price' in sdbar:
        st.header("Visualization")
        st.subheader('  NDX Closing Price')
        plot_market_graph(market_symbol, market_df)

    #  Plot stock graph and TAs
    if 'Technical Indicators' in sdbar:
        st.header("Visualization")
        st.subheader('  Technical Indicators: ')
        df = load_price_data(inp_stock, past_date_str, today_str)
        plot_stock_graph(inp_stock, df)

plot_g()


if st.sidebar.checkbox("Retrieve News", False):
    # Retrieving and displaying NEWS about the selected stock with Google News

    googlenews.search(inp_stock)
    results = googlenews.results(sort=True)
    googlenews.clear()

    st.header("News on {} Stock".format(inp_stock))
    for result in results:
        st.write('TITLE: ', result['title'])
        st.write('DESC:  ', result['desc'])
        st.write('URL:   ', result['link'])
        st.write('-' * 20)

#------------------------------------------------------
## Data Cleaning/ Preparation
#____________________________________________________

#Retrieving Stock's Full Name from Stock's Ticker symbol
msft = yf.Ticker(inp_stock)
company_name = msft.info['longName']

# Adding Name of the company Column into the Dataframe
stock["Company_Name"] = company_name
stock = stock.replace({'\$':''}, regex = True) # To remove $ sign
stock = stock.astype({"Close": float})

if 'Descriptive Analysis' in sdbar:
    st.header("Detailed Description")
    # Description of The Dataset
    st.write("Description of The Dataset: ",stock.describe())
    st.write('')

    # Shape of the Dataset
    st.write("Shape of The Dataset:", stock.shape)
    st.write('')

    # Columns of the Dataset
    st.write("Columns in the Dataset: ", stock.columns)
    st.write('')



st.set_option('deprecation.showPyplotGlobalUse', False)
if 'Adjusted Closing Price' in sdbar:
    ## Closing Price of given Stock
    st.header("Historical View of Adjusted closing Price of {}".format(company_name))
    ## Closing Price of given Stock
    plt.figure(figsize=(15, 6))
    stock['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Adjusted Closing Price of {company_name} Stock")
    fig = plt.tight_layout()
    st.pyplot(fig)

if 'Sales Volume' in sdbar:
    st.header("Sales Volume of {}".format(company_name))
    ## Sales Volume of Given Stock

    plt.figure(figsize=(15, 6))
    stock['Volume'].plot()
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"Sales Volume of {company_name} Stock")
    fig = plt.tight_layout()
    st.pyplot(fig)

if 'Moving Average' in sdbar:
    st.header("Moving Average of {}".format(company_name))
    ma_day = [10, 20, 50]

    ## Calculating Moving Average of Given Stock
    for ma in ma_day:
        column_name = f"MA for {ma} days"
        stock[column_name] = stock['Adj Close'].rolling(ma).mean()

    ## Plotting Moving Average

    plt.figure(figsize=(15, 6))
    stock['Adj Close'].plot(legend=True)
    stock['MA for 10 days'].plot(legend=True)
    stock['MA for 20 days'].plot(legend=True)
    stock['MA for 50 days'].plot(legend=True)

    plt.title(company_name)
    fig = plt.tight_layout()
    st.pyplot(fig)

if 'Average Daily Return' in sdbar:
    st.header("Average Daily Return of {}".format(company_name))
    stock['Daily Return'] = stock['Adj Close'].pct_change()
    plt.figure(figsize=(15, 6))
    stock['Daily Return'].plot(legend=True, linestyle='--', marker='o')
    plt.title(company_name)
    fig = plt.tight_layout()
    st.pyplot(fig)

if 'Avg Daily Return Using Hist' in sdbar:
    st.header("Average Daily Return Using Histogram of {}".format(company_name))
    ## Visualization of Average Daily return on given Stock Using Histogram

    plt.figure(figsize=(12, 7))
    plt.plot()
    stock['Daily Return'].hist(bins=50)
    plt.ylabel('Daily Return')
    plt.title(company_name)
    fig = plt.tight_layout()
    st.pyplot(fig)

# Making Dataframe The Contains Closing Price of The Stock
### closing_df = pd.DataFrame(DataReader(inp_stock, 'yahoo', start, end)['Adj Close'])
closing_df = pd.DataFrame(stock['Adj Close'])
#display(closing_df)


####--------------------------------------------
######## MODEL TRAINING #######
####--------------------------------------------
if st.sidebar.checkbox("Apply LSTM",False):
    st.subheader("LSTM Prediction Model")

    ## Creating Dataframe that contains closing Price of Stock
    data = pd.DataFrame(stock['Close'])

    # Replacing NaN values to 0 if Exists
    data = data.fillna(0)

    for i in range(0, len(data)):
        data["Close"][i] = data["Close"][i]

    ##Converting the the values into arrays to avoid errors
    dataset = data.values

    # Fetching the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * .95))

    st.write("Length of the Dataset:", training_data_len)

    # Min-Max Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(dataset)


    #Training Dataset
    # Creating training Dataset
    # Creating scaled training Dataset

    train_data = scaled_data[0:int(training_data_len), :]

    # Spliting the dataset into x_train and y_train dataset
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            print(x_train)
            print(y_train)
            print()

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshaping the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    print("Shape of x_train: ", x_train.shape)
    print('Shape of Label tensor: ', y_train.shape)

    ## Creating the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    print(model.summary())
    # Compiling the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Training the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    ## Creating the testing dataset
    test_data = scaled_data[training_data_len - 60:, :]

    ## Creating the dataset x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    ## Converting the data to numpy array to avoid errors
    x_test = np.array(x_test)

    # Reshaping the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # model = load_model('StockClosingPricePred.h5')
    result = model.evaluate(x_test, y_test)

    #print("test loss, test acc:", result)
    # fecthing the models predicted Closing price
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)


#####Model Testing/ Using Model for Prediction
    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print("RMSE :", rmse)




    df_info1 = ['Closing Price', 'Predicted Closing Price', 'Display Prediction Data']

    st.sidebar.subheader("Display Visualizations")
    sdbar1 = st.sidebar.multiselect("Select:", df_info1)
    
    def plot_pred_closing_p():
        ## Plotting Using Matplotlib
        global valid
        train = data[:training_data_len]
        valid = data[training_data_len:]

        valid['Predictions'] = predictions
        st.header("Predicted Closing Price of {} Stock".format(company_name))
        plt.figure(figsize=(16, 6))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price in USD', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        fig = plt.tight_layout()
        st.pyplot(fig)
       
    def plot_closing_p():
        st.header("Closing Price of {} Stock".format(company_name))
        plt.figure(figsize=(15, 6))
        stock['Close'].plot()
        plt.ylabel('Close Price in USD', fontsize=18)
        plt.xlabel('Date', fontsize=18)
        plt.title(f"Closing Price of {company_name} Stock")
        fig = plt.tight_layout()
        st.pyplot(fig)
    
    if 'Closing Price' in sdbar1:
        plot_closing_p()
        

        #####
        

    if 'Predicted Closing Price' in sdbar1:
        # Plotting the Predicted Closing Prices & Compraing with Historical Data
        plot_pred_closing_p()
        ## Plotting Using Matplotlib
    if 'Display Prediction Data' in sdbar1:
        st.header("Predicted Closing Data of {} Stock".format(company_name))
        st.write(valid)




# Import necessary libraries
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from prophet import Prophet

# Load data
dateparse = lambda dates: datetime.strptime(dates, '%d-%m-%Y')
ldf_sales_data = pd.read_csv('sales.csv', parse_dates=['Date'], date_parser=dateparse)
ldf_store_data = pd.read_csv('store.csv')

ldf_training_dataset = pd.merge(ldf_store_data, ldf_sales_data, on='StoreID', how='inner')

# opened stores with zero sales
zero_sales = ldf_training_dataset[(ldf_training_dataset.StoreOpen != 0) & (ldf_training_dataset.Sales == 0)]
#removing stores that are opened with zero sales 
ldf_training_dataset = ldf_training_dataset[(ldf_training_dataset["StoreOpen"] != 0) & (ldf_training_dataset['Sales'] != 0)]



# data extraction
ldf_training_dataset['Year'] = ldf_training_dataset.Date.dt.year
ldf_training_dataset['Month'] = ldf_training_dataset.Date.dt.month
ldf_training_dataset['Day'] = ldf_training_dataset.Date.dt.day
ldf_training_dataset['WeekOfYear'] = ldf_training_dataset.Date.dt.isocalendar().week
# ldf_training_dataset['WeekOfYear'] = ldf_training_dataset.Date.dt.week

# adding new variable
ldf_training_dataset['SalePerCustomer'] = ldf_training_dataset['Sales']/ldf_training_dataset['Customers']

# create holidays dataframe
state_dates = ldf_training_dataset[(ldf_training_dataset.HolidayFlag == 'h1') | (ldf_training_dataset.HolidayFlag == 'h2') & (ldf_training_dataset.HolidayFlag == 'h3')].loc[:, 'Date'].values
school_dates = ldf_training_dataset[ldf_training_dataset.SchUnivClose == 1].loc[:, 'Date'].values

state = pd.DataFrame({'holiday': 'state_holiday',
                      'ds': pd.to_datetime(state_dates)})
school = pd.DataFrame({'holiday': 'school_holiday',
                      'ds': pd.to_datetime(school_dates)})
holidays = pd.concat((state, school))

# Initialize Dash
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Sales Prediction Dashboard"),
    
    # Dropdown for store selection
    dcc.Dropdown(
        id='store-dropdown',
        options=[
            {'label': f'Store {store}', 'value': store}
            for store in ldf_training_dataset['StoreID'].unique()
        ],
        value=1  # Default selected store
    ),

    # Input for the number of days to forecast
    dcc.Input(id='forecast-days', type='number', value=30, placeholder='Enter days for forecasting'),

    # Tabs for different sections of the dashboard
    dcc.Tabs([
        dcc.Tab(label='Sales Overview', children=[
            # Graph to display sales data
            dcc.Graph(id='sales-graph'),
        ]),
        dcc.Tab(label='Sales Forecast', children=[
            # Graph to display sales forecast
            dcc.Graph(id='forecast-graph'),
        ]),
        dcc.Tab(label='Exploratory Data Analysis', children=[
            # Add components for EDA here
            dcc.Graph(id='eda-chart-1'),  # Update the ID here
            dcc.Graph(id='eda-chart-2'),  # Update the ID here
            dcc.Graph(id='eda-chart-3'),  # Update the ID here
            dcc.Graph(id='eda-chart-4'),  # Update the ID here
            dcc.Graph(id='eda-chart-5'),  # Update the ID here
        ]),

        # Inside the "Insights" tab
dcc.Tab(label='Insights', children=[
    # Add components for insights here
    html.Div([
        html.H2('Key Insight:'),
        html.P("The following are insights from the analysis:"),

        html.H3('Data Extraction and Transformation:'),
        html.P("The data has been extracted and transformed with the following steps:"),
        html.P("- Extracted Year, Month, Day, and WeekOfYear from the Date column."),
        html.P("- Created a new variable 'SalePerCustomer' by dividing Sales by Customers."),
        html.P("- Converted all columns into numeric type except StoreType which is categorical."),
        html.Br(),

        html.H3('Basic Information'),
        dcc.Markdown(
            "```" +
            ldf_training_dataset['SalePerCustomer'].describe().to_markdown() +
            "```"
        ),
        html.P("On average customers spend about 9.50$ per day. Though there are days with Sales equal to zero"),

        html.H3('Opened Stores with Zero Sales:'),
        html.P("Details of open stores with zero sales:"),
        dcc.Markdown(
            "```" +
            zero_sales.to_markdown(index=False) +
            "```"
        ),


        html.H3('Sales Statistics by Locality Type:'),
        dcc.Markdown(
            "```" +
            ldf_training_dataset.groupby('LocalityType')[['Customers', 'Sales']].sum().reset_index().to_markdown(index=False) +
            "```"
        ),

        html.H3("Holidays"),
        dcc.Markdown(
            "```"
            + holidays.head().to_markdown()+
            "```"
            )

            ])
        ]),
    ]),
])

# Define callback functions to update graphs based on user input
@app.callback(
    [Output('sales-graph', 'figure'), Output('forecast-graph', 'figure')],
    [Input('store-dropdown', 'value'), Input('forecast-days', 'value')]
)


def update_graphs(selected_store, forecast_days):
    # Filter data for the selected store
    store_data = ldf_training_dataset[ldf_training_dataset.StoreID == selected_store]
    
    # Create a time series plot of sales
    sales_figure = px.line(store_data, x='Date', y='Sales', title='Sales Over Time', markers=True)
    
    # Perform time series forecasting using Prophet
    sales_data = store_data[['Date', 'Sales']]
    sales_data = sales_data.rename(columns={'Date': 'ds', 'Sales': 'y'})
    
    # Initialize and fit the Prophet model
    model = Prophet(interval_width=0.95)
    model.fit(sales_data)
    
    # Create a future dataframe for forecasting
    future = model.make_future_dataframe(periods=forecast_days)  # Forecast for the given duration
    
    # Generate forecasts
    forecast = model.predict(future)
    
    # Create a plot of sales forecasts
    forecast_figure = go.Figure()
    forecast_figure.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    forecast_figure.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='Lower Bound'))
    forecast_figure.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='Upper Bound'))
    forecast_figure.update_layout(title='Sales Forecast')
    
    return sales_figure, forecast_figure

# Add additional callback functions for EDA and insights here
@app.callback(
    [
        Output('eda-chart-1', 'figure'), 
        Output('eda-chart-2', 'figure'), 
        Output('eda-chart-3', 'figure'),
        Output('eda-chart-4', 'figure'),
        Output('eda-chart-5', 'figure'),
        ],
    [Input('store-dropdown', 'value')]
)

def update_eda_chart(selected_store):
    """ Updates the charts in the EDA tab based off selected store"""

    # Define callback function to create EDA charts using Plotly Express
    store_data = ldf_training_dataset[ldf_training_dataset.StoreID == selected_store]
    
    # Sales trends by Month and LocalityType
    fig1 = px.line(store_data, x='Month', y='Sales', color='LocalityType', facet_col='PromoFlag',
                   labels={'Sales': 'Sales Amount', 'Month': 'Month'},
                   title='Sales Trends by Month and LocalityType')
    
    # Sales trends by Month and DOW
    fig2 = px.line(store_data, x='Month', y='Sales', color='LocalityType', facet_row='LocalityType',
                   facet_col='DOW', labels={'Sales': 'Sales Amount', 'Month': 'Month'},
                   title='Sales Trends by Month and DOW')
    
    # SalePerCustomer trends by Month and LocalityType
    fig3 = px.line(store_data, x='Month', y='SalePerCustomer', color='LocalityType', facet_col='PromoFlag',
                   labels={'SalePerCustomer': 'Sale Per Customer', 'Month': 'Month'},
                   title='Sale Per Customer Trends by Month and LocalityType')
    # Sales trends by Month, PromoFlag, and LocalityType
    fig4 = px.line(
        store_data, x='Month', y='SalePerCustomer', color='LocalityType',
        facet_row='PromoFlag', #palette='plasma',
        labels={'SalePerCustomer': 'Sale Per Customer', 'Month': 'Month'},
        title='Sales Trends by Month, PromoFlag, and LocalityType'
    )
    # Sales trends by Month, DOW, and LocalityType
    fig5 = px.line(
        store_data, x='Month', y='Sales', color='LocalityType',
        facet_row='DOW', #palette='plasma',
        labels={'Sales': 'Sales Amount', 'Month': 'Month'},
        title='Sales Trends by Month, DOW, and LocalityType'
    )
    
    return fig1, fig2, fig3, fig4,fig5



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

# Import necessary libraries
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from prophet import Prophet

# Load data (Replace 'sales.csv' and 'store.csv' with your actual file paths)
dateparse = lambda dates: datetime.strptime(dates, '%d-%m-%Y')
ldf_sales_data = pd.read_csv('sales.csv', parse_dates=['Date'], date_parser=dateparse)
ldf_store_data = pd.read_csv('store.csv')

ldf_training_dataset = pd.merge(ldf_store_data, ldf_sales_data, on='StoreID', how='inner')

# opened stores with zero sales
zero_sales = ldf_training_dataset[(ldf_training_dataset.StoreOpen != 0) & (ldf_training_dataset.Sales == 0)]
# Removing stores that are opened with zero sales
ldf_training_dataset = ldf_training_dataset[(ldf_training_dataset["StoreOpen"] != 0) & (ldf_training_dataset['Sales'] != 0)]

# Data extraction
ldf_training_dataset['Year'] = ldf_training_dataset.Date.dt.year
ldf_training_dataset['Month'] = ldf_training_dataset.Date.dt.month
ldf_training_dataset['Day'] = ldf_training_dataset.Date.dt.day
ldf_training_dataset['WeekOfYear'] = ldf_training_dataset.Date.dt.isocalendar().week

# Adding a new variable
ldf_training_dataset['SalePerCustomer'] = ldf_training_dataset['Sales'] / ldf_training_dataset['Customers']

# Create holidays dataframe
state_dates = ldf_training_dataset[(ldf_training_dataset.HolidayFlag.isin(['h1', 'h2', 'h3']))].loc[:, 'Date'].values
school_dates = ldf_training_dataset[ldf_training_dataset.SchUnivClose == 1].loc[:, 'Date'].values

state = pd.DataFrame({'holiday': 'state_holiday',
                      'ds': pd.to_datetime(state_dates)})
school = pd.DataFrame({'holiday': 'school_holiday',
                       'ds': pd.to_datetime(school_dates)})
holidays = pd.concat((state, school))

# Initialize Dash
app = dash.Dash(__name__)

# Custom CSS styling
app.css.append_css({
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css'
})

# Define the layout of the dashboard with advanced styling
app.layout = html.Div([
    # Header with advanced styling
    html.Div([
        html.H1("A Project on Store Sales Prediction", style={'font-family': 'Arial', 'color': 'white'}),
        html.H1("", style={'font-family': 'Arial', 'color': 'white'}),
        html.P("", style={'font-family': 'Arial', 'color': '#999'}),
        html.P("Canigbobi@gmail.com", style={'font-family': 'Arial', 'color': '#999'}),
        html.P("Advanced Customization", style={'font-family': 'Arial', 'color': '#999'}),
    ], style={'background-color': '#007BFF', 'padding': '20px', 'text-align': 'center'}),
    
    # Dropdown for store selection with styling
    dcc.Dropdown(
        id='store-dropdown',
        options=[
            {'label': f'Store {store}', 'value': store}
            for store in ldf_training_dataset['StoreID'].unique()
        ],
        value=1,  # Default selected store
        style={'width': '50%', 'margin': '0 auto', 'font-family': 'Arial', 'background-color': '#f4f4f4', 'color': '#333'}
    ),

    # Input for the number of days to forecast with styling
    dcc.Input(
        id='forecast-days',
        type='number',
        value=30,
        placeholder='Enter days for forecasting',
        style={
            'width': '20%',  # Adjust the width as needed
            'margin': '10px auto',
            'font-family': 'Arial',
            'background-color': '#f4f4f4',  # Light gray background
            'color': '#333'
        }
    ),
    
    # Tabs for different sections of the dashboard with advanced styling
    dcc.Tabs([
        dcc.Tab(label='Sales Overview', children=[
            # Graph to display sales data
            dcc.Graph(id='sales-graph'),
        ], style={'background-color': 'lightblue'}),  # Set background color for this tab
        
        dcc.Tab(label='Sales Forecast', children=[
            # Graph to display sales forecast
            dcc.Graph(id='forecast-graph'),
        ], style={'background-color': 'lightgreen'}),  # Set background color for this tab
        
        dcc.Tab(label='Exploratory Data Analysis', children=[
            # Add components for EDA here
            dcc.Graph(id='eda-chart-1'),  # Update the ID here
            dcc.Graph(id='eda-chart-2'),  # Update the ID here
            dcc.Graph(id='eda-chart-3'),  # Update the ID here
            dcc.Graph(id='eda-chart-4'),  # Update the ID here
        ], style={'background-color': 'lightyellow'}),  # Set background color for this tab
        
        # Inside the "Insights" tab
        dcc.Tab(label='Insights', children=[
            # Add components for insights here
            html.Div([
                html.H2('Key Insight:', style={'margin': '20px 0'}),
                html.P("The following are insights from the analysis:", style={'margin': '10px 0'}),

                html.H3('Data Extraction and Transformation:', style={'margin': '20px 0'}),
                html.P("The data has been extracted and transformed with the following steps:",
                       style={'margin': '10px 0'}),
                html.P("- Extracted Year, Month, Day, and WeekOfYear from the Date column.",
                       style={'margin': '10px 0'}),
                html.P("- Created a new variable 'SalePerCustomer' by dividing Sales by Customers.",
                       style={'margin': '10px 0'}),
                html.P("- Converted all columns into numeric type except StoreType which is categorical.",
                       style={'margin': '10px 0'}),
                html.Br(),

                html.H3('Basic Information', style={'margin': '20px 0'}),
                dcc.Markdown(
                    "```" +
                    ldf_training_dataset['SalePerCustomer'].describe().to_markdown() +
                    "```"
                ),
                html.P("On average customers spend about 9.50$ per day. Though there are days with Sales equal to zero",
                       style={'margin': '10px 0'}),

                html.H3('Opened Stores with Zero Sales:', style={'margin': '20px 0'}),
                html.P("Details of open stores with zero sales:", style={'margin': '10px 0'}),
                dcc.Markdown(
                    "```" +
                    zero_sales.to_markdown(index=False) +
                    "```"
                ),

                html.H3('Sales Statistics by Locality Type:', style={'margin': '20px 0'}),
                dcc.Markdown(
                    "```" +
                    ldf_training_dataset.groupby('LocalityType')[['Customers', 'Sales']].sum().reset_index().to_markdown(index=False) +
                    "```"
                ),

                html.H3("Holidays", style={'margin': '20px 0'}),
                dcc.Markdown(
                    "```" +
                    holidays.head().to_markdown() +
                    "```"
                )

            ], style={'background-color': 'lightcoral'}),  # Set background color for this content
        ], style={'background-color': 'lightcoral'}),  # Set background color for this tab
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
    #sales_figure = px.line(store_data, x='Date', y='Sales', title='Sales Over Time', markers=True, marker_symbol='circle')
    #sales_figure = px.line(store_data, x='Date', y='Sales', title='Sales Over Time')
    # Create a time series plot of sales
    sales_figure = go.Figure()

    # Add the sales data as a line plot with markers
    sales_figure.add_trace(go.Scatter(x=store_data['Date'], y=store_data['Sales'], mode='lines+markers', name='Sales'))

    # Update the layout with a title
    sales_figure.update_layout(title='Sales Over Time')

    # You can also customize other layout properties as needed
    sales_figure.update_xaxes(title='Date')
    sales_figure.update_yaxes(title='Sales')

    # Optionally, you can customize marker properties like size and color
    sales_figure.update_traces(marker=dict(size=6, color='blue', symbol='circle'))

    # If you want to add additional traces or customize the figure further, you can do so here


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
    ],
    [Input('store-dropdown', 'value')]
)
def update_eda_chart(selected_store):
    """ Updates the charts in the EDA tab based on selected store"""

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
        facet_row='PromoFlag',  # palette='plasma',
        labels={'SalePerCustomer': 'Sale Per Customer', 'Month': 'Month'},
        title='Sales Trends by Month, PromoFlag, and LocalityType'
    )

    return fig1, fig2, fig3, fig4

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

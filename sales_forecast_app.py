import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sales_forecast import SalesForecastModel, ModelParameters

st.set_page_config(page_title="Enterprise Sales Forecast Model", layout="wide")

# Initialize model
model = SalesForecastModel()

# Sidebar - Model Assumptions
st.sidebar.title("Model Assumptions")
target_arr = st.sidebar.number_input("Target ARR ($)", value=2000000, step=100000, format="%d")
leads_per_rep = st.sidebar.number_input("Leads per Sales Rep", value=30, min_value=1, step=5)

# Segment Parameters
st.sidebar.title("Segment Parameters")
segments = ['Small', 'Small-Medium', 'Medium', 'Medium-Large', 'Large', 'Extra Large']

# Create expanders for each parameter type
with st.sidebar.expander("Close Rates (%)", expanded=False):
    close_rates = {}
    # Default rates
    default_rates = {
        'Small': 25.0,
        'Small-Medium': 25.0,
        'Medium': 15.0,
        'Medium-Large': 15.0,
        'Large': 10.0,
        'Extra Large': 10.0
    }
    for segment in segments:
        close_rates[segment] = st.number_input(
            f"{segment} Close Rate",
            value=default_rates[segment],
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            key=f"close_rate_{segment}"
        ) / 100.0

with st.sidebar.expander("Days to Close", expanded=False):
    days_to_close = {}
    default_close_days = {
        'Small': 90,
        'Small-Medium': 90,
        'Medium': 120,
        'Medium-Large': 120,
        'Large': 180,
        'Extra Large': 365
    }
    for segment in segments:
        days_to_close[segment] = st.number_input(
            f"{segment} Days",
            value=default_close_days[segment],
            min_value=30,
            step=30,
            key=f"days_close_{segment}"
        )

with st.sidebar.expander("Days to Implement", expanded=False):
    days_to_impl = {}
    default_impl_days = {
        'Small': 30,
        'Small-Medium': 60,
        'Medium': 90,
        'Medium-Large': 120,
        'Large': 180,
        'Extra Large': 270
    }
    for segment in segments:
        days_to_impl[segment] = st.number_input(
            f"{segment} Days",
            value=default_impl_days[segment],
            min_value=30,
            step=30,
            key=f"days_impl_{segment}"
        )

# Create parameters object
params = ModelParameters(
    target_arr=target_arr,
    leads_per_rep=leads_per_rep,
    close_rates=close_rates,
    days_to_close=days_to_close,
    days_to_impl=days_to_impl,
    segment_distribution={
        'Small': 0.30,
        'Small-Medium': 0.30,
        'Medium': 0.25,
        'Medium-Large': 0.15,
        'Large': 0.0,
        'Extra Large': 0.0
    }
)

# Calculate results
results = model.calculate_revenue(params)

# Dashboard
st.title("Enterprise Sales Forecast Dashboard")

# Key Metrics Row 1
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Target ARR", f"${target_arr:,.0f}")
with col2:
    st.metric("Total ARR", f"${results['Total_ARR'].sum():,.0f}")
with col3:
    st.metric("ARR Achievement", f"{(results['Total_ARR'].sum() / target_arr * 100):.1f}%")
with col4:
    max_pipeline_slots = results['Required_Pipeline_Slots'].max()
    required_reps = np.ceil(max_pipeline_slots / leads_per_rep)
    st.metric("Required Sales Reps", f"{required_reps:.0f}")

# Key Metrics Row 2
col1, col2, col3 = st.columns(3)
with col1:
    total_revenue = results['Total_Revenue'].sum()
    st.metric("2025 Revenue", f"${total_revenue:,.0f}")
with col2:
    total_impl = results['Total_Impl_Revenue'].sum()
    st.metric("Implementation Revenue", f"${total_impl:,.0f}")
with col3:
    st.metric("Total 2025 Revenue", f"${total_revenue + total_impl:,.0f}")

# Monthly Progress Charts
st.subheader("Monthly Progression")

# Prepare monthly data
monthly_data = []
for month in range(1, 13):
    monthly_data.append({
        'Month': month,
        'New_Deals': results[f'Month_{month}_Deals'].sum(),
        'Active_Deals': results[f'Month_{month}_Active_Deals'].sum(),
        'Pipeline_Slots': results[f'Month_{month}_Pipeline'].sum(),
        'New_ARR': results[f'Month_{month}_ARR'].sum(),
        'Revenue': results[f'Month_{month}_Revenue'].sum()
    })
monthly_df = pd.DataFrame(monthly_data)

# ARR and Revenue Chart
col1, col2 = st.columns(2)

with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=monthly_df['Month'], y=monthly_df['New_ARR'], name='New ARR'))
    fig1.add_trace(go.Line(x=monthly_df['Month'], y=monthly_df['Revenue'], name='Revenue'))
    fig1.update_layout(title='Monthly ARR and Revenue',
                      xaxis_title='Month',
                      yaxis_title='Amount ($)',
                      height=400)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=monthly_df['Month'], y=monthly_df['New_Deals'], name='New Deals'))
    fig2.add_trace(go.Line(x=monthly_df['Month'], 
                          y=monthly_df['Pipeline_Slots'], 
                          name='Pipeline Slots'))
    fig2.update_layout(title='Monthly Deals and Pipeline',
                      xaxis_title='Month',
                      yaxis_title='Count',
                      height=400)
    st.plotly_chart(fig2, use_container_width=True)

# Segment Analysis
st.subheader("Segment Analysis")

# Prepare segment data
segment_data = results[results['Total_ARR'] > 0].copy()
segment_data = segment_data[['Segment', 'Total_ARR', 'Total_Revenue', 'Total_Deals', 
                            'Required_Pipeline_Slots', 'Close_Rate', 'Days_to_Close']]

# Segment Charts
col1, col2 = st.columns(2)

with col1:
    fig3 = px.treemap(segment_data,
                      path=['Segment'],
                      values='Total_ARR',
                      title='ARR Distribution by Segment')
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    fig4 = px.bar(segment_data,
                  x='Segment',
                  y='Total_Deals',
                  title='Deals by Segment',
                  text='Total_Deals')
    fig4.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)

# Detailed Segment Metrics
st.subheader("Segment Details")
formatted_segment_data = segment_data.copy()
formatted_segment_data['Close_Rate'] = formatted_segment_data['Close_Rate'].map('{:.1%}'.format)
formatted_segment_data['Total_ARR'] = formatted_segment_data['Total_ARR'].map('${:,.0f}'.format)
formatted_segment_data['Total_Revenue'] = formatted_segment_data['Total_Revenue'].map('${:,.0f}'.format)
formatted_segment_data = formatted_segment_data.rename(columns={
    'Close_Rate': 'Close Rate',
    'Days_to_Close': 'Days to Close',
    'Total_ARR': 'Total ARR',
    'Total_Revenue': '2025 Revenue',
    'Total_Deals': 'Total Deals',
    'Required_Pipeline_Slots': 'Pipeline Slots'
})
st.dataframe(formatted_segment_data, use_container_width=True)

# Monthly Details Table
st.subheader("Monthly Details")
formatted_monthly = monthly_df.copy()
formatted_monthly['New_ARR'] = formatted_monthly['New_ARR'].map('${:,.0f}'.format)
formatted_monthly['Revenue'] = formatted_monthly['Revenue'].map('${:,.0f}'.format)
formatted_monthly = formatted_monthly.rename(columns={
    'New_Deals': 'New Deals',
    'Active_Deals': 'Active Deals',
    'Pipeline_Slots': 'Pipeline Slots',
    'New_ARR': 'New ARR',
    'Revenue': 'Monthly Revenue'
})
st.dataframe(formatted_monthly, use_container_width=True)

# Download section
st.subheader("Export Data")
col1, col2 = st.columns(2)

with col1:
    csv_monthly = monthly_df.to_csv(index=False)
    st.download_button(
        label="Download Monthly Data",
        data=csv_monthly,
        file_name="monthly_forecast.csv",
        mime="text/csv"
    )

with col2:
    csv_segment = segment_data.to_csv(index=False)
    st.download_button(
        label="Download Segment Data",
        data=csv_segment,
        file_name="segment_forecast.csv",
        mime="text/csv"
    )

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sales_forecast import SalesForecastModel, ModelParameters, LeadStatus, Lead
from datetime import date, datetime

st.set_page_config(page_title="Enterprise Sales Forecast Model", layout="wide")

# Initialize model
model = SalesForecastModel()

# Create tabs for different sections
tab1, tab2 = st.tabs(["Forecast Model", "Lead Management"])

with tab1:
    # Sidebar - Target Selection
    st.sidebar.title("Target Selection")
    target_type = st.sidebar.selectbox(
        "Target Type",
        options=['ARR', 'Deals'],
        key='target_type'
    )

    if target_type == 'ARR':
        target_arr = st.sidebar.number_input("Target ARR ($)", value=2000000, step=100000, format="%d")
        target_deals = {segment: 0 for segment in model.base_data['Segment']}
    else:
        target_arr = 0
        st.sidebar.subheader("Target Deals by Segment")
        target_deals = {}
        for segment in model.base_data['Segment']:
            target_deals[segment] = st.sidebar.number_input(
                f"{segment} Target Deals",
                value=0,
                min_value=0,
                step=1,
                key=f"target_deals_{segment}"
            )

    # Other Model Assumptions
    st.sidebar.title("Model Assumptions")
    leads_per_rep = st.sidebar.number_input("Leads per Sales Rep", value=30, min_value=1, step=5)

    # Date Range Selection
    st.sidebar.subheader("Forecast Period")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2025, 1, 1),
            min_value=date(2024, 1, 1),
            max_value=date(2026, 12, 31)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date(2025, 12, 31),
            min_value=date(2024, 1, 1),
            max_value=date(2026, 12, 31)
        )

    # Segment Parameters
    segments = model.base_data['Segment']

    # Create expanders for each parameter type
    with st.sidebar.expander("Close Rates (%)", expanded=False):
        close_rates = {}
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

    with st.sidebar.expander("ARR per Customer ($)", expanded=False):
        arr_per_customer = {}
        default_arr = {
            'Small': 31500,
            'Small-Medium': 55125,
            'Medium': 75700,
            'Medium-Large': 246750,
            'Large': 294625,
            'Extra Large': 367500
        }
        for segment in segments:
            arr_per_customer[segment] = st.number_input(
                f"{segment} ARR",
                value=default_arr[segment],
                min_value=0,
                step=1000,
                format="%d",
                key=f"arr_{segment}"
            )

    with st.sidebar.expander("Implementation Revenue ($)", expanded=False):
        implement_income = {}
        default_impl = {
            'Small': 2400,
            'Small-Medium': 4200,
            'Medium': 14000,
            'Medium-Large': 85000,
            'Large': 144375,
            'Extra Large': 240000
        }
        for segment in segments:
            implement_income[segment] = st.number_input(
                f"{segment} Implementation",
                value=default_impl[segment],
                min_value=0,
                step=1000,
                format="%d",
                key=f"impl_{segment}"
            )

    # Existing Pipeline Input
    with st.sidebar.expander("Existing Pipeline", expanded=False):
        existing_pipeline = {}
        for segment in segments:
            existing_pipeline[segment] = st.number_input(
                f"{segment} Existing Deals",
                value=0,
                min_value=0,
                step=1,
                key=f"existing_pipeline_{segment}"
            )

    # Create parameters object
    params = ModelParameters(
        target_type=target_type,
        target_arr=target_arr,
        target_deals=target_deals,
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
        } if target_type == 'ARR' else {segment: 1.0 for segment in segments},
        existing_pipeline=existing_pipeline,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        arr_per_customer=arr_per_customer,
        implement_income=implement_income
    )

    # Calculate results
    results = model.calculate_revenue(params)

    # Dashboard
    st.title("Enterprise Sales Forecast Dashboard")

    # Key Metrics Row 1
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if target_type == 'ARR':
            st.metric("Target ARR", f"${target_arr:,.0f}")
        else:
            st.metric("Total Target Deals", f"{sum(target_deals.values())}")
    with col2:
        st.metric("Total ARR", f"${results['Total_ARR'].sum():,.0f}")
    with col3:
        if target_type == 'ARR':
            st.metric("ARR Achievement", f"{(results['Total_ARR'].sum() / target_arr * 100):.1f}%")
        else:
            total_actual_deals = results['Total_Deals'].sum()
            total_target_deals = sum(target_deals.values())
            st.metric("Deals Achievement", 
                     f"{(total_actual_deals / total_target_deals * 100):.1f}%" 
                     if total_target_deals > 0 else "N/A")
    with col4:
        max_pipeline_slots = results['Required_Pipeline_Slots'].max()
        required_reps = np.ceil(max_pipeline_slots / leads_per_rep)
        st.metric("Required Sales Reps", f"{required_reps:.0f}")

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
            'Revenue': results[f'Month_{month}_Revenue'].sum(),
            'Leads_Required': results[f'Month_{month}_Leads_Required'].sum()
        })
    monthly_df = pd.DataFrame(monthly_data)

    # ARR/Deals and Revenue Charts
    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        if target_type == 'ARR':
            fig1.add_trace(go.Bar(x=monthly_df['Month'], y=monthly_df['New_ARR'], name='New ARR'))
            fig1.add_trace(go.Line(x=monthly_df['Month'], y=monthly_df['Revenue'], name='Revenue'))
            fig1.update_layout(title='Monthly ARR and Revenue',
                             xaxis_title='Month',
                             yaxis_title='Amount ($)',
                             height=400)
        else:
            fig1.add_trace(go.Bar(x=monthly_df['Month'], y=monthly_df['New_Deals'], name='New Deals'))
            fig1.add_trace(go.Line(x=monthly_df['Month'], y=monthly_df['Active_Deals'], name='Active Deals'))
            fig1.update_layout(title='Monthly Deal Progress',
                             xaxis_title='Month',
                             yaxis_title='Number of Deals',
                             height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=monthly_df['Month'], y=monthly_df['Leads_Required'], name='Required Leads'))
        fig2.add_trace(go.Line(x=monthly_df['Month'], 
                             y=monthly_df['Pipeline_Slots'], 
                             name='Pipeline Slots'))
        fig2.update_layout(title='Monthly Pipeline Requirements',
                         xaxis_title='Month',
                         yaxis_title='Count',
                         height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Required Leads Analysis
    st.subheader("Required New Leads by Segment")

    # Create leads heatmap data
    leads_data = []
    for segment in segments:
        segment_leads = []
        for month in range(1, 13):
            leads = np.ceil(results[results['Segment'] == segment][f'Month_{month}_Leads_Required'].iloc[0])
            segment_leads.append(leads)
        leads_data.append(segment_leads)

    fig_leads = go.Figure(data=go.Heatmap(
        z=leads_data,
        x=list(range(1, 13)),
        y=segments,
        text=np.array(leads_data).astype(int),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale="Blues",
    ))
    fig_leads.update_layout(
        title='Required New Leads by Segment and Month',
        xaxis_title='Month',
        yaxis_title='Segment',
        height=400
    )
    st.plotly_chart(fig_leads, use_container_width=True)

    # Segment Analysis
    st.subheader("Segment Analysis")

    # Prepare segment data
    segment_data = results[results['Total_ARR'] > 0].copy()
    segment_data = segment_data[['Segment', 'Total_ARR', 'Total_Revenue', 'Total_Deals', 
                               'Required_Pipeline_Slots', 'Close_Rate', 'Days_to_Close']]

    # Segment Charts
    col1, col2 = st.columns(2)

    with col1:
        if target_type == 'ARR':
            fig3 = px.treemap(segment_data,
                           path=['Segment'],
                           values='Total_ARR',
                           title='ARR Distribution by Segment',
                           custom_data=['Total_ARR'])
            fig3.update_traces(
                texttemplate="<b>%{label}</b><br>$%{customdata[0]:,.0f}",
                textposition="middle center"
            )
        else:
            fig3 = px.treemap(segment_data,
                           path=['Segment'],
                           values='Total_Deals',
                           title='Deals Distribution by Segment',
                           custom_data=['Total_Deals'])
            fig3.update_traces(
                texttemplate="<b>%{label}</b><br>%{customdata[0]} deals",
                textposition="middle center"
            )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

with col2:
        if target_type == 'ARR':
            fig4 = px.bar(segment_data,
                       x='Segment',
                       y='Total_ARR',
                       title='ARR by Segment',
                       text='Total_ARR')
            fig4.update_traces(texttemplate='${:,.0f}', textposition='outside')
        else:
            comparison_data = []
            for segment in segments:
                if segment in segment_data['Segment'].values:
                    actual = segment_data[segment_data['Segment'] == segment]['Total_Deals'].iloc[0]
                else:
                    actual = 0
                comparison_data.append({
                    'Segment': segment,
                    'Actual Deals': actual,
                    'Target Deals': target_deals[segment]
                })
            comparison_df = pd.DataFrame(comparison_data)
            
            fig4 = px.bar(comparison_df,
                       x='Segment',
                       y=['Actual Deals', 'Target Deals'],
                       title='Target vs Actual Deals by Segment',
                       barmode='group')
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

    # Detailed Tables
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
        'Revenue': 'Monthly Revenue',
        'Leads_Required': 'Leads Required'
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

# Lead Management Tab
with tab2:
    st.title("Lead Management")
    
    lead_tab1, lead_tab2, lead_tab3 = st.tabs(["Add Lead", "Lead Status", "Pipeline Analysis"])
    
    # Tab 1: Lead Entry
    with lead_tab1:
        with st.form("new_lead_form"):
            st.subheader("Add New Lead")
            
            col1, col2 = st.columns(2)
            with col1:
                lead_name = st.text_input("Lead Name")
                segment = st.selectbox("Segment", model.base_data['Segment'])
                status = st.selectbox("Status", [s.value for s in LeadStatus])
                
            with col2:
                date_entered = st.date_input("Date Entered", value=date.today())
                expected_close = st.date_input("Expected Close Date")
                arr_value = st.number_input("ARR Value ($)", min_value=0.0, step=1000.0)
                impl_value = st.number_input("Implementation Value ($)", min_value=0.0, step=1000.0)
            
            notes = st.text_area("Notes")
            
            if st.form_submit_button("Add Lead"):
                lead_id = f"LEAD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                new_lead = Lead(
                    lead_id=lead_id,
                    name=lead_name,
                    segment=segment,
                    date_entered=date_entered,
                    status=LeadStatus(status),
                    expected_close_date=expected_close,
                    arr_value=arr_value,
                    implementation_value=impl_value,
                    notes=notes
                )
                model.lead_tracker.add_lead(new_lead)
                st.success("Lead added successfully!")
    
    # Tab 2: Lead Status
    # Tab 2: Lead Status
    with lead_tab2:
        if len(model.lead_tracker.leads) > 0:
            leads_df = pd.DataFrame([vars(lead) for lead in model.lead_tracker.leads])
            lead_to_update = st.selectbox("Select Lead to Update", 
                                        leads_df['name'].tolist())
            lead_data = leads_df[leads_df['name'] == lead_to_update].iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                status_values = [s.value for s in LeadStatus]
                current_status_index = status_values.index(lead_data['status'])
                new_status = st.selectbox("New Status", 
                                        status_values,
                                        index=current_status_index)
            with col2:
                actual_close = st.date_input("Actual Close Date", 
                                           value=date.today(),
                                           disabled=new_status not in 
                                           ['won', 'lost', 'closed'])
            
            if st.button("Update Status"):
                model.lead_tracker.update_lead_status(
                    lead_data['lead_id'],
                    LeadStatus(new_status),
                    actual_close if new_status in [LeadStatus.WON.value,
                                                 LeadStatus.LOST.value,
                                                 LeadStatus.CLOSED.value] else None
                )
                st.success("Status updated successfully!")
            
            # Display leads table
            st.subheader("Lead Overview")
            display_df = leads_df.drop('lead_id', axis=1).copy()
            display_df['arr_value'] = display_df['arr_value'].map('${:,.0f}'.format)
            display_df['implementation_value'] = display_df['implementation_value'].map('${:,.0f}'.format)
            st.dataframe(display_df)
        else:
            st.info("No leads entered yet.")
    
    # Tab 3: Pipeline Analysis
    with lead_tab3:
        metrics = model.get_lead_analysis()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Leads", metrics['total_leads'])
        with col2:
            st.metric("Active Leads", metrics['active_leads'])
        with col3:
            st.metric("Won Deals", metrics['won_deals'])
        with col4:
            st.metric("Conversion Rate", f"{metrics['conversion_rate']:.1%}")
        
        # Additional metrics
        st.metric("Total Pipeline ARR", f"${metrics['total_pipeline_arr']:,.0f}")
        if metrics['avg_sales_cycle_days'] > 0:
            st.metric("Average Sales Cycle", f"{metrics['avg_sales_cycle_days']:.0f} days")
        
        if model.lead_tracker.leads:
            leads_df = pd.DataFrame([vars(lead) for lead in model.lead_tracker.leads])
            
            # Pipeline Status Chart
            status_counts = leads_df['status'].value_counts()
            fig_status = px.pie(values=status_counts.values, 
                              names=status_counts.index,
                              title='Lead Status Distribution')
            st.plotly_chart(fig_status)
            
            # Segment Distribution Chart
            segment_arr = leads_df.groupby('segment')['arr_value'].sum()
            fig_segment = px.bar(x=segment_arr.index, 
                               y=segment_arr.values,
                               title='Pipeline ARR by Segment',
                               labels={'x': 'Segment', 'y': 'ARR ($)'})
            fig_segment.update_traces(texttemplate='${:,.0f}', textposition='outside')
            st.plotly_chart(fig_segment)
            
            # Export lead data
            st.download_button(
                label="Download Lead Data",
                data=leads_df.to_csv(index=False),
                file_name="lead_data.csv",
                mime="text/csv"
            )
        else:
            st.info("No lead data available for analysis.")

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("Enterprise Sales Forecast Model")

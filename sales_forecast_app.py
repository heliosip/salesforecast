import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PIL import Image

# Portfolio size reference data
PORTFOLIO_SIZES = {
    'Small': 400,
    'Small-Medium': 700,
    'Medium': 1200,
    'Medium-Large': 5000,
    'Large': 7500,
    'Extra Large': 10000
}

def run_forecast(data, target_arr, leads_per_rep):
    df = pd.DataFrame(data)
    
    # Validate and calculate target ARR for each segment
    total_percentage = df['Revenue_in_Segment'].sum()
    if not np.isclose(total_percentage, 1.0, rtol=1e-3):
        st.warning(f"Revenue percentages sum to {total_percentage:.2%}. They should sum to 100%.")
    
    df['Target_Segment_ARR'] = df['Revenue_in_Segment'] * target_arr
    df['Annual_Customers_Needed'] = np.ceil(df['Target_Segment_ARR'] / df['ARR_per_Customer'])
    df['Monthly_Customers_Needed'] = np.ceil(df['Annual_Customers_Needed'] / 12)
    
    df['Monthly_Leads'] = np.ceil(df['Monthly_Customers_Needed'] / df['Close_Rate'])
    df['Monthly_Closed_Deals'] = np.ceil(df['Monthly_Leads'] * df['Close_Rate'])
    df['Monthly_Lost_Deals'] = df['Monthly_Leads'] - df['Monthly_Closed_Deals']
    df['Start_Month'] = np.ceil((df['Days_to_Close'] + df['Days_to_Impl']) / 30)
    df['Months_Active_in_2025'] = 13 - df['Start_Month']
    
    df['Monthly_ARR'] = df['Monthly_Closed_Deals'] * df['ARR_per_Customer']
    df['Monthly_Impl_Income'] = df['Monthly_Closed_Deals'] * df['Implement_Income']
    
    df['Active_Pipeline_Days'] = df['Days_to_Close']
    df['Pipeline_Slots_Needed'] = np.ceil(df['Monthly_Leads'] * df['Active_Pipeline_Days'] / 30)
    
    return df

def calculate_monthly_results(df):
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
             'July', 'August', 'September', 'October', 'November', 'December']
    
    segments = {}
    for _, row in df.iterrows():
        if row['Start_Month'] <= 12:
            segments[row['Segment']] = {
                'monthly_arr': row['Monthly_ARR'],
                'monthly_impl_income': row['Monthly_Impl_Income'],
                'start_month': int(row['Start_Month']),
                'closed_deals': int(row['Monthly_Closed_Deals']),
                'monthly_leads': int(row['Monthly_Leads']),
                'lost_deals': int(row['Monthly_Lost_Deals']),
                'days_to_close': int(row['Days_to_Close'])
            }
    
    monthly_results = []
    total_arr = 0
    segment_deals = {segment: [] for segment in segments.keys()}
    pipeline_metrics = []
    open_leads = 0
    
    # Track leads and their expected close months
    lead_tracking = []  # List of tuples (close_month, segment)
    
    for month_num, month in enumerate(months, 1):
        active_segments = []
        new_arr = 0
        monthly_revenue = 0
        impl_income = 0
        new_leads = 0
        
        # Add new leads for all segments starting in January
        for segment, details in segments.items():
            new_segment_leads = details['monthly_leads']
            new_leads += new_segment_leads
            
            # Calculate when these leads will close
            close_month = month_num + (details['days_to_close'] // 30)
            for _ in range(new_segment_leads):
                lead_tracking.append((close_month, segment))
        
        # Calculate closures for this month
        closing_leads = [lead for lead in lead_tracking if lead[0] == month_num]
        closed_leads = len(closing_leads)
        lost_leads = int(closed_leads * 0.9)  # 90% lost based on 10% close rate
        closed_leads = int(closed_leads * 0.1)  # 10% close rate
        
        # Process revenue for active segments
        for segment, details in segments.items():
            if month_num >= details['start_month']:
                # Only add to active segments and revenue when segment is active
                segment_deals[segment].append(details['closed_deals'])
                active_segments.append(f"{segment} ({details['closed_deals']})")
                
                # Calculate revenue metrics
                total_segment_deals = sum(segment_deals[segment])
                new_arr += details['monthly_arr']
                monthly_revenue += (total_segment_deals * details['monthly_arr'] / 12)
                impl_income += details['monthly_impl_income']
        
        # Update open leads
        open_leads = open_leads + new_leads - closed_leads - lost_leads
        
        # Update total ARR
        total_arr += new_arr
        
        monthly_results.append({
            'Month': month,
            'New_ARR': new_arr,
            'Total_ARR': total_arr,
            'Revenue_ARR': monthly_revenue,
            'Implementation_Revenue': impl_income,
            'Total_Revenue': monthly_revenue + impl_income,
            'Active_Segments': ', '.join(active_segments)
        })
        
        pipeline_metrics.append({
            'Month': month,
            'New Leads': new_leads,
            'Open Leads': open_leads,
            'Leads Closed': closed_leads,
            'Leads Lost': lost_leads
        })
    
    return monthly_results, segments, segment_deals, pipeline_metrics

def main():
    # Page config and logo
    st.set_page_config(page_title="Enterprise Sales Forecast Model", layout="wide")
    
    # Logo and title in a row
    st.image('logo.png', width=150)
    st.title('Enterprise Sales Forecast Model')

    # Sidebar inputs
    st.sidebar.header('Model Assumptions')
    target_arr = st.sidebar.number_input('Target ARR ($)', value=2000000, step=100000, format='%d',
                                       help="Annual Recurring Revenue target for the year")
    leads_per_rep = st.sidebar.number_input('Leads per Sales Rep', value=30, step=5,
                                          help="Maximum number of leads a sales rep can actively work")
    
    # Initial data
    default_data = {
        'Segment': ['Small', 'Small-Medium', 'Medium', 'Medium-Large', 'Large', 'Extra Large'],
        'ARR_per_Customer': [31500, 55125, 75700, 246750, 294625, 367500],
        'Implement_Income': [2400, 4200, 14000, 85000, 144375, 240000],
        'Revenue_in_Segment': [0.25, 0.25, 0.25, 0.15, 0.05, 0.05],
        'Days_to_Close': [90, 90, 120, 120, 180, 365],
        'Close_Rate': [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        'Days_to_Impl': [30, 60, 90, 120, 180, 270]
    }
    
    # Sidebar parameter inputs
    st.sidebar.subheader("Model Parameters")
    edited_data = default_data.copy()
    
    with st.sidebar.expander("Revenue Parameters", expanded=False):
        for i, segment in enumerate(default_data['Segment']):
            edited_data['ARR_per_Customer'][i] = st.number_input(
                f"{segment} ARR/Customer",
                value=default_data['ARR_per_Customer'][i],
                step=1000,
                format='%d',
                help=f"Annual Recurring Revenue per customer for {segment} segment"
            )
            edited_data['Implement_Income'][i] = st.number_input(
                f"{segment} Impl Income",
                value=default_data['Implement_Income'][i],
                step=1000,
                format='%d',
                help="One-time implementation revenue per customer"
            )
            edited_data['Revenue_in_Segment'][i] = st.number_input(
                f"{segment} Revenue %",
                value=default_data['Revenue_in_Segment'][i],
                step=0.01,
                format="%.2f",
                help="Percentage of total ARR target allocated to this segment"
            )
    
    with st.sidebar.expander("Timeline Parameters", expanded=False):
        for i, segment in enumerate(default_data['Segment']):
            edited_data['Days_to_Close'][i] = st.number_input(
                f"{segment} Days to Close",
                value=default_data['Days_to_Close'][i],
                step=5,
                format='%d',
                help="Average number of days to close a deal"
            )
            edited_data['Days_to_Impl'][i] = st.number_input(
                f"{segment} Days to Impl",
                value=default_data['Days_to_Impl'][i],
                step=5,
                format='%d',
                help="Average number of days for implementation"
            )
    
    with st.sidebar.expander("Sales Parameters", expanded=False):
        for i, segment in enumerate(default_data['Segment']):
            edited_data['Close_Rate'][i] = st.number_input(
                f"{segment} Close Rate",
                value=default_data['Close_Rate'][i],
                step=0.01,
                format="%.2f",
                help="Percentage of leads that convert to customers"
            )
    
    # Add Calculate button to sidebar
    calculate_button = st.sidebar.button('Calculate Forecast')
    
    # Create main tabs
    tabs = st.tabs(['Dashboard', 'Pipeline Analysis', 'Monthly Projections', 'Segment Analysis'])
    tab1, tab2, tab3, tab4 = tabs
    
    if calculate_button:
        # Run calculations
        df = run_forecast(edited_data, target_arr, leads_per_rep)
        monthly_results, segments, segment_deals, pipeline_metrics = calculate_monthly_results(df)
        
        with tab1:
            st.header('Dashboard')
            
            # Calculate key metrics
            final_month = monthly_results[-1]
            total_impl = sum(r['Implementation_Revenue'] for r in monthly_results)
            annual_revenue_arr = sum(r['Revenue_ARR'] for r in monthly_results)
            total_deals = sum(sum(deals) for deals in segment_deals.values())
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total ARR (Book of Business)", 
                    f"${final_month['Total_ARR']:,.0f}",
                    help="Total Annual Recurring Revenue at year end"
                )
            with col2:
                st.metric(
                    "Total Revenue", 
                    f"${(annual_revenue_arr + total_impl):,.0f}",
                    help="Total revenue including ARR and implementation income"
                )
            with col3:
                st.metric(
                    "Total Deals", 
                    f"{total_deals:,.0f}",
                    help="Total number of closed deals across all segments"
                )
            
            # Enhanced Portfolio Analysis table
            st.subheader('Portfolio Analysis by Segment')
            segment_data = []
            for segment in default_data['Segment']:
                total_deals = sum(segment_deals.get(segment, [0]))
                segment_arr = total_deals * edited_data['ARR_per_Customer'][edited_data['Segment'].index(segment)]
                segment_data.append({
                    'Segment': segment,
                    'Portfolio Size': PORTFOLIO_SIZES[segment],
                    'Active Customers': total_deals,
                    'Segment ARR': segment_arr
                })
            
            segment_df = pd.DataFrame(segment_data)
            st.dataframe(
                segment_df.style.format({
                    'Portfolio Size': '{:,.0f}',
                    'Active Customers': '{:,.0f}',
                    'Segment ARR': '${:,.0f}'
                }),
                use_container_width=True
            )
        
        with tab2:
            st.header('Pipeline Analysis')
            
            total_pipeline_slots = df['Pipeline_Slots_Needed'].sum()
            total_monthly_leads = df['Monthly_Leads'].sum()
            sales_reps_needed = np.ceil(total_pipeline_slots / leads_per_rep)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Monthly New Leads", 
                    f"{total_monthly_leads:.0f}",
                    help="Total new leads needed each month across all segments"
                )
            with col2:
                st.metric(
                    "Active Pipeline Slots", 
                    f"{total_pipeline_slots:.0f}",
                    help="Total number of leads being worked at any time"
                )
            with col3:
                st.metric(
                    "Sales Reps Needed", 
                    f"{sales_reps_needed:.0f}",
                    help=f"Number of sales reps needed based on {leads_per_rep} leads per rep"
                )
            
            st.subheader('Pipeline Requirements by Segment')
            pipeline_df = df[['Segment', 'Monthly_Leads', 'Days_to_Close', 'Pipeline_Slots_Needed']]
            st.dataframe(
                pipeline_df.style.format({
                    'Monthly_Leads': '{:.1f}',
                    'Pipeline_Slots_Needed': '{:.1f}'
                }),
                use_container_width=True
            )
            
            st.subheader('Monthly Pipeline Metrics')
            pipeline_metrics_df = pd.DataFrame(pipeline_metrics)
            st.dataframe(
                pipeline_metrics_df.style.format({
                    'New Leads': '{:.0f}',
                    'Open Leads': '{:.0f}',
                    'Leads Closed': '{:.0f}',
                    'Leads Lost': '{:.0f}'
                }),
                use_container_width=True
            )

        with tab3:
            st.subheader('Monthly Revenue Projections')
            monthly_df = pd.DataFrame(monthly_results)
            st.dataframe(
                monthly_df.style.format({
                    'New_ARR': '${:,.0f}',
                    'Total_ARR': '${:,.0f}',
                    'Revenue_ARR': '${:,.0f}',
                    'Implementation_Revenue': '${:,.0f}',
                    'Total_Revenue': '${:,.0f}'
                }),
                use_container_width=True
            )     
        with tab4:
            st.subheader('Segment Analysis')
            analysis_columns = [
                'Segment', 'Target_Segment_ARR', 'Annual_Customers_Needed',
                'Monthly_Leads', 'Monthly_Closed_Deals', 'Monthly_ARR',
                'Start_Month', 'Months_Active_in_2025'
            ]
            analysis_df = df[analysis_columns]
            
            st.dataframe(
                analysis_df.style.format({
                    'Target_Segment_ARR': '${:,.0f}',
                    'Annual_Customers_Needed': '{:.1f}',
                    'Monthly_Leads': '{:.1f}',
                    'Monthly_Closed_Deals': '{:.1f}',
                    'Monthly_ARR': '${:,.0f}',
                    'Start_Month': '{:.0f}',
                    'Months_Active_in_2025': '{:.0f}'
                }),
                use_container_width=True
            )
            
            # Add explanation of calculations
            with st.expander("Calculation Details", expanded=False):
                st.markdown("""
                ### How the numbers are calculated:
                
                1. **Target Segment ARR** = Total Target ARR × Revenue in Segment %
                2. **Annual Customers Needed** = Target Segment ARR ÷ ARR per Customer
                3. **Monthly Leads Required** = (Annual Customers ÷ 12) ÷ Close Rate
                4. **Monthly Closed Deals** = Monthly Leads × Close Rate
                5. **Monthly ARR** = Monthly Closed Deals × ARR per Customer
                6. **Start Month** = (Days to Close + Days to Implement) ÷ 30
                7. **Months Active** = 13 - Start Month
                
                Note: Values are rounded up to ensure sufficient pipeline coverage.
                """)

if __name__ == '__main__':
    main()
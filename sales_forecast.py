import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initial data setup with new values
data = {
    'Segment': ['Small', 'Small-Medium', 'Medium', 'Medium-Large', 'Large', 'Extra Large'],
    'ARR_per_Customer': [31500, 55125, 75700, 246750, 294625, 367500],
    'Implement_Income': [2400, 4200, 14000, 85000, 144375, 240000],
    'Revenue_in_Segment': [0.25, 0.25, 0.25, 0.15, 0.05, 0.05],
    'Days_to_Close': [90, 90, 120, 120, 180, 365],
    'Close_Rate': [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    'Days_to_Impl': [30, 60, 90, 120, 180, 270]
}

# Sales capacity parameters
LEADS_PER_REP = 30
target_arr = 1000000

# Create DataFrame and calculate derived values
df = pd.DataFrame(data)

# Calculate target customers needed first
df['Target_Segment_ARR'] = df['Revenue_in_Segment'] * target_arr
df['Annual_Customers_Needed'] = np.ceil(df['Target_Segment_ARR'] / df['ARR_per_Customer'])
df['Monthly_Customers_Needed'] = np.ceil(df['Annual_Customers_Needed'] / 12)

# Calculate required leads and deals
df['Monthly_Leads'] = np.ceil(df['Monthly_Customers_Needed'] / df['Close_Rate'])
df['Monthly_Closed_Deals'] = np.ceil(df['Monthly_Leads'] * df['Close_Rate'])
df['Start_Month'] = np.ceil((df['Days_to_Close'] + df['Days_to_Impl']) / 30)
df['Months_Active_in_2025'] = 13 - df['Start_Month']

# Calculate Monthly ARR and Implementation Income
df['Monthly_ARR'] = df['Monthly_Closed_Deals'] * df['ARR_per_Customer']
df['Monthly_Impl_Income'] = df['Monthly_Closed_Deals'] * df['Implement_Income']

# Calculate pipeline slots
df['Active_Pipeline_Days'] = df['Days_to_Close']
df['Pipeline_Turnover_Monthly'] = 30 / df['Active_Pipeline_Days']
df['Pipeline_Slots_Needed'] = np.ceil(df['Monthly_Leads'] * df['Active_Pipeline_Days'] / 30)

# Print Segment Analysis
print("\n=== SEGMENT CONTRIBUTION ANALYSIS ===")
print("-" * 120)
print(f"{'Segment':<15} {'Target ARR':>15} {'Monthly Leads':>15} {'Monthly Deals':>15} {'Monthly ARR':>20} {'Start Month':>15}")
print("-" * 120)

for _, row in df.iterrows():
    print(f"{row['Segment']:<15} ${row['Target_Segment_ARR']:>14,.2f} {row['Monthly_Leads']:>15.1f} " +
          f"{row['Monthly_Closed_Deals']:>15.0f} ${row['Monthly_ARR']:>19,.2f} {int(row['Start_Month']):>15}")

# Print Sales Capacity Analysis
print("\n=== SALES CAPACITY ANALYSIS ===")
print("-" * 100)
total_pipeline_slots = df['Pipeline_Slots_Needed'].sum()
total_monthly_leads = df['Monthly_Leads'].sum()
sales_reps_needed = np.ceil(total_pipeline_slots / LEADS_PER_REP)

print(f"Monthly New Leads Needed: {total_monthly_leads:.0f}")
print(f"Active Pipeline Slots: {total_pipeline_slots:.0f} (leads being worked at any time)")
print(f"Sales Reps Needed: {sales_reps_needed:.0f} (at {LEADS_PER_REP} active leads per rep)")

print("\nPipeline Details by Segment:")
print("-" * 100)
print(f"{'Segment':<15} {'Monthly Leads':>15} {'Days to Close':>15} {'Active Slots':>15}")
print("-" * 100)
for _, row in df.iterrows():
    print(f"{row['Segment']:<15} {row['Monthly_Leads']:>15.1f} {row['Days_to_Close']:>15.0f} {row['Pipeline_Slots_Needed']:>15.1f}")

# Calculate monthly revenue projections
months = ['January', 'February', 'March', 'April', 'May', 'June', 
         'July', 'August', 'September', 'October', 'November', 'December']
monthly_results = []

# Create segments dictionary for active segments
segments = {}
for _, row in df.iterrows():
    if row['Start_Month'] <= 12:
        segments[row['Segment']] = {
            'monthly_arr': row['Monthly_ARR'],
            'monthly_impl_income': row['Monthly_Impl_Income'],
            'start_month': int(row['Start_Month']),
            'closed_deals': int(row['Monthly_Closed_Deals']),
            'months_active': int(row['Months_Active_in_2025'])
        }

# Calculate monthly figures
total_arr = 0  # This is the full ARR (book of business)
segment_deals = {segment: [] for segment in segments.keys()}  # Track when each deal starts

for month_num, month in enumerate(months, 1):
    active_segments = []
    new_arr = 0
    monthly_revenue_arr = 0
    impl_income = 0
    
    for segment, details in segments.items():
        if month_num >= details['start_month']:
            # Add new deals this month to book of business
            new_arr += details['monthly_arr']
            
            # Track this month's new deals
            for _ in range(details['closed_deals']):
                segment_deals[segment].append(month_num)  # Record when each deal starts
            
            # Calculate revenue from all deals for this segment
            for deal_start_month in segment_deals[segment]:
                # Calculate each deal's contribution for this month
                months_active = month_num - deal_start_month + 1
                if months_active > 0:
                    monthly_revenue_arr += details['monthly_arr'] / 12  # Monthly portion of ARR
            
            # Add implementation revenue for new deals only
            impl_income += details['monthly_impl_income']
            active_segments.append(f"{segment} ({details['closed_deals']})")
    
    total_arr += new_arr
    
    monthly_results.append({
        'Month': month,
        'New_ARR': new_arr,
        'Total_ARR': total_arr,  # Full ARR (book of business)
        'Revenue_ARR': monthly_revenue_arr,  # This month's actual billable revenue
        'Implementation_Revenue': impl_income,
        'Total_Revenue': monthly_revenue_arr + impl_income,
        'Active_Segments': ', '.join(active_segments)
    })

# Print Monthly Revenue Projections
print("\n=== MONTHLY REVENUE PROJECTIONS 2025 ===")
print("-" * 125)
print(f"{'Month':<12} {'New ARR':>15} {'Total ARR':>15} {'Revenue ARR':>15} {'Impl Revenue':>15} {'Total Revenue':>20} {'Active Segments':>25}")
print("-" * 125)

annual_revenue_arr = 0
for result in monthly_results:
    print(f"{result['Month']:<12} ${result['New_ARR']:>14,.2f} ${result['Total_ARR']:>14,.2f} " +
          f"${result['Revenue_ARR']:>14,.2f} ${result['Implementation_Revenue']:>14,.2f} " +
          f"${result['Total_Revenue']:>19,.2f} {result['Active_Segments']:<25}")
    annual_revenue_arr += result['Revenue_ARR']

final_result = monthly_results[-1]
total_impl = sum(r['Implementation_Revenue'] for r in monthly_results)
print("-" * 125)
print(f"{'TOTAL 2025':<12} ${final_result['Total_ARR']:>14,.2f} ${annual_revenue_arr:>14,.2f} ${total_impl:>14,.2f} ${(annual_revenue_arr + total_impl):>19,.2f}")

# Print detailed segment timing
print("\nSegment Timing and Contribution Details:")
for segment, details in segments.items():
    months_remaining = 13 - details['start_month']
    annual_segment_arr = details['monthly_arr'] * months_remaining
    print(f"\n{segment}:")
    print(f"  Monthly ARR: ${details['monthly_arr']:,.2f}")
    print(f"  2025 Total ARR Contribution: ${annual_segment_arr:,.2f}")
    print(f"  Monthly Implementation Income: ${details['monthly_impl_income']:,.2f}")
    print(f"  Monthly Deals: {details['closed_deals']}")
    print(f"  Starts Month: {months[details['start_month']-1]}")
    print(f"  Months Active in 2025: {details['months_active']}")

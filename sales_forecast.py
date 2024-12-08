import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime

@dataclass
class ModelParameters:
    """Container for all model parameters"""
    target_type: str  # 'ARR' or 'Deals'
    target_arr: float
    target_deals: Dict[str, int]  # Deal targets by segment
    leads_per_rep: int
    close_rates: Dict[str, float]
    days_to_close: Dict[str, int]
    days_to_impl: Dict[str, int]
    segment_distribution: Dict[str, float]
    existing_pipeline: Dict[str, int]
    start_date: str
    end_date: str

class SalesForecastModel:
    """Sales forecast model with variable parameters"""
    
    def __init__(self):
        self.base_data = {
            'Segment': ['Small', 'Small-Medium', 'Medium', 'Medium-Large', 'Large', 'Extra Large'],
            'ARR_per_Customer': [31500, 55125, 75700, 246750, 294625, 367500],
            'Implement_Income': [2400, 4200, 14000, 85000, 144375, 240000],
        }
        self.df = pd.DataFrame(self.base_data)
    
    def optimize_deals(self, target_arr: float, arr_per_customer: float) -> int:
        """Calculate optimal deals to minimize difference from target"""
        if target_arr <= 0:
            return 0
        max_deals = np.ceil(target_arr / arr_per_customer)
        min_deals = np.floor(target_arr / arr_per_customer)
        
        max_arr = max_deals * arr_per_customer
        min_arr = min_deals * arr_per_customer
        
        return int(max_deals if abs(max_arr - target_arr) < abs(min_arr - target_arr) else min_deals)
    
    def distribute_deals(self, total_deals: int, num_months: int) -> List[int]:
        """Distribute deals evenly across months, handling remainders"""
        if num_months <= 0 or total_deals <= 0:
            return []
        
        base_deals = total_deals // num_months
        extra_deals = total_deals % num_months
        
        monthly_deals = [base_deals] * num_months
        for i in range(extra_deals):
            monthly_deals[i] += 1
            
        return monthly_deals
    
    def get_required_deals(self, df: pd.DataFrame, params: ModelParameters) -> pd.DataFrame:
        """Calculate required deals based on target type"""
        if params.target_type == 'ARR':
            df['Target_Segment_ARR'] = df['Revenue_in_Segment'] * params.target_arr
            df['Additional_Deals_Required'] = df.apply(
                lambda row: max(0, self.optimize_deals(
                    row['Target_Segment_ARR'] - (row['Existing_Pipeline'] * row['ARR_per_Customer']),
                    row['ARR_per_Customer']
                )) if row['Months_Active'] > 0 else 0,
                axis=1
            )
        else:  # 'Deals'
            df['Additional_Deals_Required'] = df.apply(
                lambda row: max(0, params.target_deals.get(row['Segment'], 0) - row['Existing_Pipeline'])
                if row['Months_Active'] > 0 else 0,
                axis=1
            )
            df['Target_Segment_ARR'] = df.apply(
                lambda row: params.target_deals.get(row['Segment'], 0) * row['ARR_per_Customer'],
                axis=1
            )
        return df
    
    def calculate_revenue(self, params: ModelParameters) -> pd.DataFrame:
        """Calculate revenue with given parameters"""
        df = self.df.copy()
        
        # Calculate number of months in date range
        start_date = pd.to_datetime(params.start_date)
        end_date = pd.to_datetime(params.end_date)
        total_months = ((end_date.year - start_date.year) * 12 + 
                       end_date.month - start_date.month + 1)
        
        # Apply parameters
        df['Revenue_in_Segment'] = df['Segment'].map(params.segment_distribution)
        df['Close_Rate'] = df['Segment'].map(params.close_rates)
        df['Days_to_Close'] = df['Segment'].map(params.days_to_close)
        df['Days_to_Impl'] = df['Segment'].map(params.days_to_impl)
        df['Existing_Pipeline'] = df['Segment'].map(params.existing_pipeline)
        
        # Calculate timing
        df['Start_Month'] = np.ceil((df['Days_to_Close'] + df['Days_to_Impl']) / 30)
        df['Months_Active'] = np.maximum(0, total_months + 1 - df['Start_Month'])
        
        # Get required deals based on target type
        df = self.get_required_deals(df, params)
        
        # Initialize monthly tracking
        months = range(1, total_months + 1)
        for month in months:
            df[f'Month_{month}_Deals'] = 0
            df[f'Month_{month}_Revenue'] = 0.0
            df[f'Month_{month}_Active_Deals'] = 0
            df[f'Month_{month}_ARR'] = 0.0
            df[f'Month_{month}_Pipeline'] = 0.0
            df[f'Month_{month}_Leads_Required'] = 0.0
        
        # Calculate monthly progression
        for idx, row in df.iterrows():
            if row['Months_Active'] > 0:
                start_month = int(row['Start_Month'])
                
                # Distribute additional deals across months
                monthly_deals = self.distribute_deals(
                    int(row['Additional_Deals_Required']),
                    int(row['Months_Active'])
                )
                
                # Handle existing pipeline
                if row['Existing_Pipeline'] > 0:
                    if len(monthly_deals) > 0:
                        monthly_deals[0] = monthly_deals[0] + row['Existing_Pipeline']
                    else:
                        monthly_deals = [row['Existing_Pipeline']]
                
                # Calculate monthly leads needed (excluding existing pipeline)
                if row['Additional_Deals_Required'] > 0:
                    monthly_leads = np.ceil(row['Additional_Deals_Required'] / row['Close_Rate'] / row['Months_Active'])
                    pipeline_slots = np.ceil(monthly_leads * row['Days_to_Close'] / 30)
                else:
                    monthly_leads = 0
                    pipeline_slots = 0
                
                # Process monthly deals and revenue
                active_deals = 0
                for i, deals in enumerate(monthly_deals):
                    month = start_month + i
                    if month <= len(months):
                        df.at[idx, f'Month_{month}_Deals'] = deals
                        active_deals += deals
                        df.at[idx, f'Month_{month}_Active_Deals'] = active_deals
                        df.at[idx, f'Month_{month}_ARR'] = deals * row['ARR_per_Customer']
                        df.at[idx, f'Month_{month}_Revenue'] = active_deals * (row['ARR_per_Customer'] / 12)
                        df.at[idx, f'Month_{month}_Pipeline'] = pipeline_slots
                        df.at[idx, f'Month_{month}_Leads_Required'] = monthly_leads if deals > row['Existing_Pipeline'] else 0
        
        # Calculate totals
        df['Total_ARR'] = df.apply(
            lambda row: (row['Additional_Deals_Required'] + row['Existing_Pipeline']) * row['ARR_per_Customer']
            if row['Months_Active'] > 0 else 0,
            axis=1
        )
        df['Total_Revenue'] = sum(df[f'Month_{month}_Revenue'] for month in months)
        df['Total_Deals'] = sum(df[f'Month_{month}_Deals'] for month in months)
        df['Total_Impl_Revenue'] = df['Total_Deals'] * df['Implement_Income']
        
        # Calculate max concurrent pipeline slots
        max_monthly_pipeline = max(
            sum(df[f'Month_{month}_Pipeline']) for month in months if month >= df['Start_Month'].min()
        )
        df['Required_Pipeline_Slots'] = max_monthly_pipeline
        
        return df

# Default parameters
default_params = ModelParameters(
    target_type='ARR',
    target_arr=2000000,
    target_deals={segment: 0 for segment in ['Small', 'Small-Medium', 'Medium', 'Medium-Large', 'Large', 'Extra Large']},
    leads_per_rep=30,
    close_rates={
        'Small': 0.25,
        'Small-Medium': 0.25,
        'Medium': 0.15,
        'Medium-Large': 0.15,
        'Large': 0.10,
        'Extra Large': 0.10
    },
    days_to_close={
        'Small': 90,
        'Small-Medium': 90,
        'Medium': 120,
        'Medium-Large': 120,
        'Large': 180,
        'Extra Large': 365
    },
    days_to_impl={
        'Small': 30,
        'Small-Medium': 60,
        'Medium': 90,
        'Medium-Large': 120,
        'Large': 180,
        'Extra Large': 270
    },
    segment_distribution={
        'Small': 0.30,
        'Small-Medium': 0.30,
        'Medium': 0.25,
        'Medium-Large': 0.15,
        'Large': 0.0,
        'Extra Large': 0.0
    },
    existing_pipeline={
        'Small': 0,
        'Small-Medium': 0,
        'Medium': 0,
        'Medium-Large': 0,
        'Large': 0,
        'Extra Large': 0
    },
    start_date='2025-01-01',
    end_date='2025-12-31'
)

if __name__ == "__main__":
    model = SalesForecastModel()
    results = model.calculate_revenue(default_params)
    print(f"Target ARR: ${default_params.target_arr:,.2f}")
    print(f"Total ARR: ${results['Total_ARR'].sum():,.2f}")
    print(f"2025 Revenue: ${results['Total_Revenue'].sum():,.2f}")
    print(f"Implementation Revenue: ${results['Total_Impl_Revenue'].sum():,.2f}")
    print(f"Total 2025 Revenue: ${results['Total_Revenue'].sum() + results['Total_Impl_Revenue'].sum():,.2f}")
    print(f"Required Sales Reps: {np.ceil(results['Required_Pipeline_Slots'].max() / default_params.leads_per_rep):.0f}")

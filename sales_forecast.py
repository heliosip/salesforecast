import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, date
from enum import Enum


class LeadStatus(Enum):
    NEW = "new"
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    CLOSED = "closed"


@dataclass
class Lead:
    lead_id: str
    name: str
    segment: str
    date_entered: date
    status: LeadStatus
    expected_close_date: date
    arr_value: float
    implementation_value: float
    actual_close_date: Optional[date] = None
    notes: Optional[str] = None


class LeadTracker:
    def __init__(self):
        self.leads: List[Lead] = []

    def add_lead(self, lead: Lead) -> None:
        """Add a new lead to the tracker"""
        self.leads.append(lead)

    def update_lead_status(
        self,
        lead_id: str,
        new_status: LeadStatus,
        actual_close_date: Optional[date] = None,
    ) -> None:
        """Update the status of an existing lead"""
        for lead in self.leads:
            if lead.lead_id == lead_id:
                lead.status = new_status
                if new_status in [LeadStatus.WON, LeadStatus.LOST, LeadStatus.CLOSED]:
                    lead.actual_close_date = actual_close_date or date.today()
                break

    def get_pipeline_forecast(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Generate pipeline forecast based on lead aging and status"""
        if not self.leads:
            return pd.DataFrame()

        leads_df = pd.DataFrame([vars(lead) for lead in self.leads])
        leads_df["date_entered"] = pd.to_datetime(leads_df["date_entered"])
        leads_df["expected_close_date"] = pd.to_datetime(
            leads_df["expected_close_date"]
        )

        pipeline_data = []
        date_range = pd.date_range(start=start_date, end=end_date, freq="M")

        for current_date in date_range:
            active_leads = leads_df[
                (
                    leads_df["status"].isin(
                        [LeadStatus.NEW.value, LeadStatus.PENDING.value]
                    )
                )
                & (leads_df["expected_close_date"] <= current_date)
            ]

            active_leads["age_days"] = (current_date - leads_df["date_entered"]).dt.days

            segment_summary = (
                active_leads.groupby("segment")
                .agg(
                    {
                        "arr_value": "sum",
                        "implementation_value": "sum",
                        "lead_id": "count",
                    }
                )
                .reset_index()
            )

            for _, row in segment_summary.iterrows():
                pipeline_data.append(
                    {
                        "date": current_date,
                        "segment": row["segment"],
                        "pipeline_arr": row["arr_value"],
                        "pipeline_impl": row["implementation_value"],
                        "lead_count": row["lead_id"],
                    }
                )

        return pd.DataFrame(pipeline_data)

    def get_lead_metrics(self) -> dict:
        """Calculate key metrics for lead analysis"""
        if not self.leads:
            return {
                "total_leads": 0,
                "active_leads": 0,
                "won_deals": 0,
                "conversion_rate": 0,
                "total_pipeline_arr": 0,
                "avg_sales_cycle_days": 0,
            }

        leads_df = pd.DataFrame([vars(lead) for lead in self.leads])

        total_closed = len(
            leads_df[
                leads_df["status"].isin(
                    [
                        LeadStatus.WON.value,
                        LeadStatus.LOST.value,
                        LeadStatus.CLOSED.value,
                    ]
                )
            ]
        )
        won_deals = len(leads_df[leads_df["status"] == LeadStatus.WON.value])

        metrics = {
            "total_leads": len(self.leads),
            "active_leads": len(
                leads_df[
                    leads_df["status"].isin(
                        [LeadStatus.NEW.value, LeadStatus.PENDING.value]
                    )
                ]
            ),
            "won_deals": won_deals,
            "conversion_rate": won_deals / total_closed if total_closed > 0 else 0,
            "total_pipeline_arr": leads_df[
                leads_df["status"].isin(
                    [LeadStatus.NEW.value, LeadStatus.PENDING.value]
                )
            ]["arr_value"].sum(),
            "avg_sales_cycle_days": (
                (
                    pd.to_datetime(
                        leads_df[leads_df["actual_close_date"].notna()][
                            "actual_close_date"
                        ]
                    )
                    - pd.to_datetime(
                        leads_df[leads_df["actual_close_date"].notna()]["date_entered"]
                    )
                )
                .mean()
                .days
                if not leads_df[leads_df["actual_close_date"].notna()].empty
                else 0
            ),
        }

        return metrics


@dataclass
class ModelParameters:
    """Container for all model parameters"""

    target_type: str  # 'ARR' or 'Deals'
    target_arr: float
    target_deals: Dict[str, int]
    leads_per_rep: int
    close_rates: Dict[str, float]
    days_to_close: Dict[str, int]
    days_to_impl: Dict[str, int]
    segment_distribution: Dict[str, float]
    existing_pipeline: Dict[str, int]
    start_date: str
    end_date: str
    arr_per_customer: Dict[str, float]
    implement_income: Dict[str, float]


class SalesForecastModel:
    """Sales forecast model with variable parameters"""

    def __init__(self):
        self.base_data = {
            "Segment": [
                "Small",
                "Small-Medium",
                "Medium",
                "Medium-Large",
                "Large",
                "Extra Large",
            ]
        }
        self.df = pd.DataFrame(self.base_data)
        self.lead_tracker = LeadTracker()

    def optimize_deals(self, target_arr: float, arr_per_customer: float) -> int:
        """Calculate optimal deals to minimize difference from target"""
        if target_arr <= 0:
            return 0
        max_deals = np.ceil(target_arr / arr_per_customer)
        min_deals = np.floor(target_arr / arr_per_customer)

        max_arr = max_deals * arr_per_customer
        min_arr = min_deals * arr_per_customer

        return int(
            max_deals
            if abs(max_arr - target_arr) < abs(min_arr - target_arr)
            else min_deals
        )

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

    def get_required_deals(
        self, df: pd.DataFrame, params: ModelParameters
    ) -> pd.DataFrame:
        """Calculate required deals based on target type"""
        if params.target_type == "ARR":
            df["Target_Segment_ARR"] = df["Revenue_in_Segment"] * params.target_arr
            df["Additional_Deals_Required"] = df.apply(
                lambda row: (
                    max(
                        0,
                        self.optimize_deals(
                            row["Target_Segment_ARR"]
                            - (row["Existing_Pipeline"] * row["ARR_per_Customer"]),
                            row["ARR_per_Customer"],
                        ),
                    )
                    if row["Months_Active"] > 0
                    else 0
                ),
                axis=1,
            )
        else:  # 'Deals'
            df["Additional_Deals_Required"] = df.apply(
                lambda row: (
                    max(
                        0,
                        params.target_deals.get(row["Segment"], 0)
                        - row["Existing_Pipeline"],
                    )
                    if row["Months_Active"] > 0
                    else 0
                ),
                axis=1,
            )
            df["Target_Segment_ARR"] = df.apply(
                lambda row: params.target_deals.get(row["Segment"], 0)
                * row["ARR_per_Customer"],
                axis=1,
            )
        return df

    def calculate_revenue(self, params: ModelParameters) -> pd.DataFrame:
        """Calculate revenue with given parameters"""
        # Calculate number of months in date range
        start_date = pd.to_datetime(params.start_date)
        end_date = pd.to_datetime(params.end_date)
        total_months = (
            (end_date.year - start_date.year) * 12
            + end_date.month
            - start_date.month
            + 1
        )
        months = range(1, total_months + 1)

        # Pre-initialize all columns
        initial_columns = {
            "Segment": self.df["Segment"],
            "ARR_per_Customer": self.df["Segment"].map(params.arr_per_customer),
            "Implement_Income": self.df["Segment"].map(params.implement_income),
            "Revenue_in_Segment": self.df["Segment"].map(params.segment_distribution),
            "Close_Rate": self.df["Segment"].map(params.close_rates),
            "Days_to_Close": self.df["Segment"].map(params.days_to_close),
            "Days_to_Impl": self.df["Segment"].map(params.days_to_impl),
            "Existing_Pipeline": self.df["Segment"].map(params.existing_pipeline),
        }

        # Add monthly columns
        for month in months:
            initial_columns.update(
                {
                    f"Month_{month}_Deals": np.zeros(len(self.df)),
                    f"Month_{month}_Revenue": np.zeros(len(self.df)),
                    f"Month_{month}_Active_Deals": np.zeros(len(self.df)),
                    f"Month_{month}_ARR": np.zeros(len(self.df)),
                    f"Month_{month}_Pipeline": np.zeros(len(self.df)),
                    f"Month_{month}_Leads_Required": np.zeros(len(self.df)),
                    f"Month_{month}_Pipeline_ARR": np.zeros(len(self.df)),
                    f"Month_{month}_Pipeline_Impl": np.zeros(len(self.df)),
                    f"Month_{month}_Pipeline_Leads": np.zeros(len(self.df)),
                }
            )

        # Create DataFrame with all columns
        df = pd.DataFrame(initial_columns)

        # Calculate timing
        df["Start_Month"] = np.ceil((df["Days_to_Close"] + df["Days_to_Impl"]) / 30)
        df["Months_Active"] = np.maximum(0, total_months + 1 - df["Start_Month"])

        # Get required deals based on target type
        df = self.get_required_deals(df, params)

        # Process monthly progression
        for idx, row in df.iterrows():
            if row["Months_Active"] > 0:
                start_month = int(row["Start_Month"])
                monthly_deals = self.distribute_deals(
                    int(row["Additional_Deals_Required"]), int(row["Months_Active"])
                )

                if row["Existing_Pipeline"] > 0:
                    if len(monthly_deals) > 0:
                        monthly_deals[0] = monthly_deals[0] + row["Existing_Pipeline"]
                    else:
                        monthly_deals = [row["Existing_Pipeline"]]

                if row["Additional_Deals_Required"] > 0:
                    monthly_leads = np.ceil(
                        row["Additional_Deals_Required"]
                        / row["Close_Rate"]
                        / row["Months_Active"]
                    )
                    pipeline_slots = np.ceil(monthly_leads * row["Days_to_Close"] / 30)
                else:
                    monthly_leads = 0
                    pipeline_slots = 0

                active_deals = 0
                for i, deals in enumerate(monthly_deals):
                    month = start_month + i
                    if month <= len(months):
                        df.at[idx, f"Month_{month}_Deals"] = deals
                        active_deals += deals
                        df.at[idx, f"Month_{month}_Active_Deals"] = active_deals
                        df.at[idx, f"Month_{month}_ARR"] = (
                            deals * row["ARR_per_Customer"]
                        )
                        df.at[idx, f"Month_{month}_Revenue"] = active_deals * (
                            row["ARR_per_Customer"] / 12
                        )
                        df.at[idx, f"Month_{month}_Pipeline"] = pipeline_slots
                        df.at[idx, f"Month_{month}_Leads_Required"] = (
                            monthly_leads if deals > row["Existing_Pipeline"] else 0
                        )

        # Add pipeline forecast
        pipeline_forecast = self.lead_tracker.get_pipeline_forecast(
            start_date, end_date
        )
        if not pipeline_forecast.empty:
            for idx, row in pipeline_forecast.iterrows():
                month = row["date"].month
                segment_mask = df["Segment"] == row["segment"]
                df.loc[segment_mask, f"Month_{month}_Pipeline_ARR"] = row[
                    "pipeline_arr"
                ]
                df.loc[segment_mask, f"Month_{month}_Pipeline_Impl"] = row[
                    "pipeline_impl"
                ]
                df.loc[segment_mask, f"Month_{month}_Pipeline_Leads"] = row[
                    "lead_count"
                ]

        # Calculate totals
        df["Total_ARR"] = df.apply(
            lambda row: (
                (row["Additional_Deals_Required"] + row["Existing_Pipeline"])
                * row["ARR_per_Customer"]
                if row["Months_Active"] > 0
                else 0
            ),
            axis=1,
        )
        df["Total_Revenue"] = sum(df[f"Month_{month}_Revenue"] for month in months)
        df["Total_Deals"] = sum(df[f"Month_{month}_Deals"] for month in months)
        df["Total_Impl_Revenue"] = df["Total_Deals"] * df["Implement_Income"]

        # Calculate max concurrent pipeline slots
        max_monthly_pipeline = max(
            sum(df[f"Month_{month}_Pipeline"])
            for month in months
            if month >= df["Start_Month"].min()
        )
        df["Required_Pipeline_Slots"] = max_monthly_pipeline

        return df

    def get_lead_analysis(self) -> dict:
        """Get comprehensive lead and pipeline analysis"""
        return self.lead_tracker.get_lead_metrics()


# Default parameters
default_params = ModelParameters(
    target_type="ARR",
    target_arr=2000000,
    target_deals={
        "Small": 0,
        "Small-Medium": 0,
        "Medium": 0,
        "Medium-Large": 0,
        "Large": 0,
        "Extra Large": 0,
    },
    leads_per_rep=30,
    close_rates={
        "Small": 0.25,
        "Small-Medium": 0.25,
        "Medium": 0.15,
        "Medium-Large": 0.15,
        "Large": 0.10,
        "Extra Large": 0.10,
    },
    days_to_close={
        "Small": 90,
        "Small-Medium": 90,
        "Medium": 120,
        "Medium-Large": 120,
        "Large": 180,
        "Extra Large": 365,
    },
    days_to_impl={
        "Small": 30,
        "Small-Medium": 60,
        "Medium": 90,
        "Medium-Large": 120,
        "Large": 180,
        "Extra Large": 270,
    },
    segment_distribution={
        "Small": 0.30,
        "Small-Medium": 0.30,
        "Medium": 0.25,
        "Medium-Large": 0.15,
        "Large": 0.0,
        "Extra Large": 0.0,
    },
    existing_pipeline={
        "Small": 0,
        "Small-Medium": 0,
        "Medium": 0,
        "Medium-Large": 0,
        "Large": 0,
        "Extra Large": 0,
    },
    arr_per_customer={
        "Small": 31500,
        "Small-Medium": 55125,
        "Medium": 75700,
        "Medium-Large": 246750,
        "Large": 294625,
        "Extra Large": 367500,
    },
    implement_income={
        "Small": 2400,
        "Small-Medium": 4200,
        "Medium": 14000,
        "Medium-Large": 85000,
        "Large": 144375,
        "Extra Large": 240000,
    },
    start_date="2025-01-01",
    end_date="2025-12-31",
)

if __name__ == "__main__":
    model = SalesForecastModel()
    results = model.calculate_revenue(default_params)
    print(f"Target ARR: ${default_params.target_arr:,.2f}")
    print(f"Total ARR: ${results['Total_ARR'].sum():,.2f}")
    print(f"2025 Revenue: ${results['Total_Revenue'].sum():,.2f}")
    print(f"Implementation Revenue: ${results['Total_Impl_Revenue'].sum():,.2f}")
    print(
        f"Total 2025 Revenue: ${results['Total_Revenue'].sum() + results['Total_Impl_Revenue'].sum():,.2f}"
    )
    print(
        f"Required Sales Reps: {np.ceil(results['Required_Pipeline_Slots'].max() / default_params.leads_per_rep):.0f}"
    )

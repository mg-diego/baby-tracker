import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.charts.breastfeeding_charts import BreastfeedingChart
from src.charts.diaper_charts import DiaperCharts
from src.charts.sleep_heatmap import SleepHeatmap
from src.data_loader import DataLoader

class BabyTrackerDashboard:
    def __init__(self):
        self.data_loader = DataLoader("data/huckelberry.csv")
        self.today = datetime.datetime.now()
        self.default_start = self.today - datetime.timedelta(days=7)
        self.default_end = self.today

    def render(self):
        st.set_page_config(page_title="👶 Baby Tracker Dashboard", layout="wide")
        st.title("👶 Baby Tracker Dashboard")

        start_date = st.date_input("Start Date", self.default_start)
        end_date = st.date_input("End Date", self.default_end)

        # Sleep
        st.subheader("🛏️ Sleep Heatmap")
        sleep_df = self.data_loader.filter_by_category("Sleep")
        SleepHeatmap(sleep_df).render()

        # Diapers
        st.subheader("🧷 Diaper Summary")
        diaper_df = self.data_loader.filter_by_category("Diaper")
        DiaperCharts(diaper_df).render(start_date, end_date)

        # Feeding
        st.subheader("🍼 Breastfeeding")
        feed_df = self.data_loader.filter_by_category("Feed")
        BreastfeedingChart(feed_df).render(start_date, end_date)

if __name__ == "__main__":
    app = BabyTrackerDashboard()
    app.render()
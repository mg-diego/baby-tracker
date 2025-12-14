import datetime
import streamlit as st
import streamlit_antd_components as sac

from data_manager import DataManager
from utils import Utils
from views.overview import OverviewSection
from views.growth import GrowthSection
from views.sleep import SleepSection
from views.feeding import FeedingSection
from views.diapers import DiaperSection

st.set_page_config(page_title="👶 Baby Tracker Dashboard", layout="wide")

class BabyTrackerApp:
    def __init__(self):
        self.today = datetime.datetime.now()
        self.default_start = self.today - datetime.timedelta(days=7)
        self.default_end = self.today
        self.ona_birthday = datetime.date(2025, 1, 15)        
        self.df = DataManager.load_main_data("data/export.csv")

    def run(self):
        st.title("👶 Baby Tracker Dashboard")

        # Sidebar Menú
        with st.sidebar:
            menu_id = sac.menu(
                items=[
                    sac.MenuItem('Overview', icon='bar-chart'),
                    sac.MenuItem('Growth', icon='line-chart'),            
                    sac.MenuItem('Sleep', icon='clock-circle'),
                    sac.MenuItem('Feeding', icon='coffee'),
                    sac.MenuItem('Diapers', icon='smile'),
                ],
                open_all=True, index=0
            )

        if self.df.empty:
            st.error("No data loaded. Please check 'data/export.csv'.")
            return

        # Header con info general
        last_update = self.df["Start"].max()
        st.markdown(f'<h3 style="color: gray;">Ona is {Utils.days_to_months_days(self.ona_birthday, datetime.date.today())} old.</h3>', unsafe_allow_html=True)
        st.markdown(f'<small style="color: gray;">(Last Update: {last_update})</small>', unsafe_allow_html=True)

        # Ruteo de Vistas
        if menu_id == 'Overview':
            OverviewSection(self.df).render()
        
        elif menu_id == 'Growth':
            GrowthSection(self.df).render()
        
        else:
            # Filtro de calendario común
            start, end = Utils.calendar_filter(self.default_start, self.default_end, self.today)
            
            if menu_id == 'Sleep':
                SleepSection(self.df, start, end).render()
            elif menu_id == 'Feeding':
                FeedingSection(self.df, start, end).render()
            elif menu_id == 'Diapers':
                DiaperSection(self.df, start, end).render()

if __name__ == "__main__":
    app = BabyTrackerApp()
    app.run()
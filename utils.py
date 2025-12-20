import datetime
import streamlit as st
from dateutil.relativedelta import relativedelta
from datetime import date

class Utils:
    
    @staticmethod
    def days_to_months_days(start_date, end_date):
        diff = relativedelta(end_date, start_date)
        
        parts = []
        if diff.months:
            parts.append(f"{diff.months} month{'s' if diff.months > 1 else ''}")
        if diff.days:
            parts.append(f"{diff.days} day{'s' if diff.days > 1 else ''}")
            
        return " and ".join(parts) if parts else "0 days"

    @staticmethod
    def calendar_filter(default_start, default_end, today):
        options = ['Last 7 days', 'Last 30 days', 'Last 90 days', 'All Time', 'Custom range']
        selection = st.selectbox("Select date range", options)

        if selection == 'Custom range':
            custom_range = st.date_input(
                "Select custom date range",
                value=(default_start, default_end),
                max_value=today
            )
            if len(custom_range) == 2:
                start_date, end_date = custom_range
            else:
                st.warning("Please select both start and end dates.")
                start_date, end_date = default_start, default_end
        else:
            if selection == 'All Time':
                days = 10000
            else:
                days = int(selection.split()[1])
            start_date = today - datetime.timedelta(days=days)
            end_date = today

        if selection != 'Custom range' and selection != 'All Time':
            st.info(f"📅 Selected date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        return start_date, end_date
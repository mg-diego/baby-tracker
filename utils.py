from datetime import timedelta
import pandas as pd
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
    # Añadimos el parámetro default_index (0 por defecto)
    def calendar_filter(default_start, default_end, today, key="default", default_index=1):
        
        # Nombre de la clave compartida para sincronizar todas las pestañas
        shared_state_key = "shared_calendar_selection"
        
        # Si NO existe una selección guardada en memoria, inicializamos con el default_index
        if shared_state_key not in st.session_state:
            st.session_state[shared_state_key] = default_index

        def on_change_handler():
            new_selection = st.session_state[key]
            options_list = ['Last 7 days', 'Last 30 days', 'Last 90 days', 'All Time', 'Custom range']
            if new_selection in options_list:
                st.session_state[shared_state_key] = options_list.index(new_selection)

        options = ['Last 7 days', 'Last 30 days', 'Last 90 days', 'All Time', 'Custom range']
        
        selection = st.selectbox(
            "Select date range", 
            options, 
            index=st.session_state[shared_state_key], # Lee del estado (que ya tiene tu default)
            key=key, 
            on_change=on_change_handler
        )

        if selection == 'Custom range':
            custom_range = st.date_input(
                "Select custom date range",
                value=(default_start, default_end),
                max_value=today,
                key=f"date_input_{key}" 
            )
            if len(custom_range) == 2:
                start_date, end_date = custom_range
            else:
                st.warning("Please select both start and end dates.")
                start_date, end_date = default_start, default_end
        else:
            if selection == 'All Time':
                days = 1000 
            else:
                days = int(selection.split()[1])
            start_date = today - timedelta(days=days)
            end_date = today

        if selection != 'Custom range' and selection != 'All Time':
            st.info(f"📅 Selected: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        return pd.to_datetime(start_date), pd.to_datetime(end_date)
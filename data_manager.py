import pandas as pd
import streamlit as st

class DataManager:
    
    @staticmethod
    @st.cache_data
    def load_main_data(filepath: str):
        try:
            df = pd.read_csv(filepath, sep=",")
            # Optimización: Convertir fechas una sola vez al cargar
            if 'Start' in df.columns:
                df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
            if 'End' in df.columns:
                df['End'] = pd.to_datetime(df['End'], errors='coerce')
            return df
        except FileNotFoundError:
            st.error(f"File not found: {filepath}")
            return pd.DataFrame()

    @staticmethod
    @st.cache_data
    def load_growth_percentiles(filepath: str):
        try:
            return pd.read_csv(filepath)
        except FileNotFoundError:
            st.error(f"Percentile file not found: {filepath}")
            return pd.DataFrame()

    @staticmethod
    def filter_by_category(df, category):
        # Si 'category' es una lista o tupla, usamos .isin()
        if isinstance(category, (list, tuple)):
            return df[df['Type'].isin(category)].copy()
        # Si es un string normal, usamos ==
        return df[df['Type'] == category].copy()

    @staticmethod
    def filter_by_date_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
        mask = (df['Start'].dt.date >= pd.to_datetime(start_date).date()) & \
               (df['Start'].dt.date <= pd.to_datetime(end_date).date())
        return df[mask]
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from data_manager import DataManager

class DiaperSection:
    def __init__(self, df, start_date, end_date):
        self.df = df
        self.start_date = start_date
        self.end_date = end_date

    def render(self):
        st.header("🧷 Diaper Changes")
        st.write("Monitor output frequency and patterns.")

        # 1. KPIs Rápidos (Solo datos calculados)
        self._render_kpis()

        tab1, tab2 = st.tabs(["Overview & Trends", "💩 Poop Patterns"])

        with tab1:
            st.subheader("Daily Composition")
            st.caption("Breakdown of diaper types per day.")
            self._daily_composition_chart()
            
            st.divider()
            
            st.subheader("Hydration Tracking (Wet Diapers)")
            st.caption("Tracking daily count of Wet + Mixed diapers.")
            self._pee_trend_chart()

        with tab2:
            st.subheader("Time of Day Distribution")
            st.caption("Hourly distribution of dirty diapers.")
            self._hourly_distribution_chart()
            
            st.divider()
            
            st.subheader("Poop Timeline")
            st.caption("Visualizing regularity. Each dot is a dirty diaper.")
            self._poop_scatter_chart()

    def _get_diaper_data(self):
        """Helper para limpiar y clasificar pañales."""
        df = DataManager.filter_by_category(self.df, 'Diaper')
        df = DataManager.filter_by_date_range(df, self.start_date, self.end_date)
        
        if df.empty: return pd.DataFrame()

        df = df.copy()
        
        # Concatenamos columnas de texto relevantes para buscar keywords
        cols_to_search = [col for col in ['End Condition', 'Notes', 'Status'] if col in df.columns]
        
        def classify_diaper(row):
            text = " ".join([str(row[c]).lower() for c in cols_to_search])
            
            has_pee = 'pee' in text or 'wet' in text
            has_poo = 'poo' in text or 'dirty' in text or 'bm' in text
            
            if has_pee and has_poo:
                return 'Mixed'
            elif has_poo:
                return 'Poo'
            elif has_pee:
                return 'Pee'
            else:
                return 'Pee' # Default

        df['Type'] = df.apply(classify_diaper, axis=1)
        df['Date'] = df['Start'].dt.date
        df['Time'] = df['Start'].dt.time
        df['Hour'] = df['Start'].dt.hour
        
        return df

    def _render_kpis(self):
        df = self._get_diaper_data()
        if df.empty: return

        # --- CORRECCIÓN ---
        # En lugar de usar la diferencia de fechas del selector (que puede ser enorme),
        # contamos cuántos días únicos existen realmente en los datos filtrados.
        active_days = df['Date'].nunique()
        
        # Usamos max(..., 1) para evitar división por cero si algo falla
        denom = max(active_days, 1)
        
        # Calcular totales
        total_changes = len(df)
        total_pee = len(df[df['Type'].isin(['Pee', 'Mixed'])])
        total_poo = len(df[df['Type'].isin(['Poo', 'Mixed'])])
        
        # Calcular medias usando los días activos
        avg_changes = total_changes / denom
        avg_pee = total_pee / denom
        avg_poo = total_poo / denom

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Changes/Day", f"{avg_changes:.1f}")
        col2.metric("Avg Wet/Day", f"{avg_pee:.1f}")
        col3.metric("Avg Dirty/Day", f"{avg_poo:.1f}")
        st.divider()

    def _daily_composition_chart(self):
        df = self._get_diaper_data()
        if df.empty: 
            st.info("No data available.")
            return

        # Agrupar por Fecha y Tipo
        daily_counts = df.groupby(['Date', 'Type']).size().unstack(fill_value=0).reset_index()
        
        # Asegurar columnas
        for col in ['Pee', 'Poo', 'Mixed']:
            if col not in daily_counts.columns: daily_counts[col] = 0

        fig = go.Figure()
        
        # Orden apilado: Pee, Mixed, Poo
        fig.add_trace(go.Bar(name='Pee', x=daily_counts['Date'], y=daily_counts['Pee'], marker_color='#FFD700')) 
        fig.add_trace(go.Bar(name='Mixed', x=daily_counts['Date'], y=daily_counts['Mixed'], marker_color='#FFA500')) 
        fig.add_trace(go.Bar(name='Poo', x=daily_counts['Date'], y=daily_counts['Poo'], marker_color='#8B4513'))

        fig.update_layout(
            barmode='stack', 
            yaxis_title="Count", 
            xaxis_title="Date",
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    def _pee_trend_chart(self):
        df = self._get_diaper_data()
        if df.empty: return
        
        # Consideramos 'Pee' y 'Mixed' como pañales mojados
        wet_df = df[df['Type'].isin(['Pee', 'Mixed'])].copy()
        
        daily_wet = wet_df.groupby('Date').size().reset_index(name='Count')
        
        # Rellenar fechas
        daily_wet['Date'] = pd.to_datetime(daily_wet['Date'])
        daily_wet = daily_wet.set_index('Date')
        idx = pd.date_range(daily_wet.index.min(), daily_wet.index.max())
        daily_wet = daily_wet.reindex(idx, fill_value=0).reset_index().rename(columns={'index': 'Date'})
        daily_wet['Date'] = daily_wet['Date'].dt.date

        daily_wet['Rolling'] = daily_wet['Count'].rolling(7, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_wet['Date'], y=daily_wet['Count'], 
            mode='markers', name='Daily Wet Count',
            marker=dict(color='#FFD700', size=6, opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=daily_wet['Date'], y=daily_wet['Rolling'], 
            mode='lines', name='7-Day Avg',
            line=dict(color='#DAA520', width=3)
        ))
        
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), title="Wet Diapers Evolution")
        st.plotly_chart(fig, use_container_width=True)

    def _hourly_distribution_chart(self):
        df = self._get_diaper_data()
        if df.empty: return

        # Solo Poo y Mixed
        dirty_df = df[df['Type'].isin(['Poo', 'Mixed'])].copy()
        
        if dirty_df.empty:
            st.info("No dirty diapers recorded.")
            return

        hourly_counts = dirty_df.groupby('Hour').size().reindex(range(24), fill_value=0).reset_index(name='Count')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hourly_counts['Hour'], 
            y=hourly_counts['Count'],
            marker_color='#8B4513',
            hovertemplate='<b>%{x}:00</b><br>Count: %{y}<extra></extra>'
        ))

        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 4, 8, 12, 16, 20, 24],
                ticktext=['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
                title="Hour of Day"
            ),
            yaxis_title="Total Events",
            height=350,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    def _poop_scatter_chart(self):
        df = self._get_diaper_data()
        if df.empty: return
        
        dirty_df = df[df['Type'].isin(['Poo', 'Mixed'])].copy()
        if dirty_df.empty: return

        # Truco Y-Axis
        dirty_df['Y_Axis'] = dirty_df['Start'].apply(lambda x: x.replace(year=2000, month=1, day=1))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dirty_df['Date'],
            y=dirty_df['Y_Axis'],
            mode='markers',
            marker=dict(
                color='#8B4513', 
                size=10, 
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            text=dirty_df['Time'].astype(str),
            hovertemplate='<b>%{x}</b><br>Time: %{text}<extra></extra>'
        ))

        fig.update_layout(
            yaxis=dict(
                tickformat='%H:%M',
                range=[datetime(2000,1,1,0,0), datetime(2000,1,1,23,59)],
                title="Time of Day"
            ),
            xaxis_title="Date",
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_manager import DataManager

class DiaperSection:
    def __init__(self, df, start_date, end_date):
        self.df = df
        self.start_date = start_date
        self.end_date = end_date

    def render(self):
        st.header("🧷 Diaper Changes")
        diaper_df = DataManager.filter_by_category(self.df, 'Diaper')
        diaper_df = DataManager.filter_by_date_range(diaper_df, self.start_date, self.end_date)
        
        pee_df = diaper_df[diaper_df['End Condition'].str.contains('pee', case=False, na=False)]
        poo_df = diaper_df[diaper_df['End Condition'].str.contains('poo', case=False, na=False)]

        chart_data = pd.DataFrame({
            'Pee': pee_df.groupby(pee_df['Start'].dt.date).size(),
            'Poo': poo_df.groupby(poo_df['Start'].dt.date).size()
        }).fillna(0)

        fig = go.Figure(data=[
            go.Bar(name='Pee', x=chart_data.index, y=chart_data['Pee'], marker=dict(color='gold')),
            go.Bar(name='Poo', x=chart_data.index, y=chart_data['Poo'], marker=dict(color='saddlebrown'))
        ])
        fig.update_layout(title="Diaper Chart: Pee vs. Poo", barmode='stack')
        st.plotly_chart(fig)

        # Trend Chart
        chart_data['Pee Rolling'] = chart_data['Pee'].rolling(window=7).mean()
        chart_data['Poo Rolling'] = chart_data['Poo'].rolling(window=7).mean()
        
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Pee Rolling'], name='Pee Trend'))
        trend_fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Poo Rolling'], name='Poo Trend'))
        trend_fig.update_layout(title="Diaper Usage Trend (7-day Rolling)")
        st.plotly_chart(trend_fig)
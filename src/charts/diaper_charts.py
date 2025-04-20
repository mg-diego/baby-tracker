import streamlit as st
import pandas as pd
import plotly.graph_objects as go

class DiaperCharts:
    def __init__(self, df):
        self.df = df

    def render(self, start_date, end_date):
        df = self.df.copy()
        df = df[(df['Start'] >= start_date) & (df['Start'] <= end_date)]

        pee_df = df[df['End Condition'].str.contains('pee', case=False, na=False)]
        poo_df = df[df['End Condition'].str.contains('poo', case=False, na=False)]

        pee_counts = pee_df.groupby(pee_df['Start'].dt.date).size()
        poo_counts = poo_df.groupby(poo_df['Start'].dt.date).size()

        chart_data = pd.DataFrame({'Pee': pee_counts, 'Poo': poo_counts}).fillna(0)

        bar_chart = go.Figure(data=[
            go.Bar(name='Pee', x=chart_data.index, y=chart_data['Pee'], marker_color='gold'),
            go.Bar(name='Poo', x=chart_data.index, y=chart_data['Poo'], marker_color='saddlebrown')
        ])

        bar_chart.update_layout(
            title="Diaper Chart: Pee vs. Poo",
            xaxis_title="Date", yaxis_title="Count", barmode='stack', showlegend=True
        )

        avg_pee = chart_data['Pee'].mean()
        avg_poo = chart_data['Poo'].mean()

        df = df.sort_values(by='Start')
        df['Time Difference'] = df['Start'].diff()
        avg_time_diff = df['Time Difference'].dropna().mean()
        avg_time_diff_hours = avg_time_diff.total_seconds() / 3600

        st.markdown(
            f'<div style="text-align: center;">'
            f'<div style="display: inline-block; width: 25px; height: 25px; background-color: gold;"></div> <b>Average Pee / day:</b> {avg_pee:.2f}'
            f'<div style="display: inline-block; width: 25px; height: 25px; background-color: saddlebrown; margin-left: 30px;"></div> <b>Average Poo / day:</b> {avg_poo:.2f}'
            f'<div><b>Average Time Between Diapers:</b> {avg_time_diff_hours:.2f} hours</div>'
            f'</div>', unsafe_allow_html=True
        )

        st.plotly_chart(bar_chart)

        # Rolling Average Chart
        rolling_window = 7
        chart_data['Pee Rolling Avg'] = chart_data['Pee'].rolling(window=rolling_window).mean()
        chart_data['Poo Rolling Avg'] = chart_data['Poo'].rolling(window=rolling_window).mean()

        trend_chart = go.Figure()
        trend_chart.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Pee Rolling Avg'],
                                         mode='lines+markers', name='Pee Trend', line=dict(color='gold', width=3)))
        trend_chart.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Poo Rolling Avg'],
                                         mode='lines+markers', name='Poo Trend', line=dict(color='saddlebrown', width=3)))

        trend_chart.update_layout(
            title="Diaper Usage Trend (Pee vs Poo)",
            xaxis_title="Date", yaxis_title="Average Count (Rolling)", showlegend=True
        )
        st.plotly_chart(trend_chart)
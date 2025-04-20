import streamlit as st
import pandas as pd
import plotly.graph_objects as go

class BreastfeedingChart:
    def __init__(self, df):
        self.df = df

    def render(self, start_date, end_date):
        df = self.df.copy()
        df = df[
            (df['Start'] >= start_date) &
            (df['Start'] <= end_date) &
            (df['Start Location'].str.lower() == 'breast')
        ]

        right_df = df[df['Start Condition'].str.contains('R', case=False, na=False)]
        left_df = df[df['End Condition'].str.contains('L', case=False, na=False)]

        right_counts = right_df.groupby(right_df['Start'].dt.date).size()
        left_counts = left_df.groupby(left_df['Start'].dt.date).size()

        chart_data = pd.DataFrame({'Left': left_counts, 'Right': right_counts}).fillna(0)

        fig = go.Figure(data=[
            go.Bar(name='Left', x=chart_data.index, y=chart_data['Left'], marker_color='lightcoral'),
            go.Bar(name='Right', x=chart_data.index, y=chart_data['Right'], marker_color='mediumpurple')
        ])

        fig.update_layout(
            title="Breastfeeding Chart: Left vs. Right",
            xaxis_title="Date", yaxis_title="Count", barmode='stack', showlegend=True
        )

        avg_right = chart_data['Right'].mean()
        avg_left = chart_data['Left'].mean()

        st.markdown(
            f'<div style="text-align: center;">'
            f'<div style="display: inline-block; width: 25px; height: 25px; background-color: mediumpurple;"></div> <b>Average Right / day:</b> {avg_right:.0f}'
            f'<div style="display: inline-block; width: 25px; height: 25px; background-color: lightcoral;margin-left: 30px"></div> <b>Average Left / day:</b> {avg_left:.0f}'
            f'</div>', unsafe_allow_html=True
        )

        sorted_times = df.sort_values(by='Start')['Start']
        time_deltas = sorted_times.diff().dropna()
        if not time_deltas.empty:
            avg_timedelta = time_deltas.mean()
            avg_hours = avg_timedelta.total_seconds() // 3600
            avg_minutes = (avg_timedelta.total_seconds() % 3600) // 60
            st.markdown(f"<div style='text-align: center;'><b>Average time between feedings:</b> {int(avg_hours)}h {int(avg_minutes)}min</div>",
                        unsafe_allow_html=True)

        st.plotly_chart(fig)
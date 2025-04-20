import streamlit as st
import pandas as pd
import plotly.graph_objects as go

class SleepHeatmap:
    def __init__(self, df):
        self.df = df

    def is_asleep(self, timestamp, intervals):
        for start, end in intervals:
            if start <= timestamp < end:
                return 1
        return 0

    def render(self):
        sleep_df = self.df.copy()
        sleep_df = sleep_df.dropna(subset=['Start', 'End'])

        start_range = sleep_df['Start'].min().replace(hour=0, minute=0, second=0, microsecond=0)
        end_range = sleep_df['Start'].max().replace(hour=0, minute=0, second=0, microsecond=0)

        full_range = pd.date_range(start=start_range, end=end_range + pd.Timedelta(days=1), freq='5min')
        timestamps_df = pd.DataFrame({'datetime': full_range})
        timestamps_df['date'] = timestamps_df['datetime'].dt.date
        timestamps_df['hour'] = timestamps_df['datetime'].dt.strftime('%H:%M')

        sleep_intervals = list(zip(sleep_df['Start'], sleep_df['End']))
        timestamps_df['state'] = timestamps_df['datetime'].apply(lambda ts: self.is_asleep(ts, sleep_intervals))

        heatmap_data = timestamps_df.pivot(index='date', columns='hour', values='state').fillna(0)

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=[[0.0, "lightblue"], [1.0, "royalblue"]],
            showscale=False,
            hovertemplate='Date: %{y}<br>Hour: %{x}<br>'
        ))

        fig.update_layout(
            xaxis=dict(title="Hour", side="top"),
            yaxis=dict(title="Date", autorange=True),
            height=15 * len(heatmap_data.index)
        )

        st.markdown(
            f'<div style="text-align: center;">'
            f'<div style="display: inline-block; width: 25px; height: 25px; background-color: royalblue;"></div> Sleep &nbsp;'
            f'<div style="display: inline-block; width: 25px; height: 25px; background-color: lightblue; margin-left: 30px;"></div> Awake'
            f'</div>', unsafe_allow_html=True
        )
        st.plotly_chart(fig, use_container_width=True)
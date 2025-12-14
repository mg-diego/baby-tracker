import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_manager import DataManager

class FeedingSection:
    def __init__(self, df, start_date, end_date):
        self.df = df
        self.start_date = start_date
        self.end_date = end_date

    def render(self):
        st.header("🍼 Feeding Times")
        st.write("Display feeding sessions, types, and intervals.")
        self._breastfeeding_chart()
        self._breastfeeding_duration_chart()
        self._breastfeeding_intervals_chart()

    def _breastfeeding_chart(self):
        feed_df = DataManager.filter_by_category(self.df, 'Feed')
        feed_df = DataManager.filter_by_date_range(feed_df, self.start_date, self.end_date)
        
        # Filtrar solo 'breast'
        feed_df = feed_df[feed_df['Start Location'].str.lower() == 'breast']

        right_df = feed_df[feed_df['Start Condition'].str.contains('R', case=False, na=False)]
        left_df = feed_df[feed_df['End Condition'].str.contains('L', case=False, na=False)]

        chart_data = pd.DataFrame({
            'Left': left_df.groupby(left_df['Start'].dt.date).size(),
            'Right': right_df.groupby(right_df['Start'].dt.date).size()
        }).fillna(0)

        fig = go.Figure(data=[
            go.Bar(name='Left', x=chart_data.index, y=chart_data['Left'], marker=dict(color='lightcoral')),
            go.Bar(name='Right', x=chart_data.index, y=chart_data['Right'], marker=dict(color='mediumpurple'))
        ])
        fig.update_layout(title="Breastfeeding Chart: Left vs. Right", barmode='stack')
        
        # Calcular medias para mostrar texto
        avg_r = chart_data['Right'].mean() if not chart_data.empty else 0
        avg_l = chart_data['Left'].mean() if not chart_data.empty else 0
        
        st.markdown(
            f'<div style="text-align: center;">'
            f'<div style="display: inline-block; width: 25px; height: 25px; background-color: mediumpurple;"></div> <b>Avg Right/day:</b> {avg_r:.0f}'
            f'<div style="display: inline-block; width: 25px; height: 25px; background-color: lightcoral;margin-left: 30px"></div> <b>Avg Left/day:</b> {avg_l:.0f}'
            f'</div>', unsafe_allow_html=True
        )
        
        # Calcular intervalo promedio global
        sorted_times = feed_df.sort_values(by='Start')['Start']
        time_deltas = sorted_times.diff().dropna()
        if not time_deltas.empty:
            total_sec = time_deltas.mean().total_seconds()
            st.markdown(f"<div style='text-align: center; margin-top: 20px;'><b>Avg time between sessions:</b> {int(total_sec//3600)}h {int((total_sec%3600)//60)}m</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center;'>Not enough data for avg interval.</div>", unsafe_allow_html=True)

        st.plotly_chart(fig)

    def _breastfeeding_duration_chart(self):
        feed_df = DataManager.filter_by_category(self.df, 'Feed')
        feed_df = DataManager.filter_by_date_range(feed_df, self.start_date, self.end_date)
        
        # Filtrar solo 'breast' y que tengan fecha de fin
        feed_df = feed_df[feed_df['Start Location'].str.lower() == 'breast']
        feed_df = feed_df.dropna(subset=['End'])
        
        if feed_df.empty:
            st.info("No duration data available for selected range.")
            return

        feed_df['Duration'] = (feed_df['End'] - feed_df['Start']).dt.total_seconds() / 3600
        duration_per_day = feed_df.groupby(feed_df['Start'].dt.date)['Duration'].sum()
        rolling_avg = duration_per_day.rolling(window=7, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=duration_per_day.index, y=duration_per_day.values, name="Daily Total", marker=dict(color='teal')))
        fig.add_trace(go.Scatter(x=rolling_avg.index, y=rolling_avg.values, name="7-Day Avg", mode='lines+markers', line=dict(color='darkorange', width=2)))
        
        fig.update_layout(title="Total Breastfeeding Duration Per Day (Hours)", barmode='group', legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig)

    def _breastfeeding_intervals_chart(self):
        feed_df = DataManager.filter_by_category(self.df, 'Feed')
        feed_df = DataManager.filter_by_date_range(feed_df, self.start_date, self.end_date)
        feed_df = feed_df[feed_df['Start Location'].str.lower() == 'breast']
        
        if feed_df.empty:
            return

        avg_interval_per_day = []
        dates = []
        
        # Calcular intervalos por día
        for date, group in feed_df.groupby(feed_df['Start'].dt.date):
            day_deltas = group.sort_values(by='Start')['Start'].diff().dropna()
            if not day_deltas.empty:
                avg_interval_per_day.append(day_deltas.mean().total_seconds() / 3600)
                dates.append(date)
        
        interval_data = pd.DataFrame({'Date': dates, 'Average Interval (hrs)': avg_interval_per_day})
        
        if interval_data.empty: 
            return

        interval_data['3-Day Moving Avg'] = interval_data['Average Interval (hrs)'].rolling(window=7).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=interval_data['Date'], y=interval_data['Average Interval (hrs)'],
                                 mode='lines+markers', line=dict(color='teal'), name='Daily Average'))
        fig.add_trace(go.Scatter(x=interval_data['Date'], y=interval_data['3-Day Moving Avg'],
                                 mode='lines', line=dict(color='orange', dash='dash'), name='3-Day Mov Avg'))
        
        fig.update_layout(title="Average Time Between Breastfeeding Sessions (per day)", yaxis_title="Hours")
        st.plotly_chart(fig)
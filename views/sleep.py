import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from data_manager import DataManager

class SleepSection:
    def __init__(self, df, start_date, end_date):
        self.df = df
        self.start_date = start_date
        self.end_date = end_date
        
        # Datos Dummy originales para las pestañas 3 y 4 (Nap Duration y Night Wakeups)
        # Tal como estaban en el script original al principio
        self.dummy_data = pd.DataFrame({
            'Date': pd.date_range(start='2024-04-01', periods=7, freq='D'),
            'Total Sleep (hrs)': [12, 13, 11.5, 14, 12.2, 13.3, 12.8],
            'Nap Duration (hrs)': [2, 1.5, 1.8, 2.5, 1.2, 2.1, 1.7],
            'Night Wakeups': [2, 1, 3, 1, 2, 1, 2],
        })

    def render(self):
        st.header("💤 Sleep Patterns")
        st.write("Visualize sleep logs or averages.")

        tab1, tab2, tab3, tab4 = st.tabs(["Sleep/Awake Chart", "Daily Sleep Trends", "Nap Duration", "Nighttime Wakeups"])

        with tab1:
            st.write("🛌 Sleep/Awake Chart")
            st.write("This heatmap visualizes the baby's sleep and wake patterns... 'royalblue' indicates sleep, while 'lightblue' represents awake periods.", unsafe_allow_html=True)
            self._sleep_heatmap()
        
        with tab2:
            st.write("🌞 Daily Sleep Trends")       
            self._sleep_per_day_chart()
            self._wake_time_per_day_chart()
            self._bed_time_per_day_chart()

        with tab3:
            st.write("🌞 Nap Duration")
            fig3 = px.bar(self.dummy_data, x='Date', y='Nap Duration (hrs)', title='Daytime Nap Duration2')
            st.plotly_chart(fig3, use_container_width=True)

        with tab4:
            st.write('🌙 Nighttime Wakeups lorem ipsum')
            fig4 = px.line(self.dummy_data, x='Date', y='Night Wakeups', markers=True, title='Night Wakeups Per Day')
            st.plotly_chart(fig4, use_container_width=True)

    def _sleep_heatmap(self):
        sleep_df = DataManager.filter_by_category(self.df, 'Sleep')
        # No aplicamos filtro de fechas estricto aquí para mostrar todo el heatmap disponible o el rango relevante
        # pero usamos dropna para asegurar integridad
        sleep_df = sleep_df.dropna(subset=['Start', 'End'])
        
        if sleep_df.empty:
            st.warning("No sleep data available for heatmap.")
            return

        # Get date range based on data
        oldest_date = sleep_df['Start'].min().replace(hour=0, minute=0, second=0, microsecond=0)
        newest_date = sleep_df['Start'].max().replace(hour=0, minute=0, second=0, microsecond=0)

        data = {
            'datetime': pd.date_range(start=oldest_date, end=newest_date + pd.Timedelta(days=1), freq='5min'),
        }
        df_grid = pd.DataFrame(data)
        df_grid['date'] = df_grid['datetime'].dt.date
        df_grid['hour'] = df_grid['datetime'].dt.strftime('%H:%M')

        # Prepare intervals
        sleep_intervals = list(zip(sleep_df['Start'], sleep_df['End']))

        # Function to apply sleep state
        def is_asleep(timestamp):
            for start, end in sleep_intervals:
                if start <= timestamp < end:
                    return 1
            return 0

        # Optimization: Apply map directly (slower than numpy but safe for this logic)
        df_grid['state'] = df_grid['datetime'].map(is_asleep)

        # Pivot table
        heatmap_data = df_grid.pivot(index='date', columns='hour', values='state').fillna(0)

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
            yaxis=dict(
                title="Date",
                autorange=True,
                tickmode="array",
                tickvals=heatmap_data.index.tolist(),
                ticktext=[str(d) for d in heatmap_data.index],
            ),
            height=max(400, 15 * len(heatmap_data.index))
        )

        st.markdown(
            f'<div style="text-align: center;">'
            f'<div style="display: inline-block; width: 25px; height: 25px; background-color: royalblue;"></div> Sleep &nbsp;'
            f'<div style="display: inline-block; width: 25px; height: 25px; background-color: lightblue; margin-left: 30px;"></div> Awake'
            f'</div>',
            unsafe_allow_html=True
        )

        st.plotly_chart(fig, use_container_width=True)

    def _sleep_per_day_chart(self):
        df = DataManager.filter_by_category(self.df, 'Sleep')
        df = DataManager.filter_by_date_range(df, self.start_date, self.end_date)
        df = df.dropna(subset=['Start', 'End'])
        
        if df.empty:
            return

        df['DurationHours'] = (df['End'] - df['Start']).dt.total_seconds() / 3600
        df['Date'] = df['Start'].dt.date
        daily_sleep = df.groupby('Date')['DurationHours'].sum().reset_index()

        avg_sleep = daily_sleep['DurationHours'].mean()

        fig = go.Figure()

        # Daily sleep line
        fig.add_trace(go.Scatter(
            x=daily_sleep['Date'],
            y=daily_sleep['DurationHours'],
            mode='lines+markers+text',
            line_shape='spline',
            fill='tozeroy',
            line=dict(color='skyblue'),
            marker=dict(color='steelblue'),
            name='Total Sleep',
            hovertemplate='Date: %{x}<br>Total Sleep: %{y:.2f} hours<extra></extra>'
        ))

        # Average line
        fig.add_trace(go.Scatter(
            x=daily_sleep['Date'],
            y=[avg_sleep] * len(daily_sleep),
            mode='lines',
            line=dict(color='darkgray', width=2, dash='dot'),
            name=f'Average Sleep ({avg_sleep:.2f} hrs)',
            hovertemplate='Date: %{x}<br>Average Sleep: %{y:.2f} hours<extra></extra>'
        ))

        fig.update_layout(
            title='Total Sleep per Day',
            xaxis_title='Date',
            yaxis_title='Total Sleep (hours)',
            yaxis=dict(range=[0, 24]),
            height=600
        )

        col1, col2 = st.columns([4, 1])
        with col1:
            st.plotly_chart(fig)
        with col2:
            st.subheader("Data:")
            st.dataframe(daily_sleep)

    def _wake_time_per_day_chart(self):
        df = DataManager.filter_by_category(self.df, 'Woke up')
        df = DataManager.filter_by_date_range(df, self.start_date, self.end_date)
        
        if df.empty:
            st.warning("No 'Woke up' events found in the selected date range.")
            return

        df = df.copy()
        df['Date'] = df['Start'].dt.date
        df['HourDecimal'] = df['Start'].dt.hour + df['Start'].dt.minute / 60
        df['HourStr'] = df['Start'].dt.strftime('%H:%Mh')

        avg_wake_time = df['HourDecimal'].mean()
        avg_hour = int(avg_wake_time)
        avg_minute = int((avg_wake_time % 1) * 60)
        avg_str = f"{avg_hour:02}:{avg_minute:02}h"

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['HourDecimal'],
            mode='lines+markers',
            line_shape='spline',
            fill='tozeroy',
            line=dict(color='orange', width=2),
            marker=dict(color='darkorange'),
            name='Wake-Up Time',
            customdata=df['HourStr'],
            hovertemplate="Date: %{x}<br>Wake-up: %{customdata}<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=[avg_wake_time] * len(df),
            mode='lines',
            line=dict(color='gray', dash='dot', width=2),
            name=f'Average Wake-Up ({avg_str})',
            hovertemplate="Date: %{x}<br>Avg Wake-up: {avg_str}<extra></extra>"
        ))

        min_val = int(df['HourDecimal'].min()) - 1
        max_val = int(df['HourDecimal'].max()) + 1

        fig.update_layout(
            title='Wake-Up Time per Day',
            xaxis_title='Date',
            yaxis_title='Wake-Up Time',
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(min_val, max_val)),
                ticktext=[f"{h:02}:00h" for h in range(min_val, max_val)],
                range=[min_val, max_val]
            ),
            height=500
        )

        col1, col2 = st.columns([4, 1])
        with col1:
            st.plotly_chart(fig)
        with col2:
            st.subheader("Data:")
            st.dataframe(df[['Date', 'Start', 'HourStr']])

    def _bed_time_per_day_chart(self):
        df = DataManager.filter_by_category(self.df, 'Bed time')
        df = DataManager.filter_by_date_range(df, self.start_date, self.end_date)
        
        if df.empty:
            st.warning("No 'Bed time' events found in the selected date range.")
            return

        df = df.copy()
        df['Date'] = df['Start'].dt.date
        df['HourDecimal'] = df['Start'].dt.hour + df['Start'].dt.minute / 60
        df['HourStr'] = df['Start'].dt.strftime('%H:%Mh')

        avg_wake_time = df['HourDecimal'].mean()
        avg_hour = int(avg_wake_time)
        avg_minute = int((avg_wake_time % 1) * 60)
        avg_str = f"{avg_hour:02}:{avg_minute:02}h"

        # Polynomial Regression
        X = np.array([d.toordinal() for d in df['Date']]).reshape(-1, 1)
        y = df['HourDecimal'].values
        
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        trend_y = model.predict(X_poly)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['HourDecimal'],
            mode='lines+markers',
            line_shape='spline',
            fill='tozeroy',
            line=dict(color='red', width=2),
            marker=dict(color='darkred'),
            name='Bed Time',
            customdata=df['HourStr'],
            hovertemplate="Date: %{x}<br>Bed time: %{customdata}<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=[avg_wake_time] * len(df),
            mode='lines',
            line=dict(color='gray', dash='dot', width=2),
            name=f'Average Bed time ({avg_str})',
            hovertemplate="Date: %{x}<br>Avg Bed time: {avg_str}<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=trend_y,
            mode='lines',
            line=dict(color='blue', dash='dot', width=2),
            name='Polynomial Trend Line',
            hoverinfo='skip'
        ))

        min_val = int(df['HourDecimal'].min()) - 1
        max_val = int(df['HourDecimal'].max()) + 1

        fig.update_layout(
            title='Bed Time per Day',
            xaxis_title='Date',
            yaxis_title='Bed Time',
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(min_val, max_val)),
                ticktext=[f"{h:02}:00h" for h in range(min_val, max_val)],
                range=[min_val, max_val]
            ),
            height=500
        )

        col1, col2 = st.columns([4, 1])
        with col1:
            st.plotly_chart(fig)
        with col2:
            st.subheader("Data:")
            st.dataframe(df[['Date', 'Start', 'HourStr']])
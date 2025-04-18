import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import time

import streamlit_antd_components as sac

# https://matplotlib.org/stable/gallery/color/named_colors.html
# Dummy data
sleep_data = pd.DataFrame({
    'Date': pd.date_range(start='2024-04-01', periods=7, freq='D'),
    'Total Sleep (hrs)': [12, 13, 11.5, 14, 12.2, 13.3, 12.8],
    'Nap Duration (hrs)': [2, 1.5, 1.8, 2.5, 1.2, 2.1, 1.7],
    'Night Wakeups': [2, 1, 3, 1, 2, 1, 2],
})

    # Default values for custom input
today = datetime.datetime.now()
default_start = today - datetime.timedelta(days=7)
default_end = today

st.set_page_config(page_title="👶 Baby Tracker Dashboard", layout="wide")

df = pd.read_csv("data/huckelberry.csv", sep=",")

def filter_csv_by_category(data_frame, category):
    return data_frame[data_frame['Type'] == category]

def filter_csv_by_start_date(data_frame, start_date):
    data_frame["Start"] = pd.to_datetime(data_frame["Start"], errors='coerce')
    return data_frame[data_frame["Start"].dt.date == pd.to_datetime(start_date).date()]

def sleep_heatmap():
    # Step 1: Load and clean the sleep intervals
    sleep_df = filter_csv_by_category(df, 'Sleep')
    sleep_df['Start'] = pd.to_datetime(sleep_df['Start'], errors='coerce')
    sleep_df['End'] = pd.to_datetime(sleep_df['End'], errors='coerce')  # Assuming there's an 'End' column
    sleep_df = sleep_df.dropna(subset=['Start', 'End'])

    # Step 2: Get date range and generate hourly timestamps
    oldest_date = sleep_df['Start'].min().replace(hour=0, minute=0, second=0, microsecond=0)
    newest_date = sleep_df['Start'].max().replace(hour=0, minute=0, second=0, microsecond=0)

    data = {
        'datetime': pd.date_range(start=oldest_date, end=newest_date + pd.Timedelta(days=1), freq='5min'),
    }
    df = pd.DataFrame(data)
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.strftime('%H:%M')

    # Step 3: Define a function to check if a timestamp is within any sleep range
    def is_asleep(timestamp, intervals):
        for start, end in intervals:
            if start <= timestamp < end:
                return 1
        return 0

    # Prepare list of tuples with sleep intervals
    sleep_intervals = list(zip(sleep_df['Start'], sleep_df['End']))

    # Step 4: Apply the real sleep state to each timestamp
    df['state'] = df['datetime'].apply(lambda ts: is_asleep(ts, sleep_intervals))

    # Step 5: Create pivot table
    heatmap_data = df.pivot(index='date', columns='hour', values='state').fillna(0)

    sleep_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[  # Strict binary mapping
            [0.0, "lightblue"],   # 0 = awake
            [1.0, "royalblue"]     # 1 = asleep
        ],
        showscale=False,  # Hide color legend
        hovertemplate=(
            'Date: %{y}<br>'  # Show the date
            'Hour: %{x}<br>'  # Show the hour
        )
    ))

    sleep_heatmap.update_layout(
        xaxis=dict(title="Hour", side="top"),
        yaxis=dict(
            title="Date",
            autorange=True,
            tickmode="array",
            tickvals=heatmap_data.index.tolist(),
            ticktext=[str(d) for d in heatmap_data.index],
        ),
        height=15 * len(heatmap_data.index)
    )

    st.markdown(
        f'<div style="text-align: center;">'
        f'<div style="display: inline-block; width: 25px; height: 25px; background-color: royalblue;"></div> Sleep &nbsp;'
        f'<div style="display: inline-block; width: 25px; height: 25px; background-color: lightblue; margin-left: 30px;"></div> Awake'
        f'</div>',
        unsafe_allow_html=True
    )

    st.plotly_chart(sleep_heatmap, use_container_width=True)

def diaper_charts(start_date, end_date):
    # Step 1: Load the CSV data (assuming a function 'filter_csv_by_category' is available)
    diaper_df = filter_csv_by_category(df, 'Diaper')  # Replace with your actual CSV parsing method

    # Step 2: Convert the 'Start' column to datetime (assuming 'Start' column has the date)
    diaper_df['Start'] = pd.to_datetime(diaper_df['Start'], errors='coerce')

    # Step 3: Filter the data by the provided date range
    diaper_df = diaper_df[(diaper_df['Start'] >= start_date) & (diaper_df['Start'] <= end_date)]

    # Step 4: Filter the rows based on "End condition" containing 'pee' or 'poo'
    pee_df = diaper_df[diaper_df['End Condition'].str.contains('pee', case=False, na=False)]
    poo_df = diaper_df[diaper_df['End Condition'].str.contains('poo', case=False, na=False)]

    # Step 5: Group by the date and count the occurrences of each condition per date
    pee_count_per_day = pee_df.groupby(pee_df['Start'].dt.date).size()
    poo_count_per_day = poo_df.groupby(poo_df['Start'].dt.date).size()

    # Step 6: Merge the two series (pee and poo counts) into one DataFrame for easier plotting
    chart_data = pd.DataFrame({
        'Pee': pee_count_per_day,
        'Poo': poo_count_per_day
    }).fillna(0)  # Replace NaN with 0 for dates that don't have a specific condition

    # Step 7: Create the stacked bar chart
    diaper_chart = go.Figure(data=[
        go.Bar(name='Pee', x=chart_data.index, y=chart_data['Pee'], marker=dict(color='gold')),
        go.Bar(name='Poo', x=chart_data.index, y=chart_data['Poo'], marker=dict(color='saddlebrown'))
    ])

    # Step 8: Update layout to stack bars
    diaper_chart.update_layout(
        title="Diaper Chart: Pee vs. Poo",
        xaxis_title="Date",
        yaxis_title="Count",
        barmode='stack',
        showlegend=True
    )
    
    # Step 9: Calculate average values per category
    avg_pee = chart_data['Pee'].mean()
    avg_poo = chart_data['Poo'].mean()

        # Step 11: Calculate the average time between diaper changes
    # First, sort by the 'Start' time
    diaper_df = diaper_df.sort_values(by='Start')

    # Calculate the time difference between consecutive rows
    diaper_df['Time Difference'] = diaper_df['Start'].diff()

    # Remove the first row (which has NaT for Time Difference)
    diaper_df = diaper_df.dropna(subset=['Time Difference'])

    # Calculate the average time difference (in hours or days)
    avg_time_diff = diaper_df['Time Difference'].mean()

    # Display average time between diapers in days
    avg_time_diff_days = avg_time_diff.total_seconds() / (60 * 60)  # Convert seconds to days

    st.markdown(
        f'<div style="text-align: center;">'
        f'<div style="display: inline-block; width: 25px; height: 25px; background-color: gold"></div> <b>Average Pee / day:</b> {avg_pee:.2f}'
        f'<div style="display: inline-block; width: 25px; height: 25px; background-color: saddlebrown; margin-left: 30px;"></div> <b>Average Poo / day:</b> {avg_poo:.2f}'
        f'<div><b>Average Time Between Diapers:</b> {avg_time_diff_days:.2f} hours</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Step 10: Display the chart using Streamlit
    st.plotly_chart(diaper_chart)

    # Step 12: Calculate the rolling average (e.g., 7-day moving average) for Pee and Poo
    rolling_window = 7
    chart_data['Pee Rolling Avg'] = chart_data['Pee'].rolling(window=rolling_window).mean()
    chart_data['Poo Rolling Avg'] = chart_data['Poo'].rolling(window=rolling_window).mean()

    # Step 13: Create the trend line chart for the rolling averages
    trend_chart = go.Figure()

    # Plot Pee trend line
    trend_chart.add_trace(go.Scatter(
        x=chart_data.index, 
        y=chart_data['Pee Rolling Avg'], 
        mode='lines+markers', 
        name='Pee Trend', 
        line=dict(color='gold', width=3)
    ))

    # Plot Poo trend line
    trend_chart.add_trace(go.Scatter(
        x=chart_data.index, 
        y=chart_data['Poo Rolling Avg'], 
        mode='lines+markers', 
        name='Poo Trend', 
        line=dict(color='saddlebrown', width=3)
    ))

    # Step 14: Update layout for the trend chart
    trend_chart.update_layout(
        title="Diaper Usage Trend (Pee vs Poo)",
        xaxis_title="Date",
        yaxis_title="Average Count (Rolling)",
        showlegend=True
    )

    # Step 15: Display the trend chart using Streamlit
    st.plotly_chart(trend_chart)

def breastfeeding_chart(start_date, end_date):
    # Step 1: Load and filter CSV data
    feed_df = filter_csv_by_category(df, 'Feed')
    feed_df['Start'] = pd.to_datetime(feed_df['Start'], errors='coerce')

    # Step 2: Filter by date range and start location
    feed_df = feed_df[
        (feed_df['Start'] >= start_date) &
        (feed_df['Start'] <= end_date) &
        (feed_df['Start Location'].str.lower() == 'breast')
    ]

    # Step 3: Separate into Right and Left based on End Condition
    right_df = feed_df[feed_df['Start Condition'].str.contains('R', case=False, na=False)]
    left_df = feed_df[feed_df['End Condition'].str.contains('L', case=False, na=False)]

    # Step 4: Count occurrences by date
    right_counts = right_df.groupby(right_df['Start'].dt.date).size()
    left_counts = left_df.groupby(left_df['Start'].dt.date).size()

    # Step 5: Combine counts into one DataFrame
    chart_data = pd.DataFrame({
        'Left': left_counts,
        'Right': right_counts
    }).fillna(0)

    # Step 6: Plot stacked bar chart
    fig = go.Figure(data=[
        go.Bar(name='Left', x=chart_data.index, y=chart_data['Left'], marker=dict(color='lightcoral')),
        go.Bar(name='Right', x=chart_data.index, y=chart_data['Right'], marker=dict(color='mediumpurple'))   
    ])

    fig.update_layout(
        title="Breastfeeding Chart: Left vs. Right",
        xaxis_title="Date",
        yaxis_title="Count",
        barmode='stack',
        showlegend=True
    )
    
    avg_right = chart_data['Right'].mean()
    avg_left = chart_data['Left'].mean()

    st.markdown(
        f'<div style="text-align: center;">'
        f'<div style="display: inline-block; width: 25px; height: 25px; background-color: mediumpurple;"></div> <b>Average Right / day:</b> {avg_right:.0f}'
        f'<div style="display: inline-block; width: 25px; height: 25px; background-color: lightcoral;margin-left: 30px"></div> <b>Average Left / day:</b> {avg_left:.0f}'
        f'</div>',
        unsafe_allow_html=True
    )
    sorted_times = feed_df.sort_values(by='Start')['Start']
    time_deltas = sorted_times.diff().dropna()
    if not time_deltas.empty:
        avg_timedelta = time_deltas.mean()
        avg_hours = avg_timedelta.total_seconds() // 3600
        avg_minutes = (avg_timedelta.total_seconds() % 3600) // 60
        st.markdown(
            f"<div style='text-align: center; margin-top: 20px;'>"
            f"<b>Average time between breastfeeding sessions:</b> {int(avg_hours)}h {int(avg_minutes)}m"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<div style='text-align: center;'>Not enough data to compute average interval.</div>", unsafe_allow_html=True)

    st.plotly_chart(fig)

    # Step 10: Calculate average time between sessions per day
    avg_interval_per_day = []
    dates = []

    for date, group in feed_df.groupby(feed_df['Start'].dt.date):
        sorted_day_times = group.sort_values(by='Start')['Start']
        day_deltas = sorted_day_times.diff().dropna()

        if not day_deltas.empty:
            avg_delta = day_deltas.mean().total_seconds() / 3600  # convert to hours
            avg_interval_per_day.append(avg_delta)
            dates.append(date)

    # Create DataFrame for plotting
    interval_data = pd.DataFrame({
        'Date': dates,
        'Average Interval (hrs)': avg_interval_per_day
    })

    # Calculate 3-day moving average
    interval_data['3-Day Moving Avg'] = interval_data['Average Interval (hrs)'].rolling(window=7).mean()

    # Plot line chart with trend
    interval_chart = go.Figure()

    # Actual average interval line
    interval_chart.add_trace(go.Scatter(
        x=interval_data['Date'],
        y=interval_data['Average Interval (hrs)'],
        mode='lines+markers',
        line=dict(color='teal'),
        name='Daily Average Interval'
    ))

    # Moving average line
    interval_chart.add_trace(go.Scatter(
        x=interval_data['Date'],
        y=interval_data['3-Day Moving Avg'],
        mode='lines',
        line=dict(color='orange', dash='dash'),
        name='3-Day Moving Avg'
    ))

    interval_chart.update_layout(
        title="Average Time Between Breastfeeding Sessions (per day)",
        xaxis_title="Date",
        yaxis_title="Average Hours Between Sessions",
        showlegend=True
    )

    st.plotly_chart(interval_chart)

def breastfeeding_duration_chart(start_date, end_date):
    # Step 1: Load and filter the data
    feed_df = filter_csv_by_category(df, 'Feed')
    feed_df = feed_df[feed_df['Start Location'].str.lower() == 'breast']
    feed_df['Start'] = pd.to_datetime(feed_df['Start'], errors='coerce')
    feed_df['End'] = pd.to_datetime(feed_df['End'], errors='coerce')

    # Step 2: Filter by date range
    feed_df = feed_df[(feed_df['Start'] >= start_date) & (feed_df['Start'] <= end_date)]

    # Step 3: Calculate duration in hours
    feed_df['Duration'] = (feed_df['End'] - feed_df['Start']).dt.total_seconds() / 3600  # convert to hours

    # Step 4: Group by date and sum durations
    duration_per_day = feed_df.groupby(feed_df['Start'].dt.date)['Duration'].sum()

    # Step 5: Calculate a rolling average (7-day window as an example)
    rolling_avg = duration_per_day.rolling(window=7, min_periods=1).mean()

    # Step 6: Create the chart with two series
    fig = go.Figure()

    # Bar chart for daily totals
    fig.add_trace(go.Bar(
        x=duration_per_day.index,
        y=duration_per_day.values,
        name="Daily Total",
        marker=dict(color='teal')
    ))

    # Line chart for rolling average
    fig.add_trace(go.Scatter(
        x=rolling_avg.index,
        y=rolling_avg.values,
        name="7-Day Average",
        mode='lines+markers',
        line=dict(color='darkorange', width=2),
    ))

    # Step 7: Update layout
    fig.update_layout(
        title="Total Breastfeeding Duration Per Day (Hours)",
        xaxis_title="Date",
        yaxis_title="Total Hours",
        barmode='group',
        legend=dict(x=0.01, y=0.99)
    )

    # Step 8: Display in Streamlit
    st.plotly_chart(fig)

def days_to_months_days(days):
    months = days // 30
    remaining_days = days % 30

    parts = []
    if months == 1:
        parts.append("1 month")
    elif months > 1:
        parts.append(f"{months} months")

    if remaining_days == 1:
        parts.append("1 day")
    elif remaining_days > 1:
        parts.append(f"{remaining_days} days")

    return " and ".join(parts) if parts else "0 days"

def height_chart():
    # Load baby's height data from CSV
    df = filter_csv_by_category(df, 'Growth')
    
    df['Date'] = pd.to_datetime(df['Start'], errors='coerce')
    df['Height'] = df['Start Location'].str.replace('cm', '', regex=False).astype(float)

    df = df.dropna(subset=['Date', 'Height'])

    # Calculate age in months (or days)
    birth_date = df['Date'].min()  # You might want to store baby's DOB somewhere
    df['Months'] = ((df['Date'] - birth_date).dt.days / 30.44).round(1)

    # Load percentile reference data (example assumes it's a DataFrame)
    # WHO data typically has Age (in months) and percentile columns
    percentiles_df = pd.read_csv('data/height-girls-percentiles-who.csv')  # Columns: AgeMonths, P3, P15, P50, P85, P97

    # Plot using Plotly
    fig = go.Figure()

    # Add baby's height line
    fig.add_trace(go.Scatter(
        x=df['Months'], y=df['Height'],
        mode='lines+markers',
        name="Baby's Height", line=dict(color='black', width=3),
        hovertemplate=
            'Date: %{customdata[0]}<br>' +
            'Height: %{y:.1f} cm<br>',
        customdata=df[['Date']].values
    ))

    # Add percentile curves
    for p in ['P1', 'P5', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99']:
        fig.add_trace(go.Scatter(
            x=percentiles_df['Months'],
            y=percentiles_df[p],
            mode='lines',
            name=f'{p} Percentile',
            line=dict(dash='dot')
        ))

    # Layout
    fig.update_layout(
        title="Baby's Height vs Age (with Percentiles)",
        xaxis_title="Age (months)",
        yaxis_title="Height (cm)",
        legend_title="Legend",
        height=500
    )

    st.plotly_chart(fig)

def weight_chart():
    # Load baby's height data from CSV
    df = filter_csv_by_category(df, 'Growth')
    
    df['Date'] = pd.to_datetime(df['Start'], errors='coerce')
    df['Weight'] = df['Start Condition'].str.replace('kg', '', regex=False).astype(float)

    df = df.dropna(subset=['Date', 'Weight'])

    # Calculate age in months (or days)
    birth_date = df['Date'].min()  # You might want to store baby's DOB somewhere
    df['Months'] = ((df['Date'] - birth_date).dt.days / 30.44).round(1)
    percentiles_df = pd.read_csv('data/weight-girls-percentiles-who.csv')

    # Plot using Plotly
    fig = go.Figure()

    # Add baby's height line
    fig.add_trace(go.Scatter(
        x=df['Months'], y=df['Weight'],
        mode='lines+markers',
        name="Baby's Weight", line=dict(color='black', width=3),
        hovertemplate=
            'Date: %{customdata[0]}<br>' +
            'Weight: %{y:.1f} kg<br>',
        customdata=df[['Date']].values
    ))

    # Add percentile curves
    for p in ['P1', 'P5', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99']:
        fig.add_trace(go.Scatter(
            x=percentiles_df['Months'],
            y=percentiles_df[p],
            mode='lines',
            name=f'{p} Percentile',
            line=dict(dash='dot')
        ))

    # Layout
    fig.update_layout(
        title="Baby's Weight vs Age (with Percentiles)",
        xaxis_title="Age (months)",
        yaxis_title="Weight (kg)",
        legend_title="Legend",
        height=500
    )

    st.plotly_chart(fig)

def head_chart():
    # Load baby's height data from CSV
    df = filter_csv_by_category(df, 'Growth')
    
    df['Date'] = pd.to_datetime(df['Start'], errors='coerce')
    df['Head'] = df['End Condition'].str.replace('cm', '', regex=False).astype(float)

    df = df.dropna(subset=['Date', 'Head'])

    # Calculate age in months (or days)
    birth_date = df['Date'].min()  # You might want to store baby's DOB somewhere
    df['Months'] = ((df['Date'] - birth_date).dt.days / 30.44).round(1)
    percentiles_df = pd.read_csv('data/head-girls-percentiles-who.csv')

    # Plot using Plotly
    fig = go.Figure()

    # Add baby's height line
    fig.add_trace(go.Scatter(
        x=df['Months'], y=df['Head'],
        mode='lines+markers',
        name="Baby's Head Circumference", line=dict(color='black', width=3),
        hovertemplate=
            'Date: %{customdata[0]}<br>' +
            'Head Circumference: %{y:.1f} cm<br>',
        customdata=df[['Date']].values
    ))

    # Add percentile curves
    for p in ['P1', 'P5', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99']:
        fig.add_trace(go.Scatter(
            x=percentiles_df['Months'],
            y=percentiles_df[p],
            mode='lines',
            name=f'{p} Percentile',
            line=dict(dash='dot')
        ))

    # Layout
    fig.update_layout(
        title="Baby's Head Circumference vs Age (with Percentiles)",
        xaxis_title="Age (months)",
        yaxis_title="Head Circumference (cm)",
        legend_title="Legend",
        height=500
    )

    st.plotly_chart(fig)

def calendar_filter(): 
    # Options for quick date selections
    options = ['Last 7 days', 'Last 30 days', 'Last 90 days', 'All Time', 'Custom range']
    selection = st.selectbox("Select date range", options)

    # Get date range
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
            days = 1000
        else:
            days = int(selection.split()[1])
        start_date = today - datetime.timedelta(days=days)
        end_date = today

    if selection != 'Custom range' and selection != 'All Time':
        st.info(f"📅 Selected date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}") 
    
    return start_date, end_date

def polar_chart(data_frame):
    # Inputs (you can also use st.time_input or st.file_uploader)
    dormido = time(19, 30)
    desperto = time(20, 15)

    single_events = ['Woke up', 'Bed time', 'Diaper']
    segment_events = ['Sleep', 'Night waking']

    def to_decimal(t):
        return t.hour + t.minute / 60

    # Convert hour to polar angle (in degrees)
    def hora_a_ángulo(hora_decimal):
        return (90 - (hora_decimal % 24) * 15) % 360

    eventos = {}       

    colores = {
        "Woke up": "gold",
        "Bed time": "orange",
        "Sleep": "royalblue",
        "Night waking": "red"
    }    

    fig = go.Figure()

    for event in single_events:        
        event_df = filter_csv_by_category(data_frame, event)

        # Validate DataFrame
        if event_df.empty or 'Start' not in event_df.columns:
            st.warning(f"No '{event}' data available.")
            continue

        event_df['Start'] = pd.to_datetime(event_df['Start'], errors='coerce')
        event_df = event_df.dropna(subset=['Start'])

        # Add all valid times to the eventos dictionary
        counter = 1
        for _, row in event_df.iterrows():
            key = event if len(event_df) <= 1 else f"{event} {counter}" 
            if (event == "Diaper"):
                colores[key] = "pink"
            if (event == 'Feed' and row['Start Location'] == 'Bottle'):
                colores[key] = "lime"
           
            eventos[key] = to_decimal(row['Start'].time())
            counter += 1 

    for event in segment_events:
        event_df = filter_csv_by_category(data_frame, event)

        if event_df.empty or 'Start' not in event_df.columns:
            st.warning(f"No '{event}' data available.")
            continue

        event_df['Start'] = pd.to_datetime(event_df['Start'], errors='coerce')
        event_df['End'] = pd.to_datetime(event_df['End'], errors='coerce')
        event_df = event_df.dropna(subset=['Start', 'End'])

        for _, row in event_df.iterrows():
            start = row['Start'].time()
            end = row['End'].time()

            # Convert to hour decimals
            start_h = to_decimal(start)
            end_h = to_decimal(end)

            # Handle times that pass midnight
            if end_h < start_h:
                end_h += 24

            # Create points between start and end to draw the arc
            hours_range = np.linspace(start_h, end_h, 50)
            angles = [hora_a_ángulo(h % 24) for h in hours_range]
            radii = [1] * len(angles)

            fig.add_trace(go.Scatterpolar(
                r=radii,
                theta=angles,
                mode='lines',
                line=dict(color=colores[event], width=10),  # Thinner than base ring
                name=event,
                showlegend=True
            ))

    '''
    # Base circular "reloj"
    horas = np.linspace(hora_a_ángulo(to_decimal("19:00")), 30, 200)
    ángulos = [(90 - h * 15 % 360) for h in horas]
    radios = [1] * len(horas)

    # Reloj base
    fig.add_trace(go.Scatterpolar(
        r=radios,
        theta=ángulos,
        mode='lines',
        line=dict(color='royalblue', width=20),
        hoverinfo='skip',
        showlegend=True,
        name="Night Sleep"
    ))
    '''

    # Añadir eventos
    for evento, hora in eventos.items():
        ángulo = hora_a_ángulo(hora)
        fig.add_trace(go.Scatterpolar(
            r=[1],
            theta=[ángulo],
            mode='markers+text',
            marker=dict(color=colores[evento], size=30),
            text=[evento],
            textposition='top center',
            name=evento
        ))

    # Estética
    fig.update_layout(
        height=600,
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(
                direction='counterclockwise',
                rotation=180,
                tickmode='array',
                tickvals=[hora_a_ángulo(h) for h in range(0, 24)],
                ticktext=[f"{h%24}:00" for h in range(0, 24)],
            )
        ),
        showlegend=True,
        margin=dict(t=20, b=20, l=0, r=0)
    )

    st.plotly_chart(fig, use_container_width=False)

st.title("👶 Baby Tracker Dashboard")

with st.sidebar:
    menu_id = sac.menu(
        items=[
            sac.MenuItem('Overview', icon='bar-chart'),
            sac.MenuItem('Growth', icon='line-chart'),            
            sac.MenuItem('Sleep', icon='clock-circle'),
            sac.MenuItem('Feeding', icon='coffee'),
            sac.MenuItem('Diapers', icon='smile'),
        ],
        open_all=True,
        index=0,
    )

ona_days_old = (datetime.date.today() - datetime.date(2025, 1, 15)).days
st.markdown(f'<h3 style="color: gray;">Ona is {days_to_months_days(ona_days_old)} old.</h3>', unsafe_allow_html=True)
st.markdown(f'<small style="color: gray;">(Last Update: {df["Start"].max()})</small>', unsafe_allow_html=True)

filter_csv_by_category(df, 'Sleep')

if menu_id == 'Overview':
    st.header("📊 Overview")
    st.write("Show a summary of baby data here.")
    custom_range = st.date_input(
        "Select custom date range",
        value=today,
        max_value=today
    )
    df = filter_csv_by_start_date(df, custom_range)
    polar_chart(df)

elif menu_id == 'Growth':
    st.header("📈 Growth Tracking")
    st.write(" These standards were developed using data collected in the WHO Multicentre Growth Reference Study. The site presents documentation on how the physical growth curves and motor milestone windows of achievement were developed as well as application tools to support the implementation of the standards")

    tab1, tab2, tab3 = st.tabs(["Height", "Weight", "Head Circumference"])

    with tab1:
        st.write('https://www.who.int/tools/child-growth-standards/standards/length-height-for-age')
        height_chart()

    with tab2:
        st.write('https://www.who.int/tools/child-growth-standards/standards/weight-for-age')
        weight_chart()

    with tab3:
        st.write('https://www.who.int/tools/child-growth-standards/standards/head-circumference-for-age')
        head_chart()

elif menu_id == 'Sleep':
    st.header("💤 Sleep Patterns")
    st.write("Visualize sleep logs or averages.")
    start_date, end_date = calendar_filter()

    tab1, tab2, tab3, tab4 = st.tabs(["Sleep/Awake Chart", "Daily Sleep Trends", "Nap Duration", "Nighttime Wakeups"])

    with tab1:
        st.write("🛌 Sleep/Awake Chart")
        st.write("This heatmap visualizes the baby's sleep and wake patterns, with each row representing a date and each column representing an hourly time slot. The chart uses color coding to distinguish between sleep and awake states: 'royalblue' indicates sleep, while 'lightblue' represents awake periods. <br><br>The heatmap helps identify sleep patterns, showing when the user was asleep or awake throughout the day and night.<br>", unsafe_allow_html=True)
        sleep_heatmap()  
    
    with tab2:
        st.write("🌞 Daily Sleep Trends")
        fig2 = px.bar(sleep_data, x='Date', y='Nap Duration (hrs)', title='Daytime Nap Duration')
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.write("🌞 Nap Duration")
        fig3 = px.bar(sleep_data, x='Date', y='Nap Duration (hrs)', title='Daytime Nap Duration2')
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.write('🌙 Nighttime Wakeups lorem ipsum')
        fig4 = px.line(sleep_data, x='Date', y='Night Wakeups', markers=True, title='Night Wakeups Per Day')
        st.plotly_chart(fig4, use_container_width=True)
 

elif menu_id == 'Feeding':
    st.header("🍼 Feeding Times")
    st.write("Display feeding sessions, types, and intervals.")
    start_date, end_date = calendar_filter()
    breastfeeding_chart(start_date, end_date)
    breastfeeding_duration_chart(start_date, end_date)

elif menu_id == 'Diapers':
    st.header("🧷 Diaper Changes")
    st.write("Track diaper change frequency and types.")
    start_date, end_date = calendar_filter()

    diaper_charts(start_date, end_date)

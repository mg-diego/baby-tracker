import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px       # <--- Esta es la que falta
import plotly.graph_objects as go
from datetime import timedelta, datetime
from data_manager import DataManager

class FeedingSection:
    def __init__(self, df, start_date, end_date):
        self.df = df
        self.start_date = start_date
        self.end_date = end_date

    def render(self):
        st.header("🍼 Feeding Times")
        st.write("Display feeding sessions, types, and intervals.")

        tab1, tab2, = st.tabs([
            "Breastfeeding", 
            "Bottle Feeding"
        ])

        with tab1:
            st.write("📊 Breastfeeding Statistics")
            # Gráficos existentes
            self._breastfeeding_chart() # Lado Izq/Der
            
            st.divider()
            
            st.subheader("☀️ Day vs 🌙 Night")
            st.caption(" Comparison of feeding duration between Day (07:00-19:00) and Night (19:00-07:00).")
            self._day_night_comparison() # <--- NUEVO: Ritmo Circadiano
            
            st.divider()
            
            self._breastfeeding_intervals_chart()

        with tab2:
            st.write("🍼 Bottle Feeding Trends")
            self._bottle_intake_chart()
            
            st.divider()

            # --- NUEVO: INTERVALOS ---
            st.subheader("⏳ Time Between Bottles")
            st.caption("Are intervals getting longer and more consistent?")
            self._bottle_interval_trend()

            st.divider()

            # 2. Gráfica de Patrones (Hora vs Cantidad)
            st.subheader("🕑 Feed Sizes & Schedule")
            st.caption("Distribution of bottle sizes throughout the day. Are morning feeds larger?")
            self._bottle_scatter_chart()
            
            st.divider()
            
            # 3. Frecuencia
            st.subheader("📊 Frequency")
            self._bottle_frequency_chart()

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

    def _day_night_comparison(self):
        feed_df = DataManager.filter_by_category(self.df, 'Feed')
        feed_df = DataManager.filter_by_date_range(feed_df, self.start_date, self.end_date)
        feed_df = feed_df[feed_df['Start Location'].str.lower() == 'breast']
        feed_df = feed_df.dropna(subset=['Start', 'End'])

        if feed_df.empty: return

        # Calcular duración en horas
        feed_df['DurationHours'] = (feed_df['End'] - feed_df['Start']).dt.total_seconds() / 3600
        
        # Definir Día vs Noche (Ejemplo: Día 07:00 a 19:00)
        # Usamos una función lambda rápida para etiquetar
        feed_df['Period'] = feed_df['Start'].dt.hour.apply(
            lambda h: 'Day (07-19)' if 7 <= h < 19 else 'Night (19-07)'
        )
        feed_df['Date'] = feed_df['Start'].dt.date

        # Agrupar
        daily_period = feed_df.groupby(['Date', 'Period'])['DurationHours'].sum().unstack(fill_value=0)

        # Asegurar que existan ambas columnas aunque sean 0
        if 'Day (07-19)' not in daily_period.columns: daily_period['Day (07-19)'] = 0
        if 'Night (19-07)' not in daily_period.columns: daily_period['Night (19-07)'] = 0

        feed_df['Duration'] = (feed_df['End'] - feed_df['Start']).dt.total_seconds() / 3600
        duration_per_day = feed_df.groupby(feed_df['Start'].dt.date)['Duration'].sum()
        rolling_avg = duration_per_day.rolling(window=7, min_periods=1).mean()

        # Crear Gráfico
        fig = go.Figure()        
        fig.add_trace(go.Scatter(x=rolling_avg.index, y=rolling_avg.values, name="7-Day Avg", mode='lines+markers', line=dict(color='darkorange', width=2)))
        
        # Barra Día (Naranja/Amarillo suave)
        fig.add_trace(go.Bar(
            x=daily_period.index, 
            y=daily_period['Day (07-19)'], 
            name='Day',
            marker_color='#f4a261' 
        ))

        # Barra Noche (Azul oscuro/Morado)
        fig.add_trace(go.Bar(
            x=daily_period.index, 
            y=daily_period['Night (19-07)'], 
            name='Night',
            marker_color='#264653'
        ))

        fig.update_layout(
            title="Day vs. Night Consumption (Total Hours)",
            barmode='stack', # Apilado para ver el volumen total también
            yaxis_title="Hours Breastfeeding",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    def _get_bottle_data(self):
        """Helper para extraer y limpiar datos de biberones."""
        df = DataManager.filter_by_category(self.df, 'Feed')
        df = DataManager.filter_by_date_range(df, self.start_date, self.end_date)
        
        if 'Start Location' in df.columns:
            df = df[df['Start Location'].str.lower() == 'bottle']
        
        col_amount = 'Duration' if 'Duration' in df.columns else 'Quantity' # Fallback
        
        if col_amount not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        
        def parse_amount(val):
            if pd.isna(val): return 0.0
            s = str(val).lower().replace('ml', '').replace('oz', '').strip()
            try:
                return float(s)
            except:
                return 0.0        
        
        df['AmountNum'] = df[col_amount].apply(parse_amount)
        df = df[df['AmountNum'] > 0]
        
        return df

    def _bottle_intake_chart(self):
        df = self._get_bottle_data()
        
        if df.empty:
            st.info("No bottle data available with amounts.")
            return

        # Agrupar por día
        df['Date'] = df['Start'].dt.date
        daily_vol = df.groupby('Date')['AmountNum'].sum().reset_index()
        
        # Calcular media móvil (7 días)
        daily_vol['Date'] = pd.to_datetime(daily_vol['Date'])
        daily_vol = daily_vol.set_index('Date')
        idx = pd.date_range(daily_vol.index.min(), daily_vol.index.max())
        daily_vol = daily_vol.reindex(idx, fill_value=0).reset_index().rename(columns={'index': 'Date'})
        daily_vol['Date'] = daily_vol['Date'].dt.date
        
        daily_vol['RollingAvg'] = daily_vol['AmountNum'].rolling(window=7, min_periods=1).mean()

        # Gráfico
        fig = go.Figure()
        
        # Barras (Volumen Diario)
        fig.add_trace(go.Bar(
            x=daily_vol['Date'],
            y=daily_vol['AmountNum'],
            name='Daily Volume (ml)',
            marker_color='#87CEEB', # SkyBlue
            hovertemplate='<b>%{x}</b><br>Total: %{y:.0f} ml<extra></extra>'
        ))
        
        # Línea (Tendencia)
        fig.add_trace(go.Scatter(
            x=daily_vol['Date'],
            y=daily_vol['RollingAvg'],
            mode='lines',
            name='7-Day Avg',
            line=dict(color='royalblue', width=3),
            hoverinfo='skip'
        ))

        # Calcular media total para referencia visual
        avg_total = daily_vol['AmountNum'][daily_vol['AmountNum'] > 0].mean()

        fig.update_layout(
            title=f"Total Daily Intake (Avg: {avg_total:.0f} ml)",
            xaxis_title="Date",
            yaxis_title="Volume (ml)",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _bottle_scatter_chart(self):
        df = self._get_bottle_data()
        if df.empty: return

        df['Date'] = df['Start'].dt.date
        df['Hour'] = df['Start'].dt.hour + df['Start'].dt.minute/60
        df['TimeStr'] = df['Start'].dt.strftime('%H:%M')

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Hour'],
            y=df['AmountNum'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['Hour'], # Color según hora del día
                colorscale='Temps', # Escala azul-roja (frío noche, cálido día)
                showscale=False,
                opacity=0.7
            ),
            customdata=np.stack((df['Date'], df['TimeStr']), axis=-1),
            hovertemplate='<b>%{customdata[1]}</b><br>Amount: %{y} ml<br>Date: %{customdata[0]}<extra></extra>'
        ))

        fig.update_layout(
            xaxis=dict(
                title="Time of Day",
                range=[0, 24],
                tickmode='array',
                tickvals=[0, 4, 8, 12, 16, 20, 24],
                ticktext=['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00']
            ),
            yaxis_title="Amount (ml)",
            height=400,
            margin=dict(l=0, r=0, t=20, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

    def _bottle_frequency_chart(self):
        df = self._get_bottle_data()
        if df.empty: return

        df['Date'] = df['Start'].dt.date
        daily_counts = df.groupby('Date').size().reset_index(name='Count')
        
        # Reindexar para mostrar días con 0
        daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])
        daily_counts = daily_counts.set_index('Date')
        idx = pd.date_range(daily_counts.index.min(), daily_counts.index.max())
        daily_counts = daily_counts.reindex(idx, fill_value=0).reset_index().rename(columns={'index': 'Date'})
        daily_counts['Date'] = daily_counts['Date'].dt.date

        fig = px.bar(
            daily_counts, 
            x='Date', 
            y='Count', 
            title='Number of Bottles per Day',
        )
        fig.update_traces(marker_color='#4682B4') # SteelBlue
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(tickmode='linear', dtick=1) # Forzar enteros en eje Y
        )
        st.plotly_chart(fig, use_container_width=True)

    def _bottle_interval_trend(self):
        df = self._get_bottle_data()
        if df.empty: return

        # Ordenar cronológicamente
        df = df.sort_values(by='Start')
        
        # Calcular diferencia con la fila anterior (en horas)
        df['IntervalHours'] = df['Start'].diff().dt.total_seconds() / 3600
        
        # Filtrar intervalos lógicos:
        # - Menor de 12 horas (para no contar la noche como un intervalo diurno normal)
        # - Mayor de 1 hora (para no contar una pausa breve como nueva toma)
        df_intervals = df[(df['IntervalHours'] > 0.5) & (df['IntervalHours'] < 10)].copy()
        
        if df_intervals.empty:
            st.info("Not enough sequential data for intervals.")
            return

        df_intervals['Date'] = df_intervals['Start'].dt.date
        
        # Agrupar por día para ver el promedio diario
        daily_avg = df_intervals.groupby('Date')['IntervalHours'].mean().reset_index()
        
        # Tendencia suavizada
        daily_avg['Trend'] = daily_avg['IntervalHours'].rolling(window=7, min_periods=1).mean()

        fig = go.Figure()

        # Puntos reales (dispersos) para ver la variabilidad
        fig.add_trace(go.Scatter(
            x=df_intervals['Start'],
            y=df_intervals['IntervalHours'],
            mode='markers',
            name='Individual Feed',
            marker=dict(color='lightgray', size=4, opacity=0.5),
            hoverinfo='skip'
        ))

        # Línea de promedio diario
        fig.add_trace(go.Scatter(
            x=daily_avg['Date'],
            y=daily_avg['IntervalHours'],
            mode='lines+markers',
            name='Daily Avg Interval',
            line=dict(color='#4682B4', width=2), # SteelBlue
            hovertemplate='<b>%{x}</b><br>Avg Interval: %{y:.1f} h<extra></extra>'
        ))

        # Línea de tendencia
        fig.add_trace(go.Scatter(
            x=daily_avg['Date'],
            y=daily_avg['Trend'],
            mode='lines',
            name='7-Day Trend',
            line=dict(color='navy', width=3, dash='dot'),
            hoverinfo='skip'
        ))

        fig.update_layout(
            title="Gap Between Feeds Evolution",
            xaxis_title="Date",
            yaxis_title="Hours Between Bottles",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)
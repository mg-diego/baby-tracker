import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta, datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from data_manager import DataManager

# --- FUNCIONES AUXILIARES ---

def _format_duration_str(seconds):
    """Convierte segundos a formato '1h 30m' o '45m'"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"

def _prepare_sleep_gantt_data(main_df):
    """
    Versión actualizada para dibujar barras ROJAS en Night Waking.
    """
    cats = ['Sleep', 'Bed time', 'Woke up', 'Night waking']
    
    # Limpieza previa
    if 'Type' in main_df.columns:
        main_df['Type'] = main_df['Type'].fillna("").astype(str).str.strip()
    if 'Notes' not in main_df.columns:
        main_df['Notes'] = ""
    main_df['Notes'] = main_df['Notes'].fillna("").astype(str).str.strip()

    df = main_df[main_df['Type'].isin(cats)].copy()
    
    if df.empty: return pd.DataFrame()

    df['Start'] = pd.to_datetime(df['Start'])
    df['End'] = pd.to_datetime(df['End'])

    # Prioridad visual (Order):
    # Queremos que el despertar se pinte "encima" si hay solapamiento, 
    # aunque en teoría no debería haber solapamiento temporal.
    priority_map = {'Bed time': 0, 'Sleep': 1, 'Night waking': 2, 'Woke up': 3}
    df['Order'] = df['Type'].map(priority_map).fillna(1)

    df = df.sort_values(by=['Start', 'Order']).reset_index(drop=True)

    processed_rows = []
    
    is_night_mode = False
    night_sleep_buffer = [] 
    has_waking = False
    last_sleep_end = None
    
    for _, row in df.iterrows():
        evt_type = row['Type']
        note_val_lower = row['Notes'].lower()
        
        if evt_type == 'Bed time':
            is_night_mode = True
            night_sleep_buffer = []
            has_waking = False
            
        elif evt_type == 'Woke up':
            if is_night_mode:
                quality = "🟣 Night Sleep (Solid)" if (not has_waking and len(night_sleep_buffer) == 1) else "🔵 Night Sleep (Interrupted)"
                for s_row, s_wake_str in night_sleep_buffer:
                    _process_visual_row(s_row, quality, processed_rows, s_wake_str)
                is_night_mode = False
                night_sleep_buffer = []
            else:
                is_night_mode = False

        elif evt_type == 'Night waking':
            if is_night_mode:
                has_waking = True
            
            # --- CAMBIO AQUÍ: DIBUJAR LA BARRA ROJA ---
            # Procesamos visualmente este evento como una barra independiente
            _process_visual_row(row, "🔴 Night Waking", processed_rows, wake_window_str="-")
            # ------------------------------------------

        elif evt_type == 'Sleep':
            wake_window_str = "-"
            if last_sleep_end is not None:
                diff_seconds = (row['Start'] - last_sleep_end).total_seconds()
                if diff_seconds > 0:
                    wake_window_str = _format_duration_str(diff_seconds)
            
            last_sleep_end = row['End']

            if 'nap' in note_val_lower:
                _process_visual_row(row, "🌫️ Nap", processed_rows, wake_window_str)
            elif 'night sleep' in note_val_lower:
                if is_night_mode:
                    night_sleep_buffer.append((row, wake_window_str))
                else:
                    _process_visual_row(row, "🔵 Night Sleep (Interrupted)", processed_rows, wake_window_str)
            else:
                if is_night_mode:
                    night_sleep_buffer.append((row, wake_window_str))
                else:
                    _process_visual_row(row, "🌫️ Nap", processed_rows, wake_window_str)

    if is_night_mode and night_sleep_buffer:
        status = "Night Sleep (Interrupted)" if has_waking else "Night Sleep (Solid)"
        for s_row, s_wake_str in night_sleep_buffer:
             _process_visual_row(s_row, status, processed_rows, s_wake_str)

    return pd.DataFrame(processed_rows)

def _process_visual_row(row, status, processed_rows_list, wake_window_str="-"):
    """
    Añade la fila visual con el nuevo parámetro wake_window_str
    """
    s_start = row['Start']
    s_end = row['End']
    
    if pd.isnull(s_start) or pd.isnull(s_end):
        return

    total_seconds = (s_end - s_start).total_seconds()
    if total_seconds <= 0: return
    total_duration_str = _format_duration_str(total_seconds)

    def add_item(date_val, start_h, duration_h, s_str, e_str):
        processed_rows_list.append({
            'Date': date_val,
            'StartHour': start_h,
            'Duration': duration_h,
            'StartTimeStr': s_str,
            'EndTimeStr': e_str,
            'TotalDurationStr': total_duration_str,
            'Status': status,
            'Color': _get_color_by_status(status),
            'WakeWindow': wake_window_str  # <--- NUEVO CAMPO
        })

    if s_start.date() == s_end.date():
        duration_h = (s_end - s_start).total_seconds() / 3600
        add_item(s_start.date(), s_start.hour + s_start.minute/60, duration_h, 
                 s_start.strftime('%H:%M'), s_end.strftime('%H:%M'))
    else:
        midnight_day1 = (s_start + timedelta(days=1)).replace(hour=0, minute=0, second=0)
        duration_1 = (midnight_day1 - s_start).total_seconds() / 3600
        add_item(s_start.date(), s_start.hour + s_start.minute/60, duration_1, 
                 s_start.strftime('%H:%M'), "24:00")
        
        midnight_day2 = s_end.replace(hour=0, minute=0, second=0)
        duration_2 = (s_end - midnight_day2).total_seconds() / 3600
        if duration_2 > 0:
            add_item(s_end.date(), 0, duration_2, "00:00", s_end.strftime('%H:%M'))

def _get_color_by_status(status):
    if status == "🟣 Night Sleep (Solid)":
        return "#8A2BE2" # BlueViolet
    elif status == "🔵 Night Sleep (Interrupted)":
        return "royalblue"
    elif status == "🔴 Night Waking": # <--- NUEVO CASO
        return "#b00000" # OrangeRed (Rojo vibrante)
    else:
        return "#ADD8E6" # LightBlue (Nap)

# --- CLASE PRINCIPAL ---

class SleepSection:
    def __init__(self, df, start_date, end_date):
        self.df = df
        self.start_date = start_date
        self.end_date = end_date

    def render(self):
        st.header("💤 Sleep Patterns")
        st.write("Visualize sleep logs or averages.")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Sleep Timeline", 
            "Sleep Trends", 
            "Nap Trends", 
            "Nighttime Wakeups"
        ])

        with tab1:
            st.write("🛌 Sleep Timeline")
            st.caption("Blue bars indicate sleep periods.")
            self._sleep_timeline_chart()
        
        with tab2:
            st.write("💤 Sleep Trends")
            col_ctrl, _ = st.columns([1, 3])
            with col_ctrl:
                show_labels = st.toggle("Show chart values", value=True)  
            self._sleep_per_day_chart(show_text=show_labels)
            self._wake_time_per_day_chart(show_text=show_labels)
            self._bed_time_per_day_chart(show_text=show_labels)

        with tab3:
            st.write("🌞 Nap Trends")
            self._naps_stacked_chart()

            st.subheader("Wake Windows Evolution")
            st.caption("Time spent awake between naps (Start of day -> Nap 1 -> Nap 2 ... -> Bed time).")
            self._wake_window_evolution_chart()

        with tab4:
            st.subheader("Impact (Total Duration)")
            st.caption("Total minutes spent awake during the night (sum of all disruptions).")
            self._night_waking_duration_chart()
            
            st.divider()
            
            col_chart, col_controls = st.columns([5, 1], gap="medium")
            st.subheader("Patterns & Severity")
            st.caption("Larger/Redder bubbles indicate longer wake periods. Use the slider to filter by duration.")
            self._night_waking_scatter_chart()

            st.divider()
            st.subheader('The "Cursed Hour" 🕰️')
            st.caption("Distribution of wakings by hour. The darkest bar indicates the most frequent waking time.")
            self._cursed_hour_histogram()

    def _sleep_timeline_chart(self):
        # 1. Filtrar datos
        df_filtered = DataManager.filter_by_date_range(self.df, self.start_date, self.end_date)
        gantt_data = _prepare_sleep_gantt_data(df_filtered)
        
        if gantt_data.empty:
            st.info("No sleep data available.")
            return

        # --- NUEVO: SWITCH MODO NOCHE ---
        # Colocamos el toggle arriba a la derecha o izquierda
        col_tog, _ = st.columns([1, 4])
        with col_tog:
            is_night_mode = st.toggle("🌙 Night Mode", value=False, help="Centers the chart around the night and zooms in on sleep hours.")

        # --- LÓGICA DE TRANSFORMACIÓN DE DATOS ---
        # Trabajamos sobre una copia para no romper nada
        plot_data = gantt_data.copy()
        
        # Configuración de Ejes por defecto (Modo Día 0-24)
        xaxis_range = [0, 24]
        tick_vals = [0, 4, 8, 12, 16, 20, 24]
        tick_text = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00']

        if is_night_mode:
            # 1. Transformar las horas de la madrugada (AM)
            # Si una tarea empieza antes de las 12:00 del mediodía, asumimos que pertenece 
            # visualmente a la "noche extendida" del día anterior.
            # Ejemplo: 02:00 AM pasa a ser la hora 26:00 del día previo.
            
            # Identificamos filas de la madrugada (StartHour < 12)
            am_mask = plot_data['StartHour'] < 12
            
            # Desplazamos la hora +24
            plot_data.loc[am_mask, 'StartHour'] += 24
            
            # Desplazamos la fecha -1 día para que se agrupe con la noche anterior
            plot_data.loc[am_mask, 'Date'] -= timedelta(days=1)
            
            # 2. Calcular el rango dinámico (Filtro del eje X)
            # Buscamos el Bed time más temprano (mínimo) y el Woke up más tardío (máximo extendido)
            
            # Mínimo: Consideramos horas a partir de las 12:00 (mediodía original)
            # para evitar coger siestas de la mañana como inicio de la noche.
            valid_starts = plot_data['StartHour']
            min_x = valid_starts.min()
            
            # Máximo: StartHour + Duration
            max_x = (plot_data['StartHour'] + plot_data['Duration']).max()
            
            # Añadimos un pequeño margen (pad) de 30 mins (0.5h)
            min_x =  max(12, int(min_x) - 0.5) # No bajar de 12:00
            max_x =  int(max_x) + 0.5
            
            xaxis_range = [min_x, max_x]
            
            # 3. Generar Ticks bonitos para el eje extendido
            # Creamos ticks cada 2 horas dentro del rango
            tick_vals = list(range(int(min_x), int(max_x) + 2, 2))
            # La función lambda convierte "26" en "02:00"
            tick_text = [f"{(h-24):02d}:00" if h >= 24 else f"{h:02d}:00" for h in tick_vals]

        # --- NUEVO: LÓGICA DE SEPARADORES DE MES ---
        # Obtenemos las fechas únicas ordenadas cronológicamente
        unique_dates = sorted(gantt_data['Date'].unique())
        
        month_lines = []
        month_annotations = []
        
        # Iteramos para encontrar dónde cambia el mes
        for i in range(1, len(unique_dates)):
            curr_date = unique_dates[i]
            prev_date = unique_dates[i-1]
            
            # Si el mes cambia respecto al día anterior
            if curr_date.month != prev_date.month:
                # En un eje categórico, la posición entre el índice i-1 y el i es i-0.5
                line_pos = i - 0.5
                
                # Etiqueta (Ej: "February 2025")
                month_label = curr_date.strftime("%B %Y")
                
                # 1. La Línea Horizontal
                month_lines.append(dict(
                    type="line",
                    x0=0, x1=24,       # De 00:00 a 24:00
                    y0=line_pos, y1=line_pos,
                    line=dict(color="black", width=1, dash="dash"),
                    layer="below"      # Detrás de las barras
                ))
                
                # 2. El Texto del Mes
                mid_point = (xaxis_range[0] + xaxis_range[1]) / 2
                month_annotations.append(dict(
                    x=mid_point,              # Centrado en el mediodía
                    y=line_pos,
                    text=month_label,
                    showarrow=False,
                    yshift=0,
                    font=dict(color="black", size=10, weight="bold"),
                    # Fondo blanco semi-transparente para que se lea si cae sobre una barra azul
                    bgcolor="rgba(255, 255, 255, 0.8)", 
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4
                ))

        # --- CREACIÓN DEL GRÁFICO ---
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=plot_data['Date'],
            x=plot_data['Duration'],
            base=plot_data['StartHour'],
            orientation='h',
            marker=dict(
                color=plot_data['Color'],
                line=dict(width=0)
            ),
            name='Sleep',
            customdata=np.stack((
                plot_data['StartTimeStr'],      # 0
                plot_data['EndTimeStr'],        # 1
                plot_data['TotalDurationStr'],  # 2
                plot_data['Status'],            # 3
                plot_data['WakeWindow']         # 4
            ), axis=-1),

            hovertemplate=(
                '<b>%{y}</b><br>' +
                '%{customdata[3]}<br>' +
                'Time: %{customdata[0]} - %{customdata[1]}<br>' +
                'Duration: %{customdata[2]}<br>' +
                '<b>Awake Before: %{customdata[4]}</b>' +
                '<extra></extra>'
            )
        ))

        fig.update_layout(
            xaxis=dict(
                title="Time of Day",
                range=xaxis_range, # Rango dinámico
                tickmode='array',
                tickvals=tick_vals, # Ticks dinámicos
                ticktext=tick_text, # Etiquetas dinámicas (26 -> 02:00)
                side='top',
                showgrid=True,
                gridcolor='rgba(200,200,200,0.2)',
                gridwidth=1,
                fixedrange=True,
                showspikes=True, 
                spikemode='across', 
                spikesnap='cursor', 
                showline=True, 
                spikecolor="black"
            ),
            yaxis=dict(
                title=None,
                type='category',
                fixedrange=True
            ),
            shapes=month_lines,
            annotations=month_annotations,
            height=max(400, 25 * len(unique_dates)),
            margin=dict(l=0, r=10, t=30, b=0),
            showlegend=False,
            hovermode="closest"
        )
        
        st.caption("🟣 Solid Night | 🔵 Interrupted Night | 🔴 Night Waking | 🌫️ Nap")
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})

    def _sleep_per_day_chart(self, show_text=True):
        df = DataManager.filter_by_category(self.df, 'Sleep')
        df = DataManager.filter_by_date_range(df, self.start_date, self.end_date)
        df = df.dropna(subset=['Start', 'End'])
        
        if df.empty:
            return

        df['DurationHours'] = (df['End'] - df['Start']).dt.total_seconds() / 3600
        df['Date'] = df['Start'].dt.date
        daily_sleep = df.groupby('Date')['DurationHours'].sum().reset_index()

        avg_sleep = daily_sleep['DurationHours'].mean()

        mode_val = 'lines+markers+text' if show_text else 'lines+markers'
        text_tpl = '%{y:.1f} h' if show_text else None
        hover_info = 'skip' if show_text else None 
        hover_tpl = None if show_text else '<b>%{x}</b><br>Total: %{y:.1f} h<extra></extra>'

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily_sleep['Date'],
            y=daily_sleep['DurationHours'],
            mode=mode_val,
            line_shape='spline',
            fill='tozeroy',
            line=dict(color='skyblue'),
            marker=dict(color='steelblue'),
            name='Total Sleep',texttemplate=text_tpl,
            textposition="top center",
            hoverinfo=hover_info,
            hovertemplate=hover_tpl
        ))

        fig.add_trace(go.Scatter(
            x=daily_sleep['Date'],
            y=[avg_sleep] * len(daily_sleep),
            mode='lines',
            line=dict(color='darkgray', width=2, dash='dot'),
            name=f'Average ({avg_sleep:.1f}h)',
            hoverinfo='skip'
        ))

        fig.update_layout(
            title='Total Sleep per Day',
            xaxis_title='Date',
            yaxis_title='Hours',
            yaxis=dict(range=[0, 24]),
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        col1, col2 = st.columns([4, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.caption("Data:")
            st.dataframe(daily_sleep, height=400)

    def _wake_time_per_day_chart(self, show_text=True):
        df = DataManager.filter_by_category(self.df, 'Woke up')
        df = DataManager.filter_by_date_range(df, self.start_date, self.end_date)
        
        if df.empty:
            st.info("No 'Woke up' events recorded.")
            return

        df = df.copy()
        df['Date'] = df['Start'].dt.date
        df['HourDecimal'] = df['Start'].dt.hour + df['Start'].dt.minute / 60
        df['HourStr'] = df['Start'].dt.strftime('%H:%M')

        avg_val = df['HourDecimal'].mean()
        avg_str = f"{int(avg_val):02}:{int((avg_val % 1) * 60):02}"
        wake_time = df.groupby('Date')['HourStr'].sum().reset_index()

        mode_val = 'lines+markers+text' if show_text else 'lines+markers'
        text_tpl = '%{customdata}' if show_text else None
        hover_info = 'skip' if show_text else None
        hover_tpl = None if show_text else '<b>%{x}</b><br>Time: %{customdata}<extra></extra>'

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['HourDecimal'],
            mode=mode_val,
            line_shape='spline',
            line=dict(color='orange', width=2),
            marker=dict(color='darkorange'),
            name='Wake Up',
            texttemplate=text_tpl,
            textposition="top center",
            hoverinfo=hover_info,
            hovertemplate=hover_tpl,
            customdata=df['HourStr']
        ))

        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=[avg_val] * len(df),
            mode='lines',
            line=dict(color='gray', dash='dot'),
            name=f'Avg: {avg_str}h',
            hoverinfo='skip'
        ))

        min_h = int(df['HourDecimal'].min()) - 1
        max_h = int(df['HourDecimal'].max()) + 1

        fig.update_layout(
            title='Wake Up Time',
            yaxis=dict(
                range=[min_h, max_h],
                tickmode='array',
                tickvals=list(range(min_h, max_h+1)),
                ticktext=[f"{h:02}:00" for h in range(min_h, max_h+1)]
            ),
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        col1, col2 = st.columns([4, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.caption("Data:")
            st.dataframe(wake_time, height=400)

    def _bed_time_per_day_chart(self, show_text=True):
        df = DataManager.filter_by_category(self.df, 'Bed time')
        df = DataManager.filter_by_date_range(df, self.start_date, self.end_date)
        
        if df.empty:
            st.info("No 'Bed time' events recorded.")
            return

        df = df.copy()
        df['Date'] = df['Start'].dt.date
        df['HourDecimal'] = df['Start'].dt.hour + df['Start'].dt.minute / 60
        df['HourStr'] = df['Start'].dt.strftime('%H:%M')
        
        bed_time = df.groupby('Date')['HourStr'].sum().reset_index()

        # Regresión
        X = np.array([d.toordinal() for d in df['Date']]).reshape(-1, 1)
        y = df['HourDecimal'].values
        avg_val = df['HourDecimal'].mean()
        avg_str = f"{int(avg_val):02}:{int((avg_val % 1) * 60):02}"
        
        try:
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            trend_y = model.predict(X_poly)
            has_trend = True
        except:
            has_trend = False

        mode_val = 'lines+markers+text' if show_text else 'lines+markers'
        text_tpl = '%{customdata}' if show_text else None
        hover_info = 'skip' if show_text else None
        hover_tpl = None if show_text else '<b>%{x}</b><br>Time: %{customdata}<extra></extra>'
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['HourDecimal'],
            mode=mode_val,
            line_shape='spline',
            line=dict(color='firebrick', width=2),
            name='Bed Time',texttemplate=text_tpl,
            textposition="top center",
            hoverinfo=hover_info,
            hovertemplate=hover_tpl,
            customdata=df['HourStr']
        ))

        if has_trend:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=trend_y,
                mode='lines',
                line=dict(color='blue', dash='dot', width=2),               
                name=f'Avg: {avg_str}h',
                hoverinfo='skip'
            ))

        min_h = int(df['HourDecimal'].min()) - 1
        max_h = int(df['HourDecimal'].max()) + 1

        fig.update_layout(
            title='Bed Time Trend',
            yaxis=dict(
                range=[min_h, max_h],
                tickmode='array',
                tickvals=list(range(min_h, max_h+1)),
                ticktext=[f"{h:02}:00" for h in range(min_h, max_h+1)]
            ),
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        col1, col2 = st.columns([4, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.caption("Data:")
            st.dataframe(bed_time, height=400)

    def _naps_stacked_chart(self):
        # 1. Filtrar Datos
        df = DataManager.filter_by_category(self.df, 'Sleep')
        df = DataManager.filter_by_date_range(df, self.start_date, self.end_date)
        
        if 'Notes' not in df.columns: df['Notes'] = ""
        df['Notes'] = df['Notes'].fillna("").astype(str)

        # Quedarnos solo con siestas (No Night Sleep)
        df = df[df['Notes'] != 'Night Sleep'].copy()
        df = df.dropna(subset=['Start', 'End'])
        
        if df.empty:
            st.info("No nap data available for this period.")
            return

        # 2. Cálculos básicos para las barras
        df['DurationHours'] = (df['End'] - df['Start']).dt.total_seconds() / 3600
        df['Date'] = df['Start'].dt.date
        df['StartTimeStr'] = df['Start'].dt.strftime('%H:%M')
        df['EndTimeStr'] = df['End'].dt.strftime('%H:%M')

        # Asignar NapRank
        df = df.sort_values(by=['Start'])
        df['NapRank'] = df.groupby('Date').cumcount() + 1
        df['NapName'] = 'Nap ' + df['NapRank'].astype(str)

        # --- NUEVO: CÁLCULO DE MEDIA MÓVIL (TREND) ---
        # 1. Agrupar por día
        daily_totals = df.groupby('Date')['DurationHours'].sum().reset_index()
        
        # 2. Asegurar continuidad de fechas (Rellenar días sin siestas con 0)
        # Convertimos a datetime para poder reindexar
        daily_totals['Date'] = pd.to_datetime(daily_totals['Date'])
        daily_totals = daily_totals.set_index('Date')
        
        # Creamos rango completo
        full_idx = pd.date_range(start=daily_totals.index.min(), end=daily_totals.index.max(), freq='D')
        daily_totals = daily_totals.reindex(full_idx, fill_value=0)
        
        # 3. Calcular Media Móvil (7 días)
        # min_periods=1 asegura que calcule algo aunque no tenga 7 días previos al principio
        daily_totals['RollingAvg'] = daily_totals['DurationHours'].rolling(window=7, min_periods=1).mean()
        
        # Volvemos a sacar la fecha como columna para el gráfico
        daily_totals = daily_totals.reset_index().rename(columns={'index': 'Date'})
        daily_totals['Date'] = daily_totals['Date'].dt.date # Volver a formato fecha simple
        # -----------------------------------------------

        # 3. Configurar Colores
        color_map = {
            'Nap 1': '#ADD8E6', 'Nap 2': '#87CEEB', 'Nap 3': '#4682B4',
            'Nap 4': '#4169E1', 'Nap 5': '#0000CD', 'Nap 6': '#00008B'
        }

        # 4. Crear el Gráfico Base
        fig = px.bar(
            df,
            x='Date',
            y='DurationHours',
            color='NapName',
            title='Total Nap Duration + 7-Day Trend',
            color_discrete_map=color_map,
            custom_data=['StartTimeStr', 'EndTimeStr', 'DurationHours']
        )

        fig.update_traces(
            hovertemplate=(
                '<b>%{x}</b><br>%{fullData.name}<br>'
                'Time: %{customdata[0]} - %{customdata[1]}<br>'
                'Duration: %{customdata[2]:.2f} h<extra></extra>'
            )
        )

        # 5. --- AÑADIR LA LÍNEA DE TENDENCIA (PROMEDIO) ---
        fig.add_trace(go.Scatter(
            x=daily_totals['Date'],
            y=daily_totals['RollingAvg'],
            mode='lines', # Solo línea, sin puntos para que sea más limpia
            name='7-Day Avg',            
            line_shape='spline',
            line=dict(color='darkorange', width=4), # Naranja fuerte para destacar sobre azul
            hoverinfo='skip', # Opcional: saltar hover si ensucia
            hovertemplate='<b>7-Day Avg: %{y:.2f} h</b><extra></extra>'
        ))

        # 6. Ajuste del Eje Y
        if not daily_totals.empty:
            # El max debe considerar tanto las barras como la línea de tendencia
            max_bar = daily_totals['DurationHours'].max()
            max_line = daily_totals['RollingAvg'].max()
            max_h = int(max(max_bar, max_line)) + 1
        else:
            max_h = 5

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Hours',
            yaxis=dict(range=[0, max_h]),
            legend_title_text='Order',
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

    def _wake_window_evolution_chart(self):
        # 1. Preparar Datos
        cats = ['Sleep', 'Woke up', 'Bed time']
        df = DataManager.filter_by_category(self.df, cats)
        df = DataManager.filter_by_date_range(df, self.start_date, self.end_date)
        
        if 'Notes' not in df.columns: df['Notes'] = ""
        df['Notes'] = df['Notes'].fillna("").astype(str).str.lower() # Normalizar a minúsculas
        
        # Filtramos: Nos quedamos con Woke Up, Bed Time, y SOLO las siestas (No Night Sleep)
        # Identificamos siestas si NO tienen "night sleep" en notas
        df = df[~df['Notes'].str.contains('night sleep')].copy()
        
        if df.empty:
            st.info("No data to calculate wake windows.")
            return

        # Ordenar cronológicamente
        df['Start'] = pd.to_datetime(df['Start'])
        df['End'] = pd.to_datetime(df['End'])
        
        # Prioridad para orden: Woke up(0) -> Sleep(1) -> Bed time(2)
        # Queremos procesar primero el despertar, luego siestas, finalmente dormir
        priority_map = {'Woke up': 0, 'Sleep': 1, 'Bed time': 2}
        df['Order'] = df['Type'].map(priority_map).fillna(1)
        df = df.sort_values(by=['Start', 'Order'])

        wake_windows = []
        
        # Estado
        last_wake_time = None
        window_counter = 1
        
        for _, row in df.iterrows():
            evt_type = row['Type']
            
            if evt_type == 'Woke up':
                # Empieza el día
                last_wake_time = row['Start'] # O End si tienes duración en Woke up, pero suele ser puntual
                window_counter = 1
                
            elif evt_type == 'Sleep':
                # Es una siesta. El tiempo desde last_wake_time hasta ahora es una Wake Window
                if last_wake_time is not None:
                    duration_sec = (row['Start'] - last_wake_time).total_seconds()
                    # Filtramos errores (ventanas negativas o absurdamente largas > 12h)
                    if 0 < duration_sec < 43200: 
                        wake_windows.append({
                            'Date': row['Start'].date(),
                            'WindowName': f"Window {window_counter}", # 1st, 2nd...
                            'DurationHours': duration_sec / 3600,
                            'DurationStr': _format_duration_str(duration_sec)
                        })
                        window_counter += 1
                
                # Actualizamos: ahora estamos despiertos desde que acabó esta siesta
                last_wake_time = row['End']
                
            elif evt_type == 'Bed time':
                # Última ventana del día (desde última siesta hasta dormir)
                if last_wake_time is not None:
                    duration_sec = (row['Start'] - last_wake_time).total_seconds()
                    if 0 < duration_sec < 43200:
                        wake_windows.append({
                            'Date': row['Start'].date(),
                            'WindowName': "Last Window", # Nombre especial para la última
                            'DurationHours': duration_sec / 3600,
                            'DurationStr': _format_duration_str(duration_sec)
                        })
                
                # Reseteamos hasta el siguiente Woke up
                last_wake_time = None

        # Crear DataFrame de resultados
        ww_df = pd.DataFrame(wake_windows)
        
        if ww_df.empty:
            st.info("Not enough sequential data (Wake Up -> Nap) to plot wake windows.")
            return

        # 2. Calcular Tendencia Promedio Diaria
        # Promedio de todas las ventanas de ese día para ver si aguantamos más despiertos en general
        daily_avg = ww_df.groupby('Date')['DurationHours'].mean().reset_index()
        # Suavizado (Rolling average 7 días)
        daily_avg['Date'] = pd.to_datetime(daily_avg['Date'])
        daily_avg = daily_avg.set_index('Date').reindex(
            pd.date_range(ww_df['Date'].min(), ww_df['Date'].max()), fill_value=np.nan
        )
        daily_avg['Rolling'] = daily_avg['DurationHours'].interpolate().rolling(window=7, min_periods=1).mean()
        daily_avg = daily_avg.reset_index().rename(columns={'index': 'Date'})

        # 3. Gráfico Scatter
        fig = go.Figure()

        # Colores para las ventanas (1, 2, 3...)
        # Mapa manual para consistencia
        colors = {
            'Window 1': '#FFB6C1', # LightPink
            'Window 2': '#FF69B4', # HotPink
            'Window 3': '#FF1493', # DeepPink
            'Window 4': '#C71585', # MediumVioletRed
            'Last Window': '#800080' # Purple (Destacar la última antes de dormir)
        }

        # Añadir puntos (Scatter)
        for name in ww_df['WindowName'].unique():
            subset = ww_df[ww_df['WindowName'] == name]
            color = colors.get(name, 'gray') # Fallback color
            
            fig.add_trace(go.Scatter(
                x=subset['Date'],
                y=subset['DurationHours'],
                mode='markers',
                name=name,
                marker=dict(color=color, size=7, opacity=0.7),
                customdata=subset['DurationStr'],
                hovertemplate='<b>%{x}</b><br>%{fullData.name}<br>Awake: %{customdata}<extra></extra>'
            ))

        # Añadir Línea de Tendencia Promedio
        fig.add_trace(go.Scatter(
            x=daily_avg['Date'],
            y=daily_avg['Rolling'],
            mode='lines',
            name='Avg Trend (7d)',
            line=dict(color='orange', width=3, dash='solid'),
            hoverinfo='skip'
        ))

        # 4. Ajustes Finales
        fig.update_layout(
            yaxis_title="Hours Awake",
            xaxis_title="Date",
            height=500,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _night_waking_scatter_chart(self):
        # 1. Filtrar solo Night Waking
        df = DataManager.filter_by_category(self.df, 'Night waking')
        df = DataManager.filter_by_date_range(df, self.start_date, self.end_date)
        
        if df.empty:
            st.info("No night wakings recorded in this period.")
            return

        # 2. Preparar datos
        df = df.copy()
        df = df.dropna(subset=['Start', 'End'])
        df['DurationSec'] = (df['End'] - df['Start']).dt.total_seconds()
        
        # Filtrar errores
        df = df[df['DurationSec'] > 0].copy()
        
        if df.empty:
            st.info("No valid night wakings found.")
            return

        df['DurationMin'] = df['DurationSec'] / 60
        df['Date'] = df['Start'].dt.date

        # 3. Configuración del Slider y Layout
        # Calculamos límites
        min_val = int(df['DurationMin'].min())
        max_val = int(np.ceil(df['DurationMin'].max()))
        
        # Creamos columnas: Gráfico (Ancho) | Controles (Estrecho)
        col_chart, col_controls = st.columns([5, 1], gap="medium")
        
        # -- COLUMNA DERECHA: SLIDER --
        with col_controls:
            st.write("Filters") # Etiqueta para dar contexto
            if max_val > min_val:
                # Nota: Streamlit no permite sliders verticales nativos todavía.
                # Lo ponemos aquí horizontal pero a la derecha del gráfico.
                rango_seleccionado = st.slider(
                    "Min/Max (mins)",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=1,
                    key="night_waking_slider"
                    # orientation="vertical"  <-- Ojalá existiera esto en Streamlit nativo
                )
                
                # Aplicar filtro
                df_filtered = df[
                    (df['DurationMin'] >= rango_seleccionado[0]) & 
                    (df['DurationMin'] <= rango_seleccionado[1])
                ].copy()
            else:
                df_filtered = df.copy()

        # Si el filtro vació los datos
        if df_filtered.empty:
            with col_chart:
                st.warning("No wakings in this range.")
            return

        # 4. Preparar datos filtrados para el gráfico
        def normalize_night_time(dt):
            dummy_date = datetime(2000, 1, 1)
            if dt.hour < 12:
                return dummy_date + timedelta(days=1, hours=dt.hour, minutes=dt.minute)
            else:
                return dummy_date + timedelta(hours=dt.hour, minutes=dt.minute)

        df_filtered['TimeForAxis'] = df_filtered['Start'].apply(normalize_night_time)
        df_filtered['TimeStr'] = df_filtered['Start'].dt.strftime('%H:%M')
        df_filtered['DurationStr'] = df_filtered['DurationSec'].apply(_format_duration_str)

        # 5. Crear Gráfico
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['TimeForAxis'],
            mode='markers',
            marker=dict(
                size=df_filtered['DurationMin'], 
                sizemode='area',
                sizeref=2.*max(df_filtered['DurationMin'].max(), 1)/(30.**2), 
                sizemin=4, 
                color=df_filtered['DurationMin'], 
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Mins")
            ),
            customdata=np.stack((df_filtered['TimeStr'], df_filtered['DurationStr']), axis=-1),
            hovertemplate=(
                '<b>%{x}</b><br>' +
                'Start: %{customdata[0]}<br>' +
                'Duration: %{customdata[1]}<br>' +
                '<extra></extra>'
            )
        ))

        # Ajustes de Ejes
        y_min = datetime(2000, 1, 1, 18, 0)
        y_max = datetime(2000, 1, 2, 9, 0)

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Time of Night",
            yaxis=dict(
                range=[y_min, y_max], 
                tickformat='%H:%M'
            ),
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        # -- COLUMNA IZQUIERDA: GRÁFICO --
        with col_chart:
            st.plotly_chart(fig, use_container_width=True)

    def _night_waking_duration_chart(self):
        # 1. Filtrar Datos
        df = DataManager.filter_by_category(self.df, 'Night waking')
        df = DataManager.filter_by_date_range(df, self.start_date, self.end_date)
        
        if df.empty:
            st.info("No night wakings data available.")
            return

        df = df.copy()
        df = df.dropna(subset=['Start', 'End'])
        
        # Calcular duración
        df['DurationSec'] = (df['End'] - df['Start']).dt.total_seconds()
        df = df[df['DurationSec'] > 0].copy() # Limpiar errores
        df['DurationMin'] = df['DurationSec'] / 60

        # 2. Lógica de "Night Date" (Asignar madrugadas a la noche anterior)
        # Si la hora es < 12:00, pertenece a la noche del día anterior.
        df['NightDate'] = df['Start'].apply(
            lambda x: x.date() - timedelta(days=1) if x.hour < 12 else x.date()
        )

        # 3. Agrupar por Noche
        daily_sum = df.groupby('NightDate')['DurationMin'].sum().reset_index()
        
        # Rellenar huecos (días con 0 despertares) para que el promedio sea real
        daily_sum['NightDate'] = pd.to_datetime(daily_sum['NightDate'])
        daily_sum = daily_sum.set_index('NightDate')
        idx = pd.date_range(daily_sum.index.min(), daily_sum.index.max())
        daily_sum = daily_sum.reindex(idx, fill_value=0).reset_index().rename(columns={'index': 'NightDate'})
        daily_sum['NightDate'] = daily_sum['NightDate'].dt.date

        # Calcular Media Móvil (7 días)
        daily_sum['RollingAvg'] = daily_sum['DurationMin'].rolling(window=7, min_periods=1).mean()

        # 4. Crear Gráfico
        fig = go.Figure()

        # Barras (Total Diario)
        fig.add_trace(go.Bar(
            x=daily_sum['NightDate'],
            y=daily_sum['DurationMin'],
            name='Total Awake Time',
            marker_color='#CD5C5C', # IndianRed (Tono rojizo suave)
            hovertemplate='<b>%{x}</b><br>Awake: %{y:.0f} mins<extra></extra>'
        ))

        # Línea (Tendencia)
        fig.add_trace(go.Scatter(
            x=daily_sum['NightDate'],
            y=daily_sum['RollingAvg'],
            mode='lines',
            name='7-Day Avg',
            line=dict(color='darkred', width=3, dash='dot'),
            hoverinfo='skip',
            hovertemplate='<b>7-Day Avg: %{y:.0f} mins</b><extra></extra>'
        ))

        # 5. Configuración visual
        # Calculamos max Y para dar espacio
        max_y = daily_sum['DurationMin'].max() if not daily_sum.empty else 60
        
        fig.update_layout(
            xaxis_title="Date (Night of...)",
            yaxis_title="Minutes Awake",
            yaxis=dict(range=[0, max_y * 1.1]), # 10% de margen arriba
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    def _cursed_hour_histogram(self):
        # 1. Filtrar Datos
        df = DataManager.filter_by_category(self.df, 'Night waking')
        df = DataManager.filter_by_date_range(df, self.start_date, self.end_date)
        
        if df.empty:
            st.info("No night wakings recorded.")
            return

        # 2. Agrupar por Hora
        # Extraemos la hora (0-23)
        hours = df['Start'].dt.hour
        counts = hours.value_counts().reset_index()
        counts.columns = ['Hour', 'Count']

        # 3. Lógica de Ordenamiento "Nocturno"
        # Queremos que la noche empiece a las 18:00 y termine a las 11:00 (aprox)
        # Asignamos un "SortKey":
        # - Si es AM (0-11): Mantenemos el número (0, 1, 2...)
        # - Si es PM (12-23): Restamos 24 para que sean negativos y vayan al principio (-6, -5...)
        #   Ejemplo: 23:00 -> -1, 00:00 -> 0. Orden correcto: -1, 0.
        
        def get_sort_key(h):
            return h - 24 if h >= 12 else h

        counts['SortKey'] = counts['Hour'].apply(get_sort_key)
        counts = counts.sort_values('SortKey')
        
        # Crear etiquetas bonitas "03:00"
        counts['Label'] = counts['Hour'].apply(lambda h: f"{h:02d}:00")

        # 4. Detectar la "Hora Maldita" (Máximo) para colorearla distinto
        max_count = counts['Count'].max()
        # Creamos una lista de colores: Rojo oscuro para el max, Naranja para el resto
        counts['Color'] = counts['Count'].apply(
            lambda x: '#8B0000' if x == max_count else '#FF7F50' # DarkRed vs Coral
        )

        # 5. Crear Gráfico
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=counts['Label'],
            y=counts['Count'],
            marker_color=counts['Color'],
            text=counts['Count'], # Poner el número encima de la barra
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Wakings: %{y}<extra></extra>'
        ))

        fig.update_layout(
            xaxis_title="Hour of Night",
            yaxis_title="Total Wakings",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)


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
    Versión ROBUSTA: Limpia espacios en blanco y normaliza texto para evitar
    que eventos se pierdan por errores de formato (ej. "Nap " vs "Nap").
    """
    # Definimos categorías base
    cats = ['Sleep', 'Bed time', 'Woke up', 'Night waking']
    
    # 1. Limpieza preliminar de columnas de texto clave
    # Aseguramos que existan y limpiamos espacios alrededor
    if 'Type' in main_df.columns:
        main_df['Type'] = main_df['Type'].fillna("").astype(str).str.strip()
    
    if 'Notes' not in main_df.columns:
        main_df['Notes'] = ""
    main_df['Notes'] = main_df['Notes'].fillna("").astype(str).str.strip()

    # 2. Filtrar
    df = main_df[main_df['Type'].isin(cats)].copy()
    
    if df.empty:
        return pd.DataFrame()

    # 3. Fechas y Ordenamiento
    df['Start'] = pd.to_datetime(df['Start'])
    df['End'] = pd.to_datetime(df['End'])

    # Prioridad: Bed time(0) -> Night waking(1) -> Sleep(2) -> Woke up(3)
    priority_map = {'Bed time': 0, 'Night waking': 1, 'Sleep': 2, 'Woke up': 3}
    df['Order'] = df['Type'].map(priority_map).fillna(2)

    df = df.sort_values(by=['Start', 'Order']).reset_index(drop=True)

    processed_rows = []
    
    is_night_mode = False
    night_sleep_buffer = [] 
    has_waking = False
    last_sleep_end = None
    
    for _, row in df.iterrows():
        evt_type = row['Type'] # Ya está limpio (sin espacios)
        note_val_lower = row['Notes'].lower() # Usamos minúsculas para comparar
        
        # --- MÁQUINA DE ESTADOS ---

        if evt_type == 'Bed time':
            is_night_mode = True
            night_sleep_buffer = []
            has_waking = False
            
        elif evt_type == 'Woke up':
            # Cerramos la noche
            if is_night_mode:
                quality = "🟣 Night Sleep (Solid)" if (not has_waking and len(night_sleep_buffer) == 1) else "🔵 Night Sleep (Interrupted)"
                for s_row, s_wake_str in night_sleep_buffer:
                    _process_visual_row(s_row, quality, processed_rows, s_wake_str)
                
                is_night_mode = False
                night_sleep_buffer = []
            else:
                # Si encontramos un Woke up pero NO estábamos en modo noche,
                # no hacemos nada crítico, solo aseguramos que el flag esté apagado.
                is_night_mode = False

        elif evt_type == 'Night waking':
            if is_night_mode:
                has_waking = True

        elif evt_type == 'Sleep':
            # Calcular Wake Window
            wake_window_str = "-"
            if last_sleep_end is not None:
                diff_seconds = (row['Start'] - last_sleep_end).total_seconds()
                if diff_seconds > 0:
                    wake_window_str = _format_duration_str(diff_seconds)
            
            last_sleep_end = row['End']

            # --- LÓGICA DE CLASIFICACIÓN MEJORADA ---
            
            # 1. Si la nota contiene "nap", ES UNA SIESTA (Prioridad absoluta)
            if 'nap' in note_val_lower:
                _process_visual_row(row, "🌫️ Nap", processed_rows, wake_window_str)
                
            # 2. Si la nota contiene "night sleep", ES SUEÑO NOCTURNO
            elif 'night sleep' in note_val_lower:
                if is_night_mode:
                    night_sleep_buffer.append((row, wake_window_str))
                else:
                    # Caso borde: Night Sleep fuera de contenedor Bedtime->Wokeup
                    _process_visual_row(row, "🔵 Night Sleep (Interrupted)", processed_rows, wake_window_str)
            
            # 3. Si no hay nota (Fallback)
            else:
                if is_night_mode:
                    # Si estamos en horario nocturno, asumimos que es parte de la noche
                    night_sleep_buffer.append((row, wake_window_str))
                else:
                    # Si es de día, asumimos que es siesta
                    _process_visual_row(row, "🌫️ Nap", processed_rows, wake_window_str)

    # Limpieza final del buffer (si el archivo acaba a mitad de noche)
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
        return "#8A2BE2" # 🟣 BlueViolet (Destacado)
    elif status == "🔵 Night Sleep (Interrupted)":
        return "royalblue" # 🔵 Azul normal
    else:
        return "#ADD8E6" # 🌫️ LightBlue (Siestas)

# --- CLASE PRINCIPAL ---

class SleepSection:
    def __init__(self, df, start_date, end_date):
        self.df = df
        self.start_date = start_date
        self.end_date = end_date
        
        self.dummy_data = pd.DataFrame({
            'Date': pd.date_range(start='2024-04-01', periods=7, freq='D'),
            'Total Sleep (hrs)': [12, 13, 11.5, 14, 12.2, 13.3, 12.8],
            'Nap Duration (hrs)': [2, 1.5, 1.8, 2.5, 1.2, 2.1, 1.7],
            'Night Wakeups': [2, 1, 3, 1, 2, 1, 2],
        })

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

        with tab4:
            st.write('🌙 Nighttime Wakeups')
            fig4 = px.line(self.dummy_data, x='Date', y='Night Wakeups', markers=True, title='Night Wakeups Per Day')
            st.plotly_chart(fig4, use_container_width=True)

    def _sleep_timeline_chart(self):
        gantt_data = _prepare_sleep_gantt_data(self.df)
        
        if gantt_data.empty:
            st.info("No sleep data available.")
            return

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=gantt_data['Date'],
            x=gantt_data['Duration'],
            base=gantt_data['StartHour'],
            orientation='h',
            marker=dict(
                color=gantt_data['Color'],
                line=dict(width=0)
            ),
            name='Sleep',
            customdata=np.stack((
                gantt_data['StartTimeStr'],      # 0
                gantt_data['EndTimeStr'],        # 1
                gantt_data['TotalDurationStr'],  # 2
                gantt_data['Status'],            # 3
                gantt_data['WakeWindow']         # 4
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
                range=[0, 24],
                tickmode='array',
                tickvals=[0, 4, 8, 12, 16, 20, 24],
                ticktext=['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
                side='top'
            ),
            yaxis=dict(
                title=None,
                type='category'
            ),
            height=max(400, 25 * len(gantt_data['Date'].unique())),
            margin=dict(l=0, r=10, t=30, b=0),
            showlegend=False,
            hovermode="closest"
        )
        
        st.caption("🟣 Solid Night | 🔵 Interrupted Night | 🌫️ Nap")
        st.plotly_chart(fig, use_container_width=True)

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
            name=f'Avg: {avg_str}',
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
                name='Trend',
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
        # 1. Filtrar Datos (Igual que antes)
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

        # Asignar NapRank (Nap 1, Nap 2...)
        df = df.sort_values(by=['Start'])
        df['NapRank'] = df.groupby('Date').cumcount() + 1
        df['NapName'] = 'Nap ' + df['NapRank'].astype(str)

        # --- NUEVO: CÁLCULO PARA LA LÍNEA DE TOTALES ---
        daily_totals = df.groupby('Date')['DurationHours'].sum().reset_index()
        # -----------------------------------------------

        # 3. Configurar Colores (Azules)
        color_map = {
            'Nap 1': '#ADD8E6', 'Nap 2': '#87CEEB', 'Nap 3': '#4682B4',
            'Nap 4': '#4169E1', 'Nap 5': '#0000CD', 'Nap 6': '#00008B'
        }

        # 4. Crear el Gráfico Base (Barras Apiladas)
        fig = px.bar(
            df,
            x='Date',
            y='DurationHours',
            color='NapName',
            title='Total Nap Duration (Stacked + Daily Total)',
            color_discrete_map=color_map,
            custom_data=['StartTimeStr', 'EndTimeStr', 'DurationHours']
        )

        # Tooltip de las barras
        fig.update_traces(
            hovertemplate=(
                '<b>%{x}</b><br>%{fullData.name}<br>'
                'Time: %{customdata[0]} - %{customdata[1]}<br>'
                'Duration: %{customdata[2]:.2f} h<extra></extra>'
            )
        )

        # 5. --- AÑADIR LA LÍNEA DE TOTALES ---
        fig.add_trace(go.Scatter(
            x=daily_totals['Date'],
            y=daily_totals['DurationHours'],
            mode='lines+markers+text', # Línea + Puntos + Texto
            name='Total Daily',
            line_shape='spline',
            line=dict(color='darkgoldenrod', width=3, dash='dot'), # Color oscuro para contrastar
            text=daily_totals['DurationHours'].apply(lambda x: f"{x:.1f}h"), # Etiqueta con el valor
            textposition='top center',
            hovertemplate='<b>Total: %{y:.2f} h</b><extra></extra>'
        ))

        # 6. Ajuste del Eje Y
        if not daily_totals.empty:
            max_h = int(daily_totals['DurationHours'].max()) + 2 # +2 para dejar espacio al texto de la línea
        else:
            max_h = 5

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Total Hours',
            yaxis=dict(range=[0, max_h]),
            legend_title_text='Order',
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
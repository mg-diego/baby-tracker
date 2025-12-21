import streamlit as st
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data_manager import DataManager

class OverviewSection:
    def __init__(self, df):
        self.df = df

    def render(self):
        st.header("📊 Overview")
        tab1, tab2 = st.tabs(["📅 Daily Space View", "🌍 Global Metrics"])

        with tab1:            
            self._render_daily_view()
        with tab2:
            self._render_global_metrics()

    def _render_daily_view(self):
        # 1. Selector de Fecha
        if not self.df.empty and 'Start' in self.df.columns:
            last_updated = self.df["Start"].max()
        else:
            last_updated = datetime.now()

        col_date, col_mode = st.columns([1, 2])
        with col_date:
            custom_date = st.date_input(
                "Select date",
                value=last_updated,
                max_value=datetime.now(),
                min_value=datetime(2024, 1, 1)
            )
        
        target_date = pd.to_datetime(custom_date).date()

        # 2. Selector de Modo (Día / Noche)
        with col_mode:
            view_mode = st.radio(
                "Select View", 
                ["☀️ Day View", "🌙 Night View"], 
                horizontal=True,
                label_visibility="collapsed"
            )
            is_night_mode = (view_mode == "🌙 Night View")

        # 3. Calcular Límites (Anchors) según el modo
        anchor_start, anchor_end = self._get_anchors(target_date, is_night_mode)

        # 4. Layout
        col1, col2 = st.columns([3, 1], gap="medium")
        
        with col1:
            self._render_space_polar_chart(anchor_start, anchor_end, is_night_mode)
        
        with col2:
            metrics = self._calculate_window_metrics(anchor_start, anchor_end, is_night_mode)
            
            st.subheader(f"{view_mode.split()[0]} {target_date.strftime('%d %b')}")
            
            # Texto dinámico según modo
            label_start = "Bed Time (Prev)" if is_night_mode else "Woke Up"
            label_end = "Woke Up" if is_night_mode else "Bed Time"
            
            st.caption(f"{anchor_start.strftime('%H:%M')} - {anchor_end.strftime('%H:%M')}")
            st.divider()
            
            c_a, c_b = st.columns(2)
            c_a.metric(label_start, metrics['start_time'])
            c_b.metric(label_end, metrics['end_time'])
            
            st.divider()
            
            # Métricas dinámicas
            if is_night_mode:
                st.metric("💤 Night Sleep Duration", f"{metrics['main_duration_h']:.1f} h")
                st.metric("⚡ Wakings", metrics['events_count']) # Wakings
            else:
                st.metric("💤 Naps Total", f"{metrics['main_duration_h']:.1f} h")
                st.metric("💤 Naps Count", metrics['events_count']) # Naps
            
            st.metric("🤱 Feeds", metrics['feeds'])
            st.metric("🩲 Diapers", metrics['diapers'])

    def _get_anchors(self, target_date, is_night_mode):
        """
        Determina la hora de inicio y fin del arco principal.
        - Día: Wake (Hoy) -> Bed (Hoy)
        - Noche: Bed (Ayer) -> Wake (Hoy)
        """
        if is_night_mode:
            # BUSCAR: Bed Time de AYER -> Wake Up de HOY
            prev_date = target_date - timedelta(days=1)
            
            # Bed time de ayer (el último registrado ayer)
            bed_events = self.df[
                (self.df['Type'] == 'Bed time') & 
                (self.df['Start'].dt.date == prev_date)
            ].sort_values('Start')
            
            if not bed_events.empty:
                start_dt = bed_events.iloc[-1]['Start']
            else:
                start_dt = datetime.combine(prev_date, datetime.strptime("20:00", "%H:%M").time())

            # Wake up de hoy (el primero de hoy)
            wake_events = self.df[
                (self.df['Type'] == 'Woke up') & 
                (self.df['Start'].dt.date == target_date)
            ].sort_values('Start')
            
            if not wake_events.empty:
                end_dt = wake_events.iloc[0]['Start']
            else:
                end_dt = datetime.combine(target_date, datetime.strptime("08:00", "%H:%M").time())
                
        else:
            # MODO DÍA: Wake (Hoy) -> Bed (Hoy)
            day_df = self.df[self.df['Start'].dt.date == target_date]
            
            wake_events = day_df[day_df['Type'] == 'Woke up'].sort_values('Start')
            bed_events = day_df[day_df['Type'] == 'Bed time'].sort_values('Start')

            if not wake_events.empty:
                start_dt = wake_events.iloc[0]['Start']
            else:
                start_dt = datetime.combine(target_date, datetime.strptime("08:00", "%H:%M").time())

            if not bed_events.empty:
                end_dt = bed_events.iloc[-1]['Start']
            else:
                # Si es hoy y aún no duerme, usar ahora. Si es pasado, 20:00.
                if target_date == datetime.now().date():
                    end_dt = datetime.now()
                else:
                    end_dt = datetime.combine(target_date, datetime.strptime("20:00", "%H:%M").time())
        
        # Corrección por si las fechas se cruzan erróneamente
        if end_dt <= start_dt:
            end_dt = start_dt + timedelta(hours=12)

        return start_dt, end_dt

    def _render_space_polar_chart(self, start_dt, end_dt, is_night_mode):
        """
        Gráfico Polar Normalizado Dinámico.
        Recibe start_dt y end_dt ya calculados.
        """
        # Filtrar datos que ocurran DENTRO de la ventana seleccionada
        # (Importante para modo noche que cruza días)
        window_df = self.df[
            (self.df['Start'] >= start_dt) & 
            (self.df['Start'] <= end_dt)
        ].copy()

        # Duración del arco principal
        total_duration_seconds = (end_dt - start_dt).total_seconds()
        if total_duration_seconds <= 0: total_duration_seconds = 43200 

        # --- B. PROYECCIÓN VISUAL ---
        VISUAL_START_DEG = 230 
        VISUAL_END_DEG = 130   
        TOTAL_SWEEP = (360 - VISUAL_START_DEG) + VISUAL_END_DEG

        def get_theta(dt):
            seconds_passed = (dt - start_dt).total_seconds()
            pct = seconds_passed / total_duration_seconds
            angle = VISUAL_START_DEG + (pct * TOTAL_SWEEP)
            return angle % 360

        def generate_arc_normalized(s_dt, e_dt, num_points=100):
            s_sec = (s_dt - start_dt).total_seconds()
            e_sec = (e_dt - start_dt).total_seconds()
            secs = np.linspace(s_sec, e_sec, num_points)
            pcts = secs / total_duration_seconds
            angles = (VISUAL_START_DEG + (pcts * TOTAL_SWEEP)) % 360
            r = [1] * num_points
            return r, angles

        # --- C. PREPARAR EVENTOS SECUNDARIOS ---
        if 'Notes' not in window_df.columns: window_df['Notes'] = ""
        
        # En MODO NOCHE, los arcos de color son Wakings. En MODO DÍA son Naps.
        if is_night_mode:
            # Buscamos eventos 'Night waking'
            segments_df = window_df[window_df['Type'] == 'Night waking'].copy()
            # Si Night waking son puntos (no tienen End), podemos simular duración o usar puntos.
            # Asumiré que tienen duración o usaremos un ancho fijo si End es NaT.
        else:
            # Buscamos Naps
            segments_df = window_df[
                (window_df['Type'] == 'Sleep') & 
                (window_df['Notes'].str.contains('Nap', case=False, na=False))
            ].copy()

        diapers_df = window_df[window_df['Type'] == 'Diaper'].copy()
        feeds_df = window_df[window_df['Type'] == 'Feed'].copy()

        # --- D. DIBUJAR ---
        fig = go.Figure()

        # 1. ARCO PRINCIPAL (FONDO)
        theta_bg = np.linspace(VISUAL_START_DEG, VISUAL_START_DEG + TOTAL_SWEEP, 200) % 360
        r_bg = [1] * 200
        
        # Color del arco: Azul oscuro para Día, Morado muy oscuro para Noche (opcional)
        bg_color = 'rgba(40, 44, 84, 1)' 
        
        fig.add_trace(go.Scatterpolar(
            r=r_bg, theta=theta_bg, mode='lines',
            line=dict(color=bg_color, width=40),
            hoverinfo='skip', showlegend=False
        ))

        # 2. SEGMENTOS DE COLOR (Naps o Wakings)
        segment_color = '#FF4500' if is_night_mode else '#7B61FF' # Rojo para Wakings, Violeta para Naps
        
        for _, row in segments_df.iterrows():
            # Si es un evento puntual (sin fin), le damos 15 min visuales
            s_t = row['Start']
            e_t = row['End'] if pd.notnull(row['End']) else s_t + timedelta(minutes=15)
            
            r_seg, theta_seg = generate_arc_normalized(s_t, e_t, 50)
            
            fig.add_trace(go.Scatterpolar(
                r=r_seg, theta=theta_seg, mode='lines',
                line=dict(color=segment_color, width=32),
                hoverinfo='text',
                hovertext=f"{row['Type']}: {s_t.strftime('%H:%M')}"
            ))

        # 3. SINGLE EVENTS (Diapers / Feeds)
        
        # Diapers
        for _, row in diapers_df.iterrows():
            angle = get_theta(row['Start'])
            is_poo = 'poo' in str(row.get('End Condition', '')).lower() or 'poo' in str(row.get('Notes', '')).lower()
            fig.add_trace(go.Scatterpolar(
                r=[1], theta=[angle], mode='markers',
                marker=dict(
                    color='#8B4513' if is_poo else '#fec0ff', 
                    size=20, symbol='circle', line=dict(color='white', width=1)
                ),
                hoverinfo='text',
                hovertext=f"Diaper: {row['Start'].strftime('%H:%M')}", showlegend=False
            ))

        # Feeds
        for _, row in feeds_df.iterrows():
            angle = get_theta(row['Start'])
            is_bottle = 'bottle' in str(row.get('Start Location', '')).lower()
            fig.add_trace(go.Scatterpolar(
                r=[1], theta=[angle], mode='markers',
                marker=dict(
                    color='#58cf39' if is_bottle else '#FF69B4', 
                    size=20, symbol='circle', line=dict(color='white', width=1)
                ),
                hoverinfo='text',
                hovertext=f"Feed: {row['Start'].strftime('%H:%M')}", showlegend=False
            ))

        # 4. MARCADORES DE INICIO / FIN (ANCHORS)
        # Inicio (Wake si es día, Bed si es noche)
        color_start = '#FF8C00' if is_night_mode else '#FFD700' # Naranja (Bed) o Amarillo (Wake)
        
        fig.add_trace(go.Scatterpolar(
            r=[1], theta=[VISUAL_START_DEG], mode='markers+text',
            marker=dict(color=color_start, size=20, line=dict(color='#1E1E1E', width=4)), 
            text=[f"{start_dt.strftime('%H:%M')}"], textposition="bottom center",
            textfont=dict(color=color_start, size=14, weight='bold'),
            hoverinfo='skip', showlegend=False
        ))

        # Fin (Bed si es día, Wake si es noche)
        color_end = '#FFD700' if is_night_mode else '#FF8C00'
        
        fig.add_trace(go.Scatterpolar(
            r=[1], theta=[VISUAL_END_DEG], mode='markers+text',
            marker=dict(color=color_end, size=20, line=dict(color='#1E1E1E', width=4)), 
            text=[f"{end_dt.strftime('%H:%M')}"], textposition="bottom center",
            textfont=dict(color=color_end, size=14, weight='bold'),
            hoverinfo='skip', showlegend=False
        ))

        # 5. TICKS HORARIOS
        # Generar horas entre inicio y fin
        # Redondeamos al la siguiente hora completa
        curr_h = start_dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        while curr_h < end_dt:
            # Solo mostramos cada 3 horas para no saturar (o todas si prefieres)
            if curr_h.hour % 3 == 0: 
                angle = get_theta(curr_h)
                fig.add_trace(go.Scatterpolar(
                    r=[1.15], theta=[angle], mode='text',
                    text=[f"<b>{curr_h.strftime('%H')}</b>"],
                    textfont=dict(color='rgba(255, 255, 255, 0.5)', size=12),
                    hoverinfo='skip', showlegend=False
                ))
            curr_h += timedelta(hours=1)

        # --- LAYOUT Y TEXTO CENTRAL ---
        
        # Calcular duración de "Segmentos" (Naps o Wakings)
        segments_duration_sec = 0
        if not segments_df.empty:
            # Si tienen End
            valid = segments_df.dropna(subset=['End'])
            segments_duration_sec = (valid['End'] - valid['Start']).dt.total_seconds().sum()
        
        segments_h = segments_duration_sec / 3600
        
        # En modo noche, quizás interesa ver cuánto durmió en total (Total ventana - Wakings)
        # O ver cuánto tiempo estuvo despierto.
        if is_night_mode:
            # Mostrar horas de sueño (Ventana - Despertares)
            sleep_time = (total_duration_seconds - segments_duration_sec) / 3600
            center_label = "Night Sleep"
            center_val = f"{sleep_time:.1f} h"
        else:
            center_label = "Nap Sleep"
            center_val = f"{segments_h:.1f} h"
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=False, range=[0, 1.25]), 
                angularaxis=dict(visible=False, direction="clockwise", rotation=90),
            ),
            margin=dict(t=0, b=0, l=0, r=0),
            height=600,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    def _calculate_window_metrics(self, start_dt, end_dt, is_night_mode):
        """Calcula métricas dentro de la ventana de tiempo específica."""
        # Filtrar DF en el rango
        window_df = self.df[
            (self.df['Start'] >= start_dt) & 
            (self.df['Start'] <= end_dt)
        ].copy()
        
        feeds = len(window_df[window_df['Type'] == 'Feed'])
        diapers = len(window_df[window_df['Type'] == 'Diaper'])
        
        main_duration_h = 0
        events_count = 0

        if is_night_mode:
            # En modo noche: Eventos principales son Wakings
            wakings = window_df[window_df['Type'] == 'Night waking']
            events_count = len(wakings)
            
            # Duración del sueño = Ventana total - tiempo despierto
            total_window_sec = (end_dt - start_dt).total_seconds()
            
            # Calcular tiempo despierto si los wakings tienen duración
            awake_sec = 0
            if not wakings.empty and 'End' in wakings.columns:
                 valid_w = wakings.dropna(subset=['End'])
                 awake_sec = (valid_w['End'] - valid_w['Start']).dt.total_seconds().sum()
            
            main_duration_h = (total_window_sec - awake_sec) / 3600

        else:
            # En modo día: Eventos principales son Naps
            if 'Notes' not in window_df.columns: window_df['Notes'] = ""
            naps = window_df[
                (window_df['Type'] == 'Sleep') & 
                (window_df['Notes'].str.contains('Nap', case=False, na=False))
            ]
            events_count = len(naps)
            
            if not naps.empty:
                valid = naps.dropna(subset=['End'])
                main_duration_h = (valid['End'] - valid['Start']).dt.total_seconds().sum() / 3600

        return {
            'start_time': start_dt.strftime('%H:%M'),
            'end_time': end_dt.strftime('%H:%M'),
            'feeds': feeds,
            'diapers': diapers,
            'main_duration_h': main_duration_h,
            'events_count': events_count
        }

    def _render_global_metrics(self):
        def fmt_num(val):
            return f"{val:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")

        def fmt_int(val):
            return f"{val:,.0f}".replace(",", ".")
        
        sleep_df = DataManager.filter_by_category(self.df, 'Sleep')
        waking_df = DataManager.filter_by_category(self.df, 'Night waking')
        
        total_sleep_hrs = 0
        night_sleep_hrs = 0
        nap_sleep_hrs = 0
        total_naps_count = 0
        
        if not sleep_df.empty:
            is_nap = sleep_df.astype(str).apply(lambda x: x.str.contains('Nap', case=False)).any(axis=1)
            total_naps_count = is_nap.sum()            
            valid_sleep = sleep_df.dropna(subset=['Start', 'End']).copy()
            if not valid_sleep.empty:
                is_nap_valid = valid_sleep.astype(str).apply(lambda x: x.str.contains('Nap', case=False)).any(axis=1)
                valid_sleep['Duration'] = (valid_sleep['End'] - valid_sleep['Start']).dt.total_seconds() / 3600
                
                nap_sleep_hrs = valid_sleep[is_nap_valid]['Duration'].sum()
                night_sleep_hrs = valid_sleep[~is_nap_valid]['Duration'].sum()
                total_sleep_hrs = nap_sleep_hrs + night_sleep_hrs

        total_wakings_count = len(waking_df)

        feed_df = DataManager.filter_by_category(self.df, 'Feed')
        is_bottle = feed_df.astype(str).apply(lambda x: x.str.contains('Bottle|Formula', case=False)).any(axis=1)        
        breast_df = feed_df[~is_bottle].copy()
        total_breast_hrs = 0
        if not breast_df.empty:
            valid_breast = breast_df.dropna(subset=['Start', 'End'])
            total_breast_hrs = (valid_breast['End'] - valid_breast['Start']).dt.total_seconds().sum() / 3600

        bottle_df = feed_df[is_bottle].copy()
        total_bottles = len(bottle_df)
        total_formula_vol = 0
        
        if not bottle_df.empty:
            def extract_ml(row):
                for val in row.values:
                    val_str = str(val).lower()
                    if 'ml' in val_str:
                        match = re.search(r'(\d+)\s*ml', val_str)
                        if match: return float(match.group(1))
                return 0
            total_formula_vol = bottle_df.apply(extract_ml, axis=1).sum()

        diaper_df = DataManager.filter_by_category(self.df, 'Diaper')
        total_diapers = len(diaper_df)
        total_pee = 0
        total_poo = 0
        
        if not diaper_df.empty:
            text_data = diaper_df.astype(str).agg(' '.join, axis=1).str.lower()
            total_pee = text_data.str.contains('pee|mojado', na=False).sum()
            total_poo = text_data.str.contains('poo|sucio', na=False).sum()
        
        st.caption(f"📊 Global stats based on {len(self.df)} records.")
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.markdown("### 💤 Sleep & Rest")
                st.metric(
                    "Total Sleep Time", 
                    f"{fmt_num(total_sleep_hrs)} h", 
                    help=f"🌙 Night: {fmt_num(night_sleep_hrs)}h\n☀️ Nap: {fmt_num(nap_sleep_hrs)}h",
                )
                st.metric("Total Naps", fmt_int(total_naps_count))
                st.metric("Night Wakings", fmt_int(total_wakings_count))

        with c2:
            with st.container(border=True):
                st.markdown("### 🍼 Feeding")
                st.metric("Breastfeeding Time", f"{fmt_num(total_breast_hrs)} h")
                st.metric("Bottles Given", fmt_int(total_bottles))
                
                if total_formula_vol > 1000:
                    vol_str = f"{fmt_num(total_formula_vol/1000)} L"
                else:
                    vol_str = f"{fmt_int(total_formula_vol)} ml"
                    
                st.metric("Formula Consumed", vol_str)

        with c3:
            with st.container(border=True):
                st.markdown("### 🩲 Diapers")
                st.metric("Total Diapers", fmt_int(total_diapers))
                st.metric("Pee (Wet)", fmt_int(total_pee))
                st.metric("Poo (Dirty)", fmt_int(total_poo))
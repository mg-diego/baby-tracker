import re
import time
import os
from appium import webdriver
from appium.options.common.base import AppiumOptions
from selenium.webdriver.common.by import By

from datetime import datetime, timedelta

def replace_spanish_months(date_str):
    # Dictionary mapping Spanish month names to English
    months_translation = {
        "Enero": "January",
        "Febrero": "February",
        "Marzo": "March",
        "Abril": "April",
        "Mayo": "May",
        "Junio": "June",
        "Julio": "July",
        "Agosto": "August",
        "Septiembre": "September",
        "Octubre": "October",
        "Noviembre": "November",
        "Diciembre": "December"
    }

    # Replace Spanish months with English months
    for spanish_month, english_month in months_translation.items():
        date_str = date_str.replace(spanish_month, english_month)

    return date_str

def convert_date_format(date_str):
    # Replace Spanish months with English months
    date_str = replace_spanish_months(date_str)
    parsed_date = datetime.strptime(date_str, "%B %d").replace(year=2025)
    formatted_date = parsed_date.strftime("%Y-%m-%d")
    return formatted_date


def convert_date_format_previous_year(date_str):
    # 1. Diccionario de abreviaturas en español
    spanish_months = {
        "ene": "Jan", "feb": "Feb", "mar": "Mar", "abr": "Apr",
        "may": "May", "jun": "Jun", "jul": "Jul", "ago": "Aug",
        "sep": "Sep", "oct": "Oct", "nov": "Nov", "dic": "Dec"
    }

    # 2. Limpieza de la cadena:
    # Quitamos el punto, el símbolo 'º' y pasamos a minúsculas para comparar
    clean_date = date_str.replace(".", "").replace("º", "").lower()
    
    # Separamos el mes y el día
    parts = clean_date.split()
    if len(parts) != 2:
        return None # O maneja el error según necesites

    month_es, day = parts[0], parts[1]

    # 3. Traducir mes
    month_en = spanish_months.get(month_es, month_es)

    # 4. Reconstruir y parsear
    # Usamos %b para abreviaturas (Jan, Dec, etc.)
    final_str = f"{month_en} {day}"
    parsed_date = datetime.strptime(final_str, "%b %d").replace(year=2025)
    
    return parsed_date.strftime("%Y-%m-%d")

def map_values(event):    
    translations = {
        "fórmula": BOTTLE_RESOURCE_ID,
        "otro": MILK_RESOURCE_ID,
        "lactancia": NURSING_RESOURCE_ID,
        "cambio de pañal": DIAPER_RESOURCE_ID,
        "se durmió": BED_TIME_RESOURCE_ID,
        "despertar por la noche": NIGHT_WAKING_RESOURCE_ID,
        "siesta": NAP_RESOURCE_ID,
        "se despertó": WOKE_UP_RESOURCE_ID,
        "mojado": "Pee",
        "sucio": "Poo",
        "sueño": NIGHT_SLEEP,
        "medicina": MEDICINE_RESOURCE_ID,
        "sólidos": SOLID_RESOURCE_ID,
    }

    # Replace Spanish months with English months
    for spanish, english in translations.items():
        event = event.lower().replace(spanish, english)

    return event.capitalize()

# Scroll down
def scroll_down():
    size = driver.get_window_size()
    start_x = size['width'] / 2
    start_y = size['height'] * 0.6  # Start from 80% of the height
    end_y = size['height'] * 0.2    # End at 20% of the height
    driver.swipe(start_x, start_y, start_x, end_y, 800)  # Duration in milliseconds
    
# Scroll down
def scroll_right_to_left():
    size = driver.get_window_size()
    start_x = size['width'] - 1
    start_y = size['height'] / 2 
    driver.swipe(start_x, start_y, 1, start_y, 800)  # Duration in milliseconds

def scroll_to_top():
    last_source = ""
    while True:
        # 1. Obtener el tamaño de la pantalla
        size = driver.get_window_size()
        width = size['width']
        height = size['height']

        # 2. Definir puntos: De arriba (20%) hacia abajo (80%) para subir el contenido
        start_x = width / 2
        start_y = height * 0.2  # Punto superior
        end_y = height * 0.8    # Punto inferior

        # 3. Realizar el swipe
        driver.swipe(start_x, start_y, start_x, end_y, 800)
        
        # Pequeña pausa para que la UI se estabilice
        time.sleep(1)

        # 4. Verificar si la pantalla cambió
        current_source = driver.page_source
        if current_source == last_source:
            print("Llegaste al principio de la página.")
            break
        last_source = current_source

def parse_time_string(time_str):
    """
    Parses a time string in various formats (e.g., '1h:50m', '15m:09s') 
    and returns the total time as a timedelta object.

    Parameters:
    time_str (str): The time string to parse.

    Returns:
    timedelta: The total time as a timedelta object.
    """
    # Initialize hours, minutes, and seconds
    hours = 0
    minutes = 0
    seconds = 0

    # Regular expression to match hours, minutes, and seconds
    hour_match = re.search(r'(\d+)h', time_str)
    minute_match = re.search(r'(\d+)m', time_str)
    second_match = re.search(r'(\d+)s', time_str)

    if hour_match:
        hours = int(hour_match.group(1))
    if minute_match:
        minutes = int(minute_match.group(1))
    if second_match:
        seconds = int(second_match.group(1))

    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def subtract_time(time_str1, time_str2):
    """
    Subtracts time_str2 from time_str1 and returns the result as a string in YYYY-MM-DD HH:MM format.

    Parameters:
    time_str1 (str): The initial time in YYYY-MM-DD HH:MM format.
    time_str2 (str): The time to subtract in various formats (e.g., '1h:50m', '15m:09s').

    Returns:
    str: The resulting time after subtraction in YYYY-MM-DD HH:MM format.
    """
    # Convert the first time string to a datetime object
    time1 = datetime.strptime(time_str1, "%Y-%m-%d %H:%M")

    # Parse the second time string to a timedelta
    time2_delta = parse_time_string(time_str2)

    # Subtract the second time from the first time
    result_time = time1 - time2_delta

    # Format the result back to a string
    return result_time.strftime("%Y-%m-%d %H:%M")


options = AppiumOptions()
options.load_capabilities({
	"platformName": "Android",
    "appPackage": "com.napper",
    "appium:skipDeviceInitialization": True,
	"appium:deviceName": "5a0dda8e1022",
	"appium:automationName": "uiautomator2",
	"appium:ensureWebviewsHavePages": True,
	"appium:nativeWebScreenshot": True,
	"appium:newCommandTimeout": 1000,
	"appium:connectHardwareKeyboard": True,
    'noReset': True,  # Do not reset app state
    'fullReset': False,  # Do not uninstall the app
})

#//android.widget.ScrollView
BASE_RESOURCE_ID = "NewBabyLogEntry"

NAP_RESOURCE_ID = "NAP"
NURSING_RESOURCE_ID = "NURSING"
BOTTLE_RESOURCE_ID = "BOTTLE"
MILK_RESOURCE_ID = "MILK"
SOLID_RESOURCE_ID = "SOLID"
WOKE_UP_RESOURCE_ID = "WOKE_UP"
NIGHT_WAKING_RESOURCE_ID = "NIGHT_WAKING"
DIAPER_RESOURCE_ID = "CHANGED_DIAPER"
BED_TIME_RESOURCE_ID = "BED_TIME"
MEDICINE_RESOURCE_ID = "MEDICINE"

NIGHT_SLEEP = "NIGHT_SLEEP"

EVENTS = [NAP_RESOURCE_ID, NURSING_RESOURCE_ID, BOTTLE_RESOURCE_ID, WOKE_UP_RESOURCE_ID, NIGHT_WAKING_RESOURCE_ID, DIAPER_RESOURCE_ID, BED_TIME_RESOURCE_ID]

driver = webdriver.Remote("http://192.168.68.55:4723", options=options)

while True:
    current_day = driver.find_element(By.XPATH, "//android.view.ViewGroup[@resource-id='TimeMachineHeader-MiddleDateButton']")
    current_day = convert_date_format(current_day.tag_name.split(",")[0])

    unique_view_groups = []
    events = driver.find_elements(By.XPATH, "//android.view.ViewGroup[starts-with(@resource-id, 'NewBabyLogEntry.')] | //android.widget.TextView[contains(@text, 'Fin de la sesión de sueño')]")
    view_groups_texts = [event.tag_name if event.tag_name is not None else event.text for event in events]

    while((len(unique_view_groups) == 0) or (unique_view_groups[-1] != view_groups_texts[-1])):
        unique_view_groups += view_groups_texts
        scroll_down()
        events = driver.find_elements(By.XPATH, "//android.view.ViewGroup[starts-with(@resource-id, 'NewBabyLogEntry.')] | //android.widget.TextView[contains(@text, 'Fin de la sesión de sueño')]")
        view_groups_texts = [event.tag_name if event.tag_name is not None else event.text for event in events]

    seen = set()
    temp = [x for x in unique_view_groups if not (x in seen or seen.add(x))]

    events_csv = []

    for register in temp:
        groups = register.split(",")
        event_type = map_values(groups[0])
        timestamp = groups[1].split("-") if len(groups) > 1 else ''
        start = timestamp[0].strip() if len(timestamp) > 0 else ''
        stop = groups[1].split("-")[1].strip() if len(timestamp) > 1 else ''
        duration_or_amount = groups[2].replace(" ", "") if len(groups) > 2 else ''

        event_type = event_type.upper()
        start_condition = ''
        start_location = ''
        end_condition = ''
        notes = ''

        if(MEDICINE_RESOURCE_ID in event_type):
            custom_type = "Medicine"

        elif(NAP_RESOURCE_ID in event_type):
            custom_type = "Sleep"
            notes = "Nap"

        elif(NIGHT_SLEEP in event_type):
            custom_type = "Sleep"
            notes = "Night Sleep"
            duration_or_amount = register.split('Fin de la sesión de sueño de')[1].replace(" ", "")
            stop = events_csv[-1]["start"]
            start = subtract_time(current_day + " " + stop, duration_or_amount)

        elif(NURSING_RESOURCE_ID in event_type):
            custom_type = "Feed"
            start_location = "Breast"
            if (len(groups) > 3):
                start_condition = "R" if "D" in groups[3] else ""
                end_condition = "L" if "I" in groups[3] else ""
            else:
                start_condition, end_condition = "", ""

        elif(BOTTLE_RESOURCE_ID in event_type):
            custom_type = "Feed"        
            start_condition = "Formula"
            start_location = "Bottle"

        elif(MILK_RESOURCE_ID in event_type):
            custom_type = "Feed"        
            start_condition = "Milk"
            start_location = "Bottle"
        
        elif(SOLID_RESOURCE_ID in event_type):
            custom_type = "Feed"
            start_condition = "Solid"

        elif(WOKE_UP_RESOURCE_ID in event_type):
            custom_type = "Woke up"
        
        elif(DIAPER_RESOURCE_ID in event_type):
            custom_type = "Diaper"
            end_condition = map_values(duration_or_amount)

        elif(BED_TIME_RESOURCE_ID in event_type):
            custom_type = "Bed time"

        elif(NIGHT_WAKING_RESOURCE_ID in event_type):
            custom_type = "Night waking"

        else:
            custom_type = "Unknown"

        events_csv.append(
            {
                "type": custom_type,
                "day": current_day,
                "start": start,
                "stop": stop,
                "duration_or_amount": duration_or_amount,
                "notes": notes,
                "start_condition": start_condition,
                "start_location": start_location,
                "end_condition": end_condition
            }
        )

    log_block = ""
    for event in events_csv:
        start = event["start"] if "-" in event["start"] else event["day"] + " " + event["start"]
        stop = event["day"] + " " + event["stop"] if event["stop"] != "" else ""
        log_block += f"\"{event['type']}\",\"{start}\",\"{stop}\",\"{event['duration_or_amount']}\",\"{event['start_condition']}\",\"{event['start_location']}\",\"{event['end_condition']}\",\"{event['notes']}\"\n"

    # Leer contenido actual
    if os.path.exists("log.txt"):
        with open("log.txt", "r", encoding="utf-8") as f:
            old_content = f.read()
    else:
        old_content = ""

    # Escribir el nuevo bloque primero, seguido del contenido anterior
    with open("log.txt", "w", encoding="utf-8") as f:
        f.write(log_block + "\n" + old_content)

    scroll_right_to_left()
    time.sleep(3)
    
import re
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

def map_values(event):    
    translations = {
        "fórmula": BOTTLE_RESOURCE_ID,
        "lactancia": NURSING_RESOURCE_ID,
        "cambio de pañal": DIAPER_RESOURCE_ID,
        "se durmió": BED_TIME_RESOURCE_ID,
        "despertar por la noche": NIGHT_WAKING_RESOURCE_ID,
        "siesta": NAP_RESOURCE_ID,
        "se despertó": WOKE_UP_RESOURCE_ID,
        "mojado": "Pee",
        "sucio": "Poo",
        "sueño": NIGHT_SLEEP
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
	"appium:deviceName": "5a0dda8e1022",
	"appium:automationName": "uiautomator2",
	"appium:ensureWebviewsHavePages": True,
	"appium:nativeWebScreenshot": True,
	"appium:newCommandTimeout": 3600,
	"appium:connectHardwareKeyboard": True,
    'noReset': True,  # Do not reset app state
    'fullReset': False,  # Do not uninstall the app
})

#//android.widget.ScrollView
BASE_RESOURCE_ID = "NewBabyLogEntry"

NAP_RESOURCE_ID = "NAP"
NURSING_RESOURCE_ID = "NURSING"
BOTTLE_RESOURCE_ID = "BOTTLE"
WOKE_UP_RESOURCE_ID = "WOKE_UP"
NIGHT_WAKING_RESOURCE_ID = "NIGHT_WAKING"
DIAPER_RESOURCE_ID = "CHANGED_DIAPER"
BED_TIME_RESOURCE_ID = "BED_TIME"

NIGHT_SLEEP = "NIGHT_SLEEP"

EVENTS = [NAP_RESOURCE_ID, NURSING_RESOURCE_ID, BOTTLE_RESOURCE_ID, WOKE_UP_RESOURCE_ID, NIGHT_WAKING_RESOURCE_ID, DIAPER_RESOURCE_ID, BED_TIME_RESOURCE_ID]

driver = webdriver.Remote("http://192.168.56.1:4723", options=options)

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
    #groups = register.tag_name.split(",")
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

    if(NAP_RESOURCE_ID in event_type):
        type = "Sleep"
        notes = "Nap"

    if(NIGHT_SLEEP in event_type):
        type = "Sleep"
        notes = "Night Sleep"
        duration_or_amount = register.split('Fin de la sesión de sueño de')[1].replace(" ", "")
        stop = events_csv[-1]["start"]
        start = subtract_time(current_day + " " + stop, duration_or_amount)

    if(NURSING_RESOURCE_ID in event_type):
        type = "Feed"
        start_location = "Breast"
        if (len(groups) > 3):
            start_condition = "R" if "D" in groups[3] else ""
            end_condition = "L" if "I" in groups[3] else ""
        else:
            start_condition, end_condition = "", ""

    if(BOTTLE_RESOURCE_ID in event_type):
        type = "Feed"        
        start_condition = "Formula"
        start_location = "Bottle"

    if(WOKE_UP_RESOURCE_ID in event_type):
        type = "Woke up"
    
    if(DIAPER_RESOURCE_ID in event_type):
        type = "Diaper"
        end_condition = map_values(duration_or_amount)

    if(BED_TIME_RESOURCE_ID in event_type):
        type = "Bed time"

    if(NIGHT_WAKING_RESOURCE_ID in event_type):
        type = "Night waking"

    events_csv.append(
        {
            "type": type,
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


for event in events_csv:
    start = event["start"] if "-" in event["start"] else event["day"] + " " + event["start"]
    stop = event["day"] + " " + event["stop"] if event["stop"] != "" else ""
    print(f"\"{event["type"]}\",\"{start}\",\"{stop}\",\"{event["duration_or_amount"]}\",\"{event["start_condition"]}\",\"{event["start_location"]}\",\"{event["end_condition"]}\",\"{event["notes"]}\"")

pass
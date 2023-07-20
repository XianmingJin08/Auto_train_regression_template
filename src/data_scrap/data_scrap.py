from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import re
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_daily_data(driver, date):
    url = "https://www.worldweatheronline.com/shaoxing-weather-history/zhejiang/cn.aspx"
    driver.get(url)
    # Fill in the date and submit the form
    date_input = driver.find_element(By.ID,'ctl00_MainContentHolder_txtPastDate')
    date_input.clear()
    driver.execute_script("arguments[0].value = arguments[1];", date_input, date.strftime("%Y-%m-%d"))
    submit_button = driver.find_element(By.ID,'ctl00_MainContentHolder_butShowPastWeather')
    driver.execute_script("arguments[0].scrollIntoView();", submit_button)
    wait = WebDriverWait(driver, 3)
    wait.until(EC.element_to_be_clickable((By.ID,'ctl00_MainContentHolder_butShowPastWeather'))).click()
    # submit_button.click()

    # Parse the page with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = soup.find('table', {'class': 'days-details-table'})

    # Extract the rows
    rows = []
    pattern = r'(\d{2}:\d{2})(\d+ °c)'
    for row in table.find_all('tr'):
        rows.append([val.text for val in row.find_all('td')])
    temp_data = []
    headers = []
    for row in rows:
        for item in row:
            matches = re.findall(pattern, item)
            for match in matches:
                headers.append(match[0])
                temp_data.append(match[1])

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(columns=headers)
    df.loc[len(df)] = temp_data
    df['Date'] = date  # Add the date column
    return df

def get_weather_data(start_date, end_date):
    all_data = pd.DataFrame()

    # Set up the web driver
    options = Options()
    options.add_argument('--ignore-ssl-errors=yes')
    options.add_argument('--ignore-certificate-errors')
    driver = webdriver.Chrome()  # or webdriver.Chrome(), etc.

    for date in daterange(start_date, end_date):
        try:
            daily_data = get_daily_data(driver, date)
            all_data = pd.concat([all_data, daily_data])
        except:
            print ("skip for date: ", date)
            continue
        print (all_data)
    driver.quit()

    # Save the data to a CSV file
    all_data.to_csv('weather_data.csv', index=False)

# Use the function
start_date = datetime(2022, 3, 1)
end_date = datetime(2023, 6, 30)

get_weather_data(start_date, end_date)

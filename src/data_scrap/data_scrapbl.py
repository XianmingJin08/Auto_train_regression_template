import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
from datetime import datetime
import re 

def get_daily_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table with the data
    table = soup.find('table', {'id': 'wt-his'})
    print ("table", table)
    # Extract the headers
    headers = [header.text for header in table.find_all('th')]  
    print ("header", headers)
    headers = [string for string in headers if re.match(r"^[0-9]", string)]
    print ("hh", headers)
    print (len(headers))
    # Extract the rows
    rows = []
    for row in table.find_all('tr'):
        for val in row.find_all('td'):
            if re.search(r"°[FC]", val.text):
                rows.append(val.text)
    print ("rows", rows)
    print (len(rows))
    # Combine the headers and rows into a dataframe
    df = pd.DataFrame(rows, columns=headers)
    return df


def get_daily_links(month, year):
    base_url = "https://www.timeanddate.com/weather/china/shaoxing/historic?month={}&year={}"
    url = base_url.format(month, year)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the links for each day
    link_elements = soup.find_all('div', {'class': 'weatherLinks'})
    print ("all link elments", link_elements)
    links = ["https://www.timeanddate.com/weather/china/shaoxing/historic?hd={}{}{:02d}".format(year, month, i+1) for i in range(len(link_elements))]

    # Find the dates for each day
    dates = [link.text + " " + str(year) for link in link_elements]

    return links, dates



def get_weather_data(start_month, start_year, end_month, end_year):
    all_data = pd.DataFrame()

    for year in range(start_year, end_year + 1):
        for month in range(start_month if year == start_year else 1, end_month + 1 if year == end_year else 13):
            daily_links, dates = get_daily_links(month, year)
            for link, date in zip(daily_links, dates):
                print ("link is ", link)
                daily_data = get_daily_data(link)
                daily_data['Date'] = date  # Add the date column
                all_data = pd.concat([all_data, daily_data])

    # Save the data to a CSV file
    all_data.to_csv('weather_data.csv', index=False)

# Use the function
get_weather_data(6, 2022, 9, 2022)
import requests
from bs4 import BeautifulSoup
import json

url = 'https://dpbosss.net.in/kalyan-morning-panel-chart.php'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'lxml')

tables = soup.find_all('table')
print(f'Found {len(tables)} tables')

if tables:
    for i, row in enumerate(tables[0].find_all('tr')[:5]):
        cols = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
        print(f'Row {i}: {cols}')

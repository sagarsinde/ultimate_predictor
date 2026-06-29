import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timedelta

def parse_main_bazar_chart(url, output_file):
    print(f"Fetching Data from {url}...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch page. Status code: {response.status_code}")
        return
        
    soup = BeautifulSoup(response.text, 'html.parser')
    
    table = soup.find('table', class_='pchart')
    if not table:
        print("Could not find the 'pchart' table.")
        return
        
    rows = table.find_all('tr')
    
    days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'] # 5-day week for Main Bazar
    
    extracted_data = []
    
    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
        # 1 date col + (5 days * 3 cols per day) = 16 cols
        if len(cols) < 16:
            continue
            
        date_str = cols[0]
        # Format is usually '15/07/2019to19/07/2019'
        try:
            start_date_str = date_str.split('to')[0].strip()
            
            # The year might be '19' or '2019'
            if len(start_date_str.split('/')[-1]) == 2:
                start_date = datetime.strptime(start_date_str, '%d/%m/%y')
            else:
                start_date = datetime.strptime(start_date_str, '%d/%m/%Y')
                
        except Exception as e:
            # Skip header rows or malformed dates
            continue
            
        col_idx = 1
        for i, day in enumerate(days_of_week):
            if col_idx + 2 >= len(cols):
                break
                
            m_panel = cols[col_idx]
            jodi = cols[col_idx+1]
            e_panel = cols[col_idx+2]
            
            col_idx += 3
            
            # Skip if there's no data (like *** or blank)
            if jodi == '' or '*' in jodi or len(jodi) != 2:
                continue
                
            morning_number = jodi[0]
            evening_number = jodi[1]
            
            # Calculate the exact date for this day
            current_date = start_date + timedelta(days=i)
            
            extracted_data.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Day': day,
                'Morning_number': morning_number,
                'Evening_number': evening_number
            })
            
    # Sort by date just in case
    extracted_data.sort(key=lambda x: x['Date'])
    
    # Save to CSV
    print(f"Extraction complete. Found {len(extracted_data)} valid draws.")
    print(f"Writing to {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Date', 'Day', 'Morning_number', 'Evening_number'])
        writer.writeheader()
        writer.writerows(extracted_data)
        
    print("Done!")

if __name__ == '__main__':
    url = "https://sattamatkadpboss.mobi/main-bazar-panel-chart.php"
    output = "main_bazar_dataset.csv"
    parse_main_bazar_chart(url, output)

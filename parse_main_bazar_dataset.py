import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timedelta
import re

def parse_main_bazar_chart_v2(url, output_file):
    """
    V2 Parser: Fixed to correctly capture ALL 5 weekdays (Mon-Fri).
    The original parser was missing Thursday data due to malformed HTML
    in the source (mixed <td>/<th> closing tags).
    """
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
    
    days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    
    extracted_data = []
    
    for row in rows:
        # Get ALL child elements (both td and th) to handle malformed HTML
        cols = row.find_all(['td', 'th'])
        
        if len(cols) < 2:
            continue
            
        # First column is the date range
        date_text = cols[0].get_text(strip=True)
        
        # Parse date range like "15/07/2019to19/07/2019" or "16/09/19to21/09/19"
        if 'to' not in date_text:
            continue
            
        try:
            start_date_str = date_text.split('to')[0].strip()
            
            # Handle both 2-digit and 4-digit years
            if len(start_date_str.split('/')[-1]) == 2:
                start_date = datetime.strptime(start_date_str, '%d/%m/%y')
            else:
                start_date = datetime.strptime(start_date_str, '%d/%m/%Y')
                
        except Exception:
            continue
        
        # Strategy: Instead of counting columns by index, we search for
        # 2-character Jodi values. Each day has 3 elements: panel1, jodi, panel2.
        # The Jodi is always the cell with exactly 2 digits (or ** for missing).
        
        # Extract text from each column
        col_texts = []
        for col in cols[1:]:  # Skip the date column
            text = col.get_text(strip=True)
            # Clean up: remove any HTML artifacts
            text = text.replace('\n', '').replace('\r', '').strip()
            col_texts.append(text)
        
        # Now find all Jodi values (2-char strings that are either digits or **)
        # Each day should produce exactly one Jodi
        jodi_indices = []
        for idx, text in enumerate(col_texts):
            # A Jodi is exactly 2 characters: either 2 digits or **
            if len(text) == 2 and (text.isdigit() or text == '**'):
                jodi_indices.append(idx)
        
        # We expect exactly 5 Jodis (one per weekday)
        # But some weeks may have fewer due to holidays
        
        for day_num, jodi_idx in enumerate(jodi_indices):
            if day_num >= 5:
                break  # Safety: max 5 days
                
            jodi = col_texts[jodi_idx]
            
            # Skip missing data
            if jodi == '**' or '*' in jodi or not jodi.isdigit():
                continue
                
            morning_number = jodi[0]
            evening_number = jodi[1]
            
            # Calculate the exact date: start_date is Monday, add day_num days
            current_date = start_date + timedelta(days=day_num)
            day_name = days_of_week[day_num] if day_num < 5 else 'Unknown'
            
            extracted_data.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Day': day_name,
                'Morning_number': morning_number,
                'Evening_number': evening_number
            })
    
    # Sort by date and remove duplicates
    extracted_data.sort(key=lambda x: x['Date'])
    
    # Deduplicate by date
    seen_dates = set()
    unique_data = []
    for row in extracted_data:
        if row['Date'] not in seen_dates:
            seen_dates.add(row['Date'])
            unique_data.append(row)
    
    # Save to CSV
    print(f"Extraction complete. Found {len(unique_data)} valid draws.")
    
    # Count by day
    day_counts = {}
    for row in unique_data:
        day_counts[row['Day']] = day_counts.get(row['Day'], 0) + 1
    print("Day Distribution:")
    for day in days_of_week:
        print(f"  {day}: {day_counts.get(day, 0)}")
    
    print(f"\nWriting to {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Date', 'Day', 'Morning_number', 'Evening_number'])
        writer.writeheader()
        writer.writerows(unique_data)
        
    print("Done!")

if __name__ == '__main__':
    url = "https://sattamatkadpboss.mobi/main-bazar-panel-chart.php"
    output = "main_bazar_dataset.csv"
    parse_main_bazar_chart_v2(url, output)

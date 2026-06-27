import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def process_cards(cards_str):
    # E.g., '340' -> sorted '0', '3', '4'
    if not cards_str or '*' in cards_str:
        return None
    try:
        cards = sorted(list(cards_str))
        return cards
    except:
        return None

def main():
    print("Fetching raw HTML from dpbosss.net.in...")
    url = 'https://dpbosss.net.in/kalyan-morning-panel-chart.php'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    
    tables = soup.find_all('table')
    if not tables:
        print("Error: Could not find any tables on the page.")
        return
        
    table = tables[0]
    rows = table.find_all('tr')[1:] # Skip header
    
    records = []
    draw_index = 1
    
    days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
        if len(cols) < 22:
            continue
            
        date_str = cols[0]
        # Format is usually '30/10/2017to05/11/2017'
        try:
            start_date_str = date_str.split('to')[0].strip()
            start_date = datetime.strptime(start_date_str, '%d/%m/%Y')
        except:
            continue # Skip malformed rows
            
        # 7 days, 3 columns per day
        idx = 1
        for day_offset in range(7):
            m_cards = cols[idx]
            jodi = cols[idx+1]
            e_cards = cols[idx+2]
            idx += 3
            
            # Check for holiday
            if '*' in m_cards or '*' in jodi or '*' in e_cards or not jodi:
                continue
                
            # Process
            m_sorted = process_cards(m_cards)
            e_sorted = process_cards(e_cards)
            
            if not m_sorted or not e_sorted or len(jodi) != 2:
                continue
                
            if len(m_sorted) != 3 or len(e_sorted) != 3:
                continue
                
            m_num = int(jodi[0])
            e_num = int(jodi[1])
            
            current_date = start_date + timedelta(days=day_offset)
            
            records.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Day_of_Week': days_of_week[day_offset],
                'Morning_card1': m_sorted[0],
                'Morning_card2': m_sorted[1],
                'Morning_card3': m_sorted[2],
                'Evening_number1': e_sorted[0],
                'Evening_number2': e_sorted[1],
                'Evening_number3': e_sorted[2],
                'Morning_number': m_num,
                'Evening_number': e_num,
                'Draw_Index': draw_index
            })
            draw_index += 1
            
    df = pd.DataFrame(records)
    
    print(f"\nSuccessfully parsed {len(df)} total valid draws (ignoring holidays).")
    
    output_file = 'true_kalyan_morning_dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}!")
    
    print("\n--- FIRST 5 ROWS ---")
    print(df.head(5))
    
    print("\n--- LAST 5 ROWS ---")
    print(df.tail(5))

if __name__ == '__main__':
    main()

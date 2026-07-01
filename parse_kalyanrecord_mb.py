import re
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def parse_kalyanrecord(file_path):
    print(f"Parsing from local file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    rows = soup.find_all('tr')
    print(f"Found {len(rows)} table rows.")
    
    data = []
    
    # Matches the date range "02/01/17 To 06/01/17"
    date_regex = re.compile(r'(\d{2}/\d{2}/\d{2})')
    
    for row in rows:
        cells = row.find_all('td')
        if not cells or len(cells) < 16:
            continue
            
        # The first cell should be the date range
        date_text = cells[0].get_text(strip=True)
        dates = date_regex.findall(date_text)
        
        if dates:
            start_date_str = dates[0]
            try:
                # Format is DD/MM/YY
                start_date = datetime.strptime(start_date_str, "%d/%m/%y")
            except ValueError:
                continue
                
            # Now we look at cells for Mon-Fri
            # Mon = 2, Tue = 5, Wed = 8, Thu = 11, Fri = 14 (0-indexed index of Jodi TD)
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
            
            for i in range(5):
                jodi_index = 1 + (i * 3) + 1
                if jodi_index < len(cells):
                    jodi_text = cells[jodi_index].get_text(strip=True)
                    
                    # A valid Jodi is exactly 2 digits
                    if len(jodi_text) == 2 and jodi_text.isdigit():
                        current_date = start_date + timedelta(days=i)
                        
                        data.append({
                            'Date': current_date.strftime('%Y-%m-%d'),
                            'Day': day_names[i],
                            'Morning_number': int(jodi_text[0]),
                            'Evening_number': int(jodi_text[1])
                        })

    print(f"Extraction complete. Found {len(data)} valid draws.")
    
    if len(data) == 0:
        return
        
    df = pd.DataFrame(data)
    
    # Verify days
    print("Day Distribution:")
    print(df['Day'].value_counts())
    
    # Sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)
    
    # Save back to main_bazar_dataset.csv
    output_file = 'main_bazar_dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"\nWriting to {output_file}...")
    print("Done!")

if __name__ == '__main__':
    parse_kalyanrecord('kalyanrecord_mb.html')

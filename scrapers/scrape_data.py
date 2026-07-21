import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import re
import os

def scrape_draw_data(source="C:/Users/admin/Downloads/Kalyan Morning Panel Chart _ Daily Panel History.html", output_file="draw_data.csv"):
    print(f"Fetching data from {source} ...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    
    try:
        if os.path.exists(source):
            with open(source, 'r', encoding='utf-8') as f:
                html_content = f.read()
        else:
            response = requests.get(source, headers=headers)
            response.raise_for_status()
            html_content = response.text
    except Exception as e:
        print(f"Failed to fetch website or read file: {e}")
        return
        
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Try to find the table rows. The structure is usually rows containing many tds.
    rows = soup.find_all('tr')
    
    all_data = []
    
    for row in rows:
        tds = row.find_all('td')
        
        # A valid row should have 1 date column + 7 days * 3 columns = 22 columns
        # Some variants might have fewer days if the week is incomplete, but usually they just leave tds empty.
        if len(tds) >= 22:
            date_col = tds[0].get_text(separator=' ')
            # Example: "22/06/2026 to 28/06/2026"
            match = re.search(r'(\d{2}/\d{2}/\d{4})', date_col)
            
            if match:
                start_date_str = match.group(1)
                try:
                    start_date = datetime.strptime(start_date_str, '%d/%m/%Y')
                except ValueError:
                    continue # skip if format doesn't match
                
                # Loop through 7 days of the week
                for day_idx in range(7):
                    # Each day consists of 3 columns
                    col_offset = 1 + (day_idx * 3)
                    
                    if col_offset + 2 < len(tds):
                        w1_td = tds[col_offset]
                        main_td = tds[col_offset + 1]
                        w2_td = tds[col_offset + 2]
                        
                        # Extract the strings separated by <br> or other elements
                        w1_cards = list(w1_td.stripped_strings)
                        w2_cards = list(w2_td.stripped_strings)
                        main_combo = main_td.get_text(strip=True)
                        
                        # Only add if the cell actually contains 3 numbers (valid draw)
                        if len(w1_cards) == 3 and len(w2_cards) == 3:
                            draw_date = start_date + timedelta(days=day_idx)
                            day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day_idx]
                            
                            all_data.append({
                                'Date': draw_date.strftime('%Y-%m-%d'),
                                'Day_of_Week': day_name,
                                'Morning_Card1': w1_cards[0],
                                'Morning_Card2': w1_cards[1],
                                'Morning_Card3': w1_cards[2],
                                'Jodi_Number': main_combo,
                                'Evening_Card1': w2_cards[0],
                                'Evening_Card2': w2_cards[1],
                                'Evening_Card3': w2_cards[2]
                            })
                            
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        print(f"Successfully extracted {len(df)} draw records and saved to '{output_file}'!")
        print(df.head())
    else:
        print("Could not extract any data. The website structure may have changed.")

if __name__ == "__main__":
    scrape_draw_data()

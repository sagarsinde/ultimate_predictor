import requests
from bs4 import BeautifulSoup
import json
import os

def scrape_drikpanchang(start_year=2010, end_year=2026):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    all_data = {}
    
    for year in range(start_year, end_year + 1):
        print(f"Scraping Year {year}...")
        all_data[year] = {"Amavasya": [], "Purnima": []}
        
        # Scrape Amavasya
        url_a = f'https://www.drikpanchang.com/vrats/amavasyadates.html?year={year}'
        resp_a = requests.get(url_a, headers=headers)
        if resp_a.status_code == 200:
            soup = BeautifulSoup(resp_a.text, 'html.parser')
            # Look for dpEventCard which contains the date and event titles
            cards = soup.find_all(class_='dpEventCard')
            for card in cards:
                date_el = card.find(class_='dpEventDateTitle')
                if not date_el: continue
                
                links = card.find_all('a')
                event_names = [a.text.strip() for a in links]
                
                is_main = False
                for name in event_names:
                    if 'Amavasya' in name and name != 'Darsha Amavasya':
                        is_main = True
                        break
                        
                if is_main:
                    all_data[year]["Amavasya"].append(date_el.text.strip())
                
        # Scrape Purnima
        url_p = f'https://www.drikpanchang.com/vrats/purnimasidates.html?year={year}'
        resp_p = requests.get(url_p, headers=headers)
        if resp_p.status_code == 200:
            soup = BeautifulSoup(resp_p.text, 'html.parser')
            cards = soup.find_all(class_='dpEventCard')
            for card in cards:
                date_el = card.find(class_='dpEventDateTitle')
                if not date_el: continue
                
                links = card.find_all('a')
                event_names = [a.text.strip() for a in links]
                
                is_main = False
                for name in event_names:
                    if 'Purnima' in name and 'Vrat' not in name:
                        is_main = True
                        break
                        
                if is_main:
                    all_data[year]["Purnima"].append(date_el.text.strip())
                
    # Save the data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(base_dir, 'drikpanchang_dates.json')
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4)
        
    print(f"Successfully saved all dates to {out_file}")
    
if __name__ == '__main__':
    scrape_drikpanchang()

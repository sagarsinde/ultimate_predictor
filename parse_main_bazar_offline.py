"""
Offline V2 Parser: Parses the already-saved HTML content to extract
Main Bazar data including Thursday draws.
"""
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timedelta
import re

def parse_from_file(html_file, output_file):
    print(f"Parsing from local file: {html_file}")
    
    with open(html_file, 'r', encoding='utf-8') as f:
        raw_content = f.read()
    
    # The content.md file has markdown headers at the top - find the HTML start
    html_start = raw_content.find('<!DOCTYPE html')
    if html_start == -1:
        html_start = raw_content.find('<html')
    if html_start == -1:
        # Try to find the table directly
        html_start = raw_content.find('<table')
    
    html_content = raw_content[html_start:] if html_start >= 0 else raw_content
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    table = soup.find('table', class_='pchart')
    if not table:
        print("Could not find the 'pchart' table. Trying any table...")
        table = soup.find('table')
    if not table:
        print("ERROR: No table found in the HTML file.")
        return
        
    rows = table.find_all('tr')
    print(f"Found {len(rows)} table rows.")
    
    days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    
    extracted_data = []
    
    for row in rows:
        cols = row.find_all(['td', 'th'])
        
        if len(cols) < 2:
            continue
            
        date_text = cols[0].get_text(strip=True)
        
        if 'to' not in date_text:
            continue
            
        try:
            start_date_str = date_text.split('to')[0].strip()
            
            if len(start_date_str.split('/')[-1]) == 2:
                start_date = datetime.strptime(start_date_str, '%d/%m/%y')
            else:
                start_date = datetime.strptime(start_date_str, '%d/%m/%Y')
                
        except Exception:
            continue
        
        # Extract text from each column after the date
        col_texts = []
        for col in cols[1:]:
            text = col.get_text(strip=True)
            text = text.replace('\n', '').replace('\r', '').strip()
            col_texts.append(text)
        
        # Find all Jodi values: 2-char strings that are either 2 digits or **
        jodi_indices = []
        for idx, text in enumerate(col_texts):
            if len(text) == 2 and (text.isdigit() or text == '**'):
                jodi_indices.append(idx)
        
        for day_num, jodi_idx in enumerate(jodi_indices):
            if day_num >= 5:
                break
                
            jodi = col_texts[jodi_idx]
            
            if jodi == '**' or '*' in jodi or not jodi.isdigit():
                continue
                
            morning_number = jodi[0]
            evening_number = jodi[1]
            
            current_date = start_date + timedelta(days=day_num)
            day_name = days_of_week[day_num] if day_num < 5 else 'Unknown'
            
            extracted_data.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Day': day_name,
                'Morning_number': morning_number,
                'Evening_number': evening_number
            })
    
    # Sort by date and deduplicate
    extracted_data.sort(key=lambda x: x['Date'])
    
    seen_dates = set()
    unique_data = []
    for r in extracted_data:
        if r['Date'] not in seen_dates:
            seen_dates.add(r['Date'])
            unique_data.append(r)
    
    print(f"\nExtraction complete. Found {len(unique_data)} valid draws.")
    
    day_counts = {}
    for r in unique_data:
        day_counts[r['Day']] = day_counts.get(r['Day'], 0) + 1
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
    import sys
    import os
    
    # Use the already-scraped HTML file
    html_file = r"C:\Users\admin\.gemini\antigravity-ide\brain\e1ca7c17-6610-4891-b927-4f6145ddf7de\.system_generated\steps\1025\content.md"
    
    if not os.path.exists(html_file):
        print(f"ERROR: HTML file not found at {html_file}")
        print("Please provide the path to the saved HTML file as argument.")
        sys.exit(1)
    
    parse_from_file(html_file, "main_bazar_dataset.csv")

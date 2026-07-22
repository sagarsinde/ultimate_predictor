import os
import re
import csv
from datetime import datetime, timedelta

def clean_panna(text):
    text = text.replace('\n', '').replace('\r', '').replace(' ', '')
    if '*' in text:
        return '***'
    if len(text) >= 3 and text[:3].isdigit():
        return text[:3]
    return '000'

def clean_jodi(text):
    text = text.replace('\n', '').replace('\r', '').replace(' ', '')
    if '*' in text:
        return '**'
    if len(text) >= 2 and text[:2].isdigit():
        return text[:2]
    return ''

def parse_rajdhani_chart(html_path, out_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # The HTML is so broken (unclosed th/tr tags) that BeautifulSoup nests the entire table
    # inside single cells. To bypass this, we use regex to extract the text of every cell linearly.
    raw_cells = re.split(r'<(?:th|td)[^>]*>', html)
    
    clean_cells = []
    for rc in raw_cells[1:]:
        content = re.split(r'</(?:th|td)>|<tr', rc)[0]
        text = re.sub(r'<[^>]+>', '', content).strip()
        text = text.replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '')
        if text:
            clean_cells.append(text)
    
    out_rows = []
    i = 0
    while i < len(clean_cells):
        text = clean_cells[i]
        if 'to' in text:
            dates = re.findall(r'\d{2}/\d{2}/\d{2,4}', text)
            if len(dates) >= 1:
                try:
                    start_date = datetime.strptime(dates[0], '%d/%m/%Y')
                except ValueError:
                    try:
                        start_date = datetime.strptime(dates[0], '%d/%m/%y')
                    except ValueError:
                        i += 1
                        continue
                
                i += 1
                # 6 days loop because Rajdhani Day is a 6-day market
                for day_idx in range(6):
                    if i + 2 < len(clean_cells):
                        m_th = clean_cells[i]
                        j_td = clean_cells[i+1]
                        e_th = clean_cells[i+2]
                        i += 3
                        
                        m_panna = clean_panna(m_th)
                        jodi = clean_jodi(j_td)
                        e_panna = clean_panna(e_th)
                        
                        if jodi:
                            if jodi == '**':
                                m_num = '*'
                                e_num = '*'
                                m_panna = '***'
                                e_panna = '***'
                            else:
                                m_num = jodi[0]
                                e_num = jodi[1]
                            
                            curr_date = start_date + timedelta(days=day_idx)
                            out_rows.append({
                                'Date': curr_date.strftime('%Y-%m-%d'),
                                'Morning_Panna': m_panna,
                                'Morning_number': m_num,
                                'Evening_number': e_num,
                                'Evening_Panna': e_panna
                            })
                continue
        i += 1

    unique_rows = {}
    for r in out_rows:
        unique_rows[r['Date']] = r
    
    sorted_rows = [unique_rows[k] for k in sorted(unique_rows.keys())]

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Date', 'Morning_Panna', 'Morning_number', 'Evening_number', 'Evening_Panna'])
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"Scraped {len(sorted_rows)} valid records to {out_path}")

if __name__ == '__main__':
    html_file = r'C:\Users\admin\.gemini\antigravity-ide\brain\e1ca7c17-6610-4891-b927-4f6145ddf7de\.system_generated\steps\3357\content.md'
    out_file = r'f:\ultimate_preducter\data\rajdhani_day_dataset.csv'
    
    if os.path.exists(html_file):
        parse_rajdhani_chart(html_file, out_file)
    else:
        print(f"Error: Could not find downloaded HTML file at {html_file}")

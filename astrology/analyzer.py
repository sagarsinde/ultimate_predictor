import pandas as pd
import ephem
from datetime import datetime, timedelta
import os

def get_moon_phases(start_year, end_year):
    """
    Calculates the exact dates (in IST) of every New Moon (Amavasya) 
    and Full Moon (Purnima) within a year range using ephem.
    """
    amavasya_dates = set()
    purnima_dates = set()
    
    start_date = ephem.Date(f"{start_year}/01/01")
    end_date = ephem.Date(f"{end_year}/12/31")
    
    # Calculate all New Moons (Amavasya)
    curr = start_date
    while curr < end_date:
        nm = ephem.next_new_moon(curr)
        # ephem returns UTC. Add 5.5 hours for IST
        nm_ist = nm.datetime() + timedelta(hours=5, minutes=30)
        amavasya_dates.add(nm_ist.strftime('%Y-%m-%d'))
        curr = nm + 1  # advance a day to find the next one
        
    # Calculate all Full Moons (Purnima)
    curr = start_date
    while curr < end_date:
        fm = ephem.next_full_moon(curr)
        fm_ist = fm.datetime() + timedelta(hours=5, minutes=30)
        purnima_dates.add(fm_ist.strftime('%Y-%m-%d'))
        curr = fm + 1
        
    return list(amavasya_dates), list(purnima_dates)

def analyze_suppression(df, dates_list, phase_name, market_name):
    # Filter out holidays
    df = df[df['Morning_number'] != '*']
    df = df[df['Evening_number'] != '*']
    
    target_dates = pd.to_datetime(dates_list)
    is_target = df['Date'].isin(target_dates)
    
    target_df = df[is_target]
    normal_df = df[~is_target]
    
    if len(target_df) == 0:
        return f"\nNo {phase_name} draws found in {market_name} dataset."
        
    m_target = target_df['Morning_number'].value_counts(normalize=True) * 100
    e_target = target_df['Evening_number'].value_counts(normalize=True) * 100
    
    m_normal = normal_df['Morning_number'].value_counts(normalize=True) * 100
    e_normal = normal_df['Evening_number'].value_counts(normalize=True) * 100
    
    out = []
    out.append(f"\n=======================================================")
    out.append(f" {market_name} Analysis on {phase_name} ({len(target_df)} days analyzed)")
    out.append(f"=======================================================")
    
    out.append("\n[MORNING DRAW SUPPRESSION]")
    for num in sorted(m_target.keys()):
        a_freq = m_target.get(num, 0)
        n_freq = m_normal.get(num, 0)
        diff = a_freq - n_freq
        out.append(f"Digit {num}: Hit {a_freq:>4.1f}% (Avg {n_freq:>4.1f}%) | Diff: {diff:>+5.1f}%")
        
    # Check for totally suppressed numbers
    missing_m = set(m_normal.keys()) - set(m_target.keys())
    if missing_m:
        out.append(f"*** WARNING: Digits {sorted(list(missing_m))} NEVER HIT on {phase_name}! (100% Suppressed) ***")
        
    out.append("\n[EVENING DRAW SUPPRESSION]")
    for num in sorted(e_target.keys()):
        a_freq = e_target.get(num, 0)
        n_freq = e_normal.get(num, 0)
        diff = a_freq - n_freq
        out.append(f"Digit {num}: Hit {a_freq:>4.1f}% (Avg {n_freq:>4.1f}%) | Diff: {diff:>+5.1f}%")

    missing_e = set(e_normal.keys()) - set(e_target.keys())
    if missing_e:
        out.append(f"*** WARNING: Digits {sorted(list(missing_e))} NEVER HIT on {phase_name}! (100% Suppressed) ***")
        
    return "\n".join(out)

if __name__ == '__main__':
    # Build a full decade of exact astrological dates
    amavasya_dates, purnima_dates = get_moon_phases(2017, 2026)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    madhur_path = os.path.join(base_dir, 'data', 'madhur_dataset.csv')
    rajdhani_path = os.path.join(base_dir, 'data', 'rajdhani_day_dataset.csv')
    
    report_out = []
    report_out.append("# Astrological Suppression Analysis (2017 - 2026)")
    
    if os.path.exists(madhur_path):
        m_df = pd.read_csv(madhur_path)
        m_df['Date'] = pd.to_datetime(m_df['Date'])
        report_out.append(analyze_suppression(m_df, amavasya_dates, "AMAVASYA (New Moon)", "Madhur"))
        report_out.append(analyze_suppression(m_df, purnima_dates, "PURNIMA (Full Moon)", "Madhur"))
        
    if os.path.exists(rajdhani_path):
        r_df = pd.read_csv(rajdhani_path)
        r_df['Date'] = pd.to_datetime(r_df['Date'])
        report_out.append(analyze_suppression(r_df, amavasya_dates, "AMAVASYA (New Moon)", "Rajdhani Day"))
        report_out.append(analyze_suppression(r_df, purnima_dates, "PURNIMA (Full Moon)", "Rajdhani Day"))
        
    report_str = "\n".join(report_out)
    
    # Save the report
    report_path = os.path.join(base_dir, 'astrology', 'astrology_report.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_str)
        
    print(f"Analysis complete! Report saved to {report_path}")

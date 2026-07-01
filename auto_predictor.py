import pandas as pd
import subprocess
import re
import sys
from datetime import datetime, timedelta

def get_next_playing_day(current_date, market):
    # Main bazar plays Mon-Fri (0-4)
    # Kalyan plays Mon-Sat (0-5)
    next_date = current_date + timedelta(days=1)
    if market == 'mb':
        while next_date.weekday() > 4: # Skip Sat/Sun
            next_date += timedelta(days=1)
    elif market == 'kalyan':
        while next_date.weekday() > 5: # Skip Sun
            next_date += timedelta(days=1)
    return next_date

def get_top_prediction(script_name):
    # Run the prediction script and capture output
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    output = result.stdout
    
    # We need to parse the AI's top prediction from the output
    # Looking for lines like:
    # Digit 8: 14.78%  ██████████████
    # The first one under Morning and the first one under Evening.
    
    morning_digit = None
    evening_digit = None
    
    # Simple regex to find digits and confidences
    digit_pattern = re.compile(r'Digit (\d): \d+\.\d+%')
    
    is_morning_section = False
    is_evening_section = False
    
    for line in output.split('\n'):
        if 'OPEN (Morning)' in line or 'Morning Prediction' in line or 'Morning Draw' in line:
            is_morning_section = True
            is_evening_section = False
        elif 'CLOSE (Evening)' in line or 'Evening Prediction' in line or 'Evening Draw' in line:
            is_morning_section = False
            is_evening_section = True
            
        match = digit_pattern.search(line)
        if match:
            if is_morning_section and morning_digit is None:
                morning_digit = int(match.group(1))
            elif is_evening_section and evening_digit is None:
                evening_digit = int(match.group(1))
                
    # If the script recommends SKIP, we still need a digit to append to the dataset.
    # The first digit printed is the highest probability one.
    
    if morning_digit is None or evening_digit is None:
        print(output)
        raise ValueError("Could not parse predictions from output!")
        
    return morning_digit, evening_digit

def append_to_dataset(csv_file, date, morning, evening, market):
    df = pd.read_csv(csv_file)
    day_name = date.strftime('%a')
    
    # Build new row
    if market == 'kalyan':
        # Kalyan dataset has panel numbers. We don't have predicted panel numbers, 
        # so we will just fill them with 0s.
        new_row = {
            'Date': date.strftime('%Y-%m-%d'),
            'Day_of_Week': day_name,
            'Morning_card1': 0, 'Morning_card2': 0, 'Morning_card3': 0,
            'Evening_number1': 0, 'Evening_number2': 0, 'Evening_number3': 0,
            'Morning_number': morning,
            'Evening_number': evening,
            'Draw_Index': df['Draw_Index'].max() + 1 if 'Draw_Index' in df.columns else len(df) + 1
        }
    else:
        # Main Bazar columns: Date,Day,Morning_number,Evening_number
        new_row = {
            'Date': date.strftime('%Y-%m-%d'),
            'Day': day_name,
            'Morning_number': morning,
            'Evening_number': evening
        }
        
    new_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(csv_file, index=False)
    print(f"Appended {date.strftime('%Y-%m-%d')} ({day_name}) -> Morning: {morning}, Evening: {evening} to {csv_file}")

def run_auto_predictor(market, target_date_str):
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    
    if market == 'kalyan':
        csv_file = 'true_kalyan_main_dataset.csv'
        predict_script = 'predict_tomorrow.py'
        feature_script = 'build_features.py'
    else:
        csv_file = 'main_bazar_dataset.csv'
        predict_script = 'predict_tomorrow_mb.py'
        feature_script = 'build_features_mb.py'
        
    while True:
        df = pd.read_csv(csv_file)
        last_date_str = df['Date'].iloc[-1]
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
        
        next_date = get_next_playing_day(last_date, market)
        
        if next_date > target_date:
            print(f"\n========================================================")
            print(f" Dataset is up to date through {target_date_str}")
            print(f" Now predicting NEXT playing day: {next_date.strftime('%Y-%m-%d')} ({next_date.strftime('%A')})")
            print(f"========================================================\n")
            # Run predict one final time to show tomorrow's prediction
            result = subprocess.run(['python', predict_script], capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            break
            
        print(f"\n========================================================")
        print(f" Predicting for {next_date.strftime('%Y-%m-%d')}...")
        print(f"========================================================")
        
        # 1. Predict the next day
        morning, evening = get_top_prediction(predict_script)
        
        # 2. If the next day is exactly our target date, we just want to print it and STOP!
        if next_date == target_date:
            print(f"🎯 TARGET DATE REACHED: {next_date.strftime('%Y-%m-%d')} 🎯")
            print(f"Final AI Prediction -> Morning: {morning}, Evening: {evening}")
            
            # Print the actual full output for the user to see the confidence scores
            result = subprocess.run(['python', predict_script], capture_output=True, text=True)
            print("\n" + result.stdout)
            break
            
        # 3. Otherwise, we append it to the dataset to use it as context for the next day!
        print(f"Auto-feeding AI prediction ({morning}, {evening}) into dataset for future context...")
        append_to_dataset(csv_file, next_date, morning, evening, market)
        
        # 4. Rebuild features so the AI knows about the new streaks/lags
        subprocess.run(['python', feature_script])

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python auto_predictor.py [kalyan|mb] [target_date_YYYY-MM-DD]")
        sys.exit(1)
        
    market_arg = sys.argv[1].lower()
    target_date_arg = sys.argv[2]
    
    run_auto_predictor(market_arg, target_date_arg)

#qui estraevo direttamente tutti i dati json in csv separati (che ho inserito nella cartella csv_output_from_json)
# , con i campi indicati nella funzione "extract_attributes"
# serve
import json
import pandas as pd
import os

def extract_attributes(entry):
    bug = entry.get('bug', {})
    attributes = {
        'short_desc': bug.get('short_desc', ''),
        'product': bug.get('product', ''),
        'priority': bug.get('priority', ''),
        'bug_severity': bug.get('bug_severity', ''),
        'days_resolution': bug.get('days_resolution', '')
    }
    
    # Extract all long_desc (comments) and concatenate them
    long_desc_list = bug.get("long_desc", [])
    all_long_desc = "\n\n".join(comment.get("thetext", "") for comment in long_desc_list)
    attributes['comments'] = all_long_desc
    # Extract 'thetext' from the first comment if available
    #long_desc = bug.get('long_desc', [])
    #if long_desc and isinstance(long_desc, list) and len(long_desc) > 0:
    #    attributes['comments'] = long_desc[0].get('thetext', '')
    #else:
    #    attributes['comments'] = ''
    
    return attributes

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        try:
            # Try to parse the entire file as a single JSON object
            data = json.load(file)
            if isinstance(data, dict):
                return [data]  # Wrap single object in a list
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError:
            # If parsing as a single object fails, try line by line
            file.seek(0)  # Reset file pointer to the beginning
            data = []
            for line in file:
                try:
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}")
    return data

def json_to_csv(json_file, csv_file):
    # Check if the file exists
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"The file {json_file} does not exist.")

    # Read JSON file
    data = read_json_file(json_file)

    if not data:
        raise ValueError("No valid JSON objects found in the file.")

    # Extract specified attributes from each entry
    extracted_data = [extract_attributes(entry) for entry in data]

    # Create a DataFrame
    df = pd.DataFrame(extracted_data)

    # Fill NaN values with an empty string
    df = df.fillna('')

    # Write to CSV
    df.to_csv(csv_file, index=False)

    print(f"CSV file '{csv_file}' has been created successfully.")

files = ['Eclipse','FreeDesktop','Gentoo','KDE','LibreOffice','LiveCode','NetBeans','Novell','OpenOffice','OpenXchange','W3C']
# Usage
for file in files:
    json_file = 'Dataset/'+ file +'_original.json'  # Replace with your JSON file path
    csv_file = 'csv_output_from_json/'+file + '_data.csv'  # Replace with your desired output CSV file path
    try:
        json_to_csv(json_file, csv_file)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
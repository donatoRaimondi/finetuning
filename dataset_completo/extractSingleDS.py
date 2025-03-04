# serve per estrarre in csv il singolo json e lo "appiattiamo" poichè i dati non sono organizzati in modo sequenziale 

import json
import pandas as pd
import os

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            for i, a in enumerate(x):
                flatten(a, name + str(i) + '_')
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def read_json_file(file_path):
    data = []
    with open(file_path, 'r') as file:
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

    # Flatten each entry
    flattened_data = [flatten_json(entry) for entry in data]

    # Create a DataFrame
    df = pd.DataFrame(flattened_data)

    # Fill NaN values with an empty string
    df = df.fillna('')

    # Write to CSV
    df.to_csv(csv_file, index=False)

    print(f"CSV file '{csv_file}' has been created successfully.")

# Usage
json_file = 'Dataset/FreeDesktop_original.json'  # Replace with your JSON file path #come json_file qui estraevo uno ad uno i dataset json
csv_file = 'output.csv'  # Replace with your desired output CSV file path

try:
    json_to_csv(json_file, csv_file)
except Exception as e:
    print(f"An error occurred: {str(e)}")
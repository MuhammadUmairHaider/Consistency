import pandas as pd
import numpy as np
import re

def parse_data(text):
    # Find all tau sections
    tau_sections = re.split(r'Tao:\s+', text)[1:]  # Skip the first empty element
    
    # Initialize dictionary to store results
    results = {}
    
    for section in tau_sections:
        # Extract tau value
        tau_match = re.match(r'(\d+\.?\d*)', section)
        if not tau_match:
            continue
        
        tau = float(tau_match.group(1))
        
        # Extract class data
        class_data = []
        for class_idx in range(4):
            pattern = r'Class ' + str(class_idx) + r'\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)\s+'
            match = re.search(pattern, section)
            
            if match:
                row_data = [float(match.group(i)) for i in range(1, 9)]
                class_data.append(row_data)
        
        # Convert to numpy array
        if class_data:
            results[tau] = np.array(class_data)
    
    return results

def create_averages_table(parsed_data):
    # Column names
    columns = [
        'Base Accuracy', 'Base Confidence', 'Base Complement Acc', 'Base Compliment Conf',
        'STD Accuracy', 'STD Confidence', 'STD compliment ACC', 'STD compliment Conf'
    ]
    
    # Calculate averages for each tau value
    results = []
    for tau, data in sorted(parsed_data.items()):
        row = [tau]
        row.extend([np.mean(data[:, i]) for i in range(data.shape[1])])
        results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results, columns=['Tau'] + columns)
    return df.round(4)

# Read data from file
with open('tao_abiliation.txt', 'r') as file:
    data = file.read()

# Parse the data
parsed_data = parse_data(data)

# Create and display the averages table
result_table = create_averages_table(parsed_data)
print(result_table)

# Save to CSV
result_table.to_csv('tau_averages.csv', index=False)
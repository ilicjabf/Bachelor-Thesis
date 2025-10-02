import re
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Define the log parsing function (provided by you)
def parse_log_file(filepath):
    logs = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            timestamp_str, event_type, pid, comm = parts[:4]
            extra = ','.join(parts[4:])
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                continue
            logs.append({
                'timestamp': timestamp,
                'event_type': event_type,
                'pid': int(pid),
                'comm': comm,
                'extra': extra
            })
    return pd.DataFrame(logs)

# Define a function to parse the 'extra' field and extract relevant details
def parse_extra_fields(extra_str, event_type):
    # For OT_INT logs, the last field is the syscall number
    if event_type == 'OT_INT':
        # Extract the last number as syscall_nr
        syscall_nr = extra_str.strip().split(',')[-1]
        return syscall_nr, None, None
    else:
        # Regex to match details like syscall_nr, ret, address
        syscall_nr_match = re.search(r'syscall_nr=(\d+)', extra_str)
        ret_match = re.search(r'ret=(-?\d+)', extra_str)
        address_match = re.search(r'address=(0x[0-9a-fA-F]+)', extra_str)
        
        syscall_nr = syscall_nr_match.group(1) if syscall_nr_match else None
        ret = ret_match.group(1) if ret_match else None
        address = address_match.group(1) if address_match else None
        
        return syscall_nr, ret, address

# Function to extract the unique log entries based on key attributes
def extract_unique_logs(df):
    unique_logs = set()
    
    for _, row in df.iterrows():
        syscall_nr, ret, address = parse_extra_fields(row['extra'], row['event_type'])
        
        # Create a unique tuple of event info (pid is excluded)
        # log_key = (row['event_type'], syscall_nr, row['comm'], ret, address)
        log_key = (row['event_type'], syscall_nr, row['comm'], ret)
        unique_logs.add(log_key)
    
    return unique_logs

def print_grouped_by_event_type(title, unique_logs):
    print(f"\n{title}:")
    grouped = defaultdict(list)
    
    for log in unique_logs:
        event_type = log[0]
        grouped[event_type].append(log)

    for event_type in sorted(grouped.keys()):
        print(f"\nEvent Type: {event_type}")
        for log in sorted(grouped[event_type]):
            print(f"  {log}")

# Process log files
log_file_1 = 'ssh_VM.log'
log_file_2 = 'ssh_VM_attack.log'

# Parse the log files
df1 = parse_log_file(log_file_1)
df2 = parse_log_file(log_file_2)

# Extract unique logs from both DataFrames
unique_logs_1 = extract_unique_logs(df1)
unique_logs_2 = extract_unique_logs(df2)

# Find the unique logs in log_file_1 that are not in log_file_2
unique_to_file_1 = unique_logs_1.difference(unique_logs_2)

# Find the unique logs in log_file_2 that are not in log_file_1
unique_to_file_2 = unique_logs_2.difference(unique_logs_1)

# # Print the unique logs from log_file_1
# print("Unique logs in log_file_1:")
# for log in unique_to_file_1:
#     print(log)

# # Print the unique logs from log_file_2
# print("\nUnique logs in log_file_2:")
# for log in unique_to_file_2:
#     print(log)

print_grouped_by_event_type("Unique logs in log_file_1", unique_to_file_1)
print_grouped_by_event_type("Unique logs in log_file_2", unique_to_file_2)

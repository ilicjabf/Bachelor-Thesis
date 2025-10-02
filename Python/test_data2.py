import pandas as pd
from datetime import datetime, timedelta
from collections import Counter, defaultdict

WINDOW_SIZE_MS = 500

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

def extract_syscalls(df):
    syscalls = []
    for _, row in df.iterrows():
        if row['event_type'] == "SYSCALL_ENTRY":
            parts = row['extra'].split("syscall_nr=")
            if len(parts) > 1:
                try:
                    syscall_num = int(parts[1].split(',')[0])
                    syscalls.append(syscall_num)
                except ValueError:
                    continue
    return syscalls

def extract_syscall_sequences(df):
    sequences = defaultdict(list)
    for _, row in df.iterrows():
        if row['event_type'] == "SYSCALL_ENTRY":
            pid = row['pid']
            parts = row['extra'].split("syscall_nr=")
            if len(parts) > 1:
                try:
                    syscall_num = int(parts[1].split(',')[0])
                    sequences[pid].append(syscall_num)
                except ValueError:
                    continue
    return sequences

def sliding_window_analysis(df, window_size_ms=500):
    results = []
    df = df.sort_values('timestamp').reset_index(drop=True)
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    delta = timedelta(milliseconds=window_size_ms)

    current = start_time
    while current < end_time:
        next_time = current + delta
        window_df = df[(df['timestamp'] >= current) & (df['timestamp'] < next_time)]

        if not window_df.empty and 'sshd' in window_df['comm'].values:
            freq = Counter(window_df['event_type'])
            syscalls = extract_syscalls(window_df)
            syscall_seqs = extract_syscall_sequences(window_df)
            results.append({
                'start': current,
                'end': next_time,
                'freq': freq,
                'syscalls': syscalls,
                'sequences': syscall_seqs
            })

        current = next_time
    return results

def summarize_results(results):
    for i, entry in enumerate(results):
        print(f"\nWindow {i+1} — {entry['start']} to {entry['end']}")
        print("  Event Types:")
        for k, v in entry['freq'].items():
            print(f"    {k}: {v}")
        print("  Unique Syscalls:", sorted(set(entry['syscalls'])))
        print("  Syscall Sequences:")
        for pid, seq in entry['sequences'].items():
            print(f"    PID {pid}: {' → '.join(map(str, seq))}")

# === MAIN ===

log_file = "ssh_HV.log"  
df = parse_log_file(log_file)
results = sliding_window_analysis(df, WINDOW_SIZE_MS)
summarize_results(results)

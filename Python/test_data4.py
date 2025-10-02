import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity

WINDOW_SIZE_MS = 200

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
        
        if not window_df.empty:
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

def compute_window_similarity(normal_results, abnormal_results):
    """Compute similarity between corresponding windows in normal and abnormal logs"""
    similarities = []
    
    # Ensure we have the same number of windows to compare
    min_windows = min(len(normal_results), len(abnormal_results))
    
    for i in range(min_windows):
        # Create feature vectors based on event frequencies
        normal_features = []
        abnormal_features = []
        
        # Get all unique event types across both windows
        all_event_types = set(normal_results[i]['freq'].keys()) | set(abnormal_results[i]['freq'].keys())
        
        # Create feature vectors
        for event_type in sorted(all_event_types):
            normal_features.append(normal_results[i]['freq'].get(event_type, 0))
            abnormal_features.append(abnormal_results[i]['freq'].get(event_type, 0))
        
        # Convert to numpy arrays for similarity calculation
        normal_vec = np.array(normal_features).reshape(1, -1)
        abnormal_vec = np.array(abnormal_features).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(normal_vec, abnormal_vec)[0][0]
        
        similarities.append({
            'window_index': i,
            'start_time': normal_results[i]['start'],
            'end_time': normal_results[i]['end'],
            'similarity': similarity
        })
    
    return similarities

def plot_event_frequencies(results, title):
    """Plot event frequencies across all windows"""
    # Extract window information
    window_indices = []
    events_count = defaultdict(list)
    
    for i, entry in enumerate(results):
        window_indices.append(i)
        for event_type, count in entry['freq'].items():
            events_count[event_type].append(count)
    
    # Make sure all event types have values for all windows
    all_event_types = set()
    for entry in results:
        all_event_types.update(entry['freq'].keys())
    
    for event_type in all_event_types:
        if event_type not in events_count:
            events_count[event_type] = [0] * len(window_indices)
        elif len(events_count[event_type]) < len(window_indices):
            events_count[event_type].extend([0] * (len(window_indices) - len(events_count[event_type])))
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    for event_type, counts in events_count.items():
        plt.plot(window_indices, counts, label=event_type, marker='o')
    
    plt.xlabel('Window Index')
    plt.ylabel('Event Count')
    plt.title(f'Event Frequencies Over Time: {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return plt

def plot_similarity(similarities):
    """Plot similarity between normal and abnormal windows"""
    plt.figure(figsize=(12, 6))
    
    window_indices = [s['window_index'] for s in similarities]
    similarity_values = [s['similarity'] for s in similarities]
    
    plt.plot(window_indices, similarity_values, marker='o', linestyle='-', color='green')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Similarity')
    
    plt.xlabel('Window Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Similarity Between Normal and Abnormal Logs')
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    
    return plt

# === MAIN ===
normal_log_file = "ssh_VM.log"
abnormal_log_file = "ssh_VM_attack.log"

# Parse both log files
print(f"Parsing normal log file: {normal_log_file}")
normal_df = parse_log_file(normal_log_file)
print(f"Found {len(normal_df)} log entries in normal log file")

print(f"\nParsing abnormal log file: {abnormal_log_file}")
abnormal_df = parse_log_file(abnormal_log_file)
print(f"Found {len(abnormal_df)} log entries in abnormal log file")

# Analyze both files with sliding windows
print("\nAnalyzing normal log file...")
normal_results = sliding_window_analysis(normal_df, WINDOW_SIZE_MS)
print(f"Created {len(normal_results)} windows for normal log file")

print("\nAnalyzing abnormal log file...")
abnormal_results = sliding_window_analysis(abnormal_df, WINDOW_SIZE_MS)
print(f"Created {len(abnormal_results)} windows for abnormal log file")

# Compute similarity between corresponding windows
print("\nComputing similarity between normal and abnormal windows...")
similarities = compute_window_similarity(normal_results, abnormal_results)

# Print similarity results
print("\nSimilarity Results:")
for sim in similarities:
    print(f"Window {sim['window_index']+1} ({sim['start_time']} to {sim['end_time']}): Similarity = {sim['similarity']:.4f}")

# Plot the event frequencies for both logs
print("\nGenerating frequency plots...")
normal_plot = plot_event_frequencies(normal_results, "Normal Log")
normal_plot.savefig("normal_event_frequencies.png")

abnormal_plot = plot_event_frequencies(abnormal_results, "Abnormal Log")
abnormal_plot.savefig("abnormal_event_frequencies.png")

# Plot the similarity between windows
sim_plot = plot_similarity(similarities)
sim_plot.savefig("window_similarities.png")

print("\nPlots saved as normal_event_frequencies.png, abnormal_event_frequencies.png, and window_similarities.png")

# Summarize results for both log files if needed
print("\n=== NORMAL LOG SUMMARY ===")
summarize_results(normal_results[:3])  # Show first 3 windows only to avoid excessive output
print("... (additional windows omitted)")

print("\n=== ABNORMAL LOG SUMMARY ===")
summarize_results(abnormal_results[:3])  # Show first 3 windows only
print("... (additional windows omitted)")

print("\nAnalysis complete!")

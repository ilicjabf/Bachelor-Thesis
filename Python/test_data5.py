import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, csr_matrix

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

def extract_syscall_ngrams(normal_results, abnormal_results, n=2):
    """
    Extract n-grams from syscall sequences across all windows
    using a common vocabulary for both normal and abnormal logs
    
    Args:
        normal_results: List of window results from normal log analysis
        abnormal_results: List of window results from abnormal log analysis
        n: Size of n-grams to extract
        
    Returns:
        normal_matrix: Normalized n-gram matrix for normal logs
        abnormal_matrix: Normalized n-gram matrix for abnormal logs
        feature_names: List of common n-gram feature names
    """
    # Prepare syscall sequences as text for both logs
    normal_syscall_texts = []
    abnormal_syscall_texts = []
    
    # Process normal logs
    for window in normal_results:
        all_syscalls = []
        for pid, seq in window['sequences'].items():
            all_syscalls.extend(seq)
        normal_syscall_texts.append(' '.join(map(str, all_syscalls)))
    
    # Process abnormal logs
    for window in abnormal_results:
        all_syscalls = []
        for pid, seq in window['sequences'].items():
            all_syscalls.extend(seq)
        abnormal_syscall_texts.append(' '.join(map(str, all_syscalls)))
    
    # Combine all texts to create a common vocabulary
    all_syscall_texts = normal_syscall_texts + abnormal_syscall_texts
    
    # Create and fit the vectorizer on all data
    vectorizer = CountVectorizer(ngram_range=(n, n), token_pattern=r'\b\d+\b')
    vectorizer.fit(all_syscall_texts)
    
    # Transform each dataset separately using the common vocabulary
    normal_matrix = vectorizer.transform(normal_syscall_texts)
    abnormal_matrix = vectorizer.transform(abnormal_syscall_texts)
    
    feature_names = vectorizer.get_feature_names_out()
    
    return normal_matrix, abnormal_matrix, feature_names, vectorizer

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

def compute_ngram_similarity(normal_ngram_matrix, abnormal_ngram_matrix):
    """Compute similarity between windows based on syscall n-grams"""
    similarities = []
    
    # Get minimum number of windows
    min_windows = min(normal_ngram_matrix.shape[0], abnormal_ngram_matrix.shape[0])
    
    for i in range(min_windows):
        # Extract the i-th row from each matrix
        normal_vec = normal_ngram_matrix[i].reshape(1, -1)
        abnormal_vec = abnormal_ngram_matrix[i].reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(normal_vec, abnormal_vec)[0][0]
        
        similarities.append({
            'window_index': i,
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

def plot_similarity(event_similarities, ngram_similarities, n):
    """Plot similarity between normal and abnormal windows"""
    plt.figure(figsize=(12, 6))
    
    window_indices = [s['window_index'] for s in event_similarities]
    event_sim_values = [s['similarity'] for s in event_similarities]
    ngram_sim_values = [s['similarity'] for s in ngram_similarities]
    
    plt.plot(window_indices, event_sim_values, marker='o', linestyle='-', color='blue', 
             label='Event Type Similarity')
    plt.plot(window_indices, ngram_sim_values, marker='s', linestyle='-', color='green', 
             label=f'Syscall {n}-gram Similarity')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Similarity')
    
    plt.xlabel('Window Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Similarity Between Normal and Abnormal Logs')
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    
    return plt

def plot_top_ngrams(ngram_matrix, feature_names, title, top_n=10):
    """Plot the most frequent n-grams across all windows"""
    # Sum up occurrences of each n-gram across all windows
    ngram_sums = np.asarray(ngram_matrix.sum(axis=0)).flatten()
    
    # Get indices of top n-grams
    top_indices = np.argsort(ngram_sums)[-top_n:]
    
    # Create a horizontal bar chart
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_indices))
    
    plt.barh(y_pos, ngram_sums[top_indices], align='center')
    plt.yticks(y_pos, [feature_names[i] for i in top_indices])
    plt.xlabel('Frequency')
    plt.title(f'Top {top_n} Syscall N-grams: {title}')
    plt.tight_layout()
    
    return plt

def plot_ngram_heatmap(ngram_matrix, feature_names, title, top_n=20):
    """Create a heatmap of the top n-grams across windows"""
    # Sum occurrences across all windows
    ngram_sums = np.asarray(ngram_matrix.sum(axis=0)).flatten()
    
    # Get indices of top n-grams
    top_indices = np.argsort(ngram_sums)[-top_n:]
    top_ngrams = [feature_names[i] for i in top_indices]
    
    # Create a dense matrix for the top n-grams
    heatmap_data = ngram_matrix[:, top_indices].toarray()
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data.T, aspect='auto', cmap='viridis')
    
    plt.yticks(range(len(top_ngrams)), top_ngrams)
    plt.xlabel('Window Index')
    plt.ylabel('N-gram')
    plt.title(f'Top {top_n} N-gram Occurrences Across Windows: {title}')
    plt.colorbar(label='Frequency')
    plt.tight_layout()
    
    return plt

def plot_ngram_differences(normal_matrix, abnormal_matrix, feature_names, top_n=15):
    """Plot n-grams with the largest differences between normal and abnormal logs"""
    # Sum occurrences across windows for each matrix
    normal_sums = np.asarray(normal_matrix.sum(axis=0)).flatten()
    abnormal_sums = np.asarray(abnormal_matrix.sum(axis=0)).flatten()
    
    # Calculate differences (ensure arrays are the same length)
    differences = np.zeros(len(feature_names))
    differences[:len(normal_sums)] = normal_sums
    
    if len(abnormal_sums) < len(differences):
        differences[:len(abnormal_sums)] -= abnormal_sums
    else:
        differences -= abnormal_sums[:len(differences)]
    
    # Get absolute differences
    abs_differences = np.abs(differences)
    
    # Get indices of top differences
    top_indices = np.argsort(abs_differences)[-top_n:]
    
    # Get actual values for both normal and abnormal
    normal_values = normal_sums[top_indices]
    
    # Make sure abnormal_values has correct length
    abnormal_values = np.zeros(len(top_indices))
    for i, idx in enumerate(top_indices):
        if idx < len(abnormal_sums):
            abnormal_values[i] = abnormal_sums[idx]
    
    # Create a grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(top_indices))
    width = 0.35
    
    ax.bar(x - width/2, normal_values, width, label='Normal')
    ax.bar(x + width/2, abnormal_values, width, label='Abnormal')
    
    ax.set_ylabel('Frequency')
    ax.set_title('N-grams with Largest Differences Between Normal and Abnormal Logs')
    ax.set_xticks(x)
    ax.set_xticklabels([feature_names[i] for i in top_indices], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    return plt

# === MAIN ===
normal_log_file = "ssh_HV.log"
abnormal_log_file = "ssh_HV_attack.log"
ngram_size = 20 

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

# Extract syscall n-grams with common vocabulary
print(f"\nExtracting syscall {ngram_size}-grams from both log files...")
normal_ngram_matrix, abnormal_ngram_matrix, feature_names, vectorizer = extract_syscall_ngrams(
    normal_results, abnormal_results, ngram_size)

print(f"Created a common feature space with {len(feature_names)} unique {ngram_size}-grams")
print(f"Normal matrix shape: {normal_ngram_matrix.shape}")
print(f"Abnormal matrix shape: {abnormal_ngram_matrix.shape}")

# Compute similarity between corresponding windows based on event types
print("\nComputing event-based similarity between normal and abnormal windows...")
event_similarities = compute_window_similarity(normal_results, abnormal_results)

# Compute similarity between corresponding windows based on n-grams
print(f"\nComputing {ngram_size}-gram similarity between normal and abnormal windows...")
ngram_similarities = compute_ngram_similarity(normal_ngram_matrix, abnormal_ngram_matrix)

# Print similarity results
print("\nSimilarity Results:")
for i, (event_sim, ngram_sim) in enumerate(zip(event_similarities, ngram_similarities)):
    print(f"Window {i+1}: Event Similarity = {event_sim['similarity']:.4f}, {ngram_size}-gram Similarity = {ngram_sim['similarity']:.4f}")

# Plot the event frequencies for both logs
print("\nGenerating frequency plots...")
normal_plot = plot_event_frequencies(normal_results, "Normal Log")
normal_plot.savefig("normal_event_frequencies.png")

abnormal_plot = plot_event_frequencies(abnormal_results, "Abnormal Log")
abnormal_plot.savefig("abnormal_event_frequencies.png")

# Plot the similarity between windows
similarity_plot = plot_similarity(event_similarities, ngram_similarities, ngram_size)
similarity_plot.savefig("window_similarities.png")

# Plot top n-grams for both logs
print("\nGenerating n-gram visualization...")
normal_ngram_plot = plot_top_ngrams(normal_ngram_matrix, feature_names, "Normal Log", top_n=15)
normal_ngram_plot.savefig("normal_top_ngrams.png")

abnormal_ngram_plot = plot_top_ngrams(abnormal_ngram_matrix, feature_names, "Abnormal Log", top_n=15)
abnormal_ngram_plot.savefig("abnormal_top_ngrams.png")

# Create heatmaps for both logs
normal_heatmap = plot_ngram_heatmap(normal_ngram_matrix, feature_names, "Normal Log")
normal_heatmap.savefig("normal_ngram_heatmap.png")

abnormal_heatmap = plot_ngram_heatmap(abnormal_ngram_matrix, feature_names, "Abnormal Log")
abnormal_heatmap.savefig("abnormal_ngram_heatmap.png")

# Plot n-gram differences
diff_plot = plot_ngram_differences(normal_ngram_matrix, abnormal_ngram_matrix, feature_names)
diff_plot.savefig("ngram_differences.png")

print("\nVisualizations saved as:")
print("- normal_event_frequencies.png")
print("- abnormal_event_frequencies.png")
print("- window_similarities.png")
print("- normal_top_ngrams.png")
print("- abnormal_top_ngrams.png")
print("- normal_ngram_heatmap.png")
print("- abnormal_ngram_heatmap.png")
print("- ngram_differences.png")

# Summarize results for both log files if needed
print("\n=== NORMAL LOG SUMMARY ===")
summarize_results(normal_results[:3])  # Show first 3 windows only to avoid excessive output
print("... (additional windows omitted)")

print("\n=== ABNORMAL LOG SUMMARY ===")
summarize_results(abnormal_results[:3])  # Show first 3 windows only
print("... (additional windows omitted)")

print("\nAnalysis complete!")

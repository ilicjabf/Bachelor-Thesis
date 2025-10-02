import pandas as pd
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import numpy as np
import re

# --- Helper functions ---
def parse_log_file(filepath):
    log_data = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            if parts[0] == 'Timestamp':  # skip header
                continue
            try:
                timestamp = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                continue  # optionally log or raise
            event_type = parts[1]
            pid = int(parts[2])
            comm = parts[3]
            details = ','.join(parts[4:])
            log_data.append((timestamp, event_type, pid, comm, details))
    return pd.DataFrame(log_data, columns=['timestamp', 'event_type', 'pid', 'comm', 'details'])

def extract_windows(df, window_size_ms=500):
    windows = []
    start = df['timestamp'].min()
    end = df['timestamp'].max()
    window_delta = timedelta(milliseconds=window_size_ms)

    while start < end:
        window_end = start + window_delta
        window_df = df[(df['timestamp'] >= start) & (df['timestamp'] < window_end)]
        if not window_df[window_df['comm'].str.contains('sshd')].empty:
            windows.append(window_df)
        start = window_end
    return windows

def compute_event_frequencies(windows):
    freq_list = []
    for win in windows:
        counts = dict(Counter(win['event_type']))
        freq_list.append(counts)
    return freq_list

def plot_event_frequencies(freqs, title):
    all_keys = set()
    for d in freqs:
        all_keys.update(d.keys())
    all_keys = sorted(all_keys)

    values = []
    for f in freqs:
        values.append([f.get(k, 0) for k in all_keys])

    values = np.array(values).T
    x = np.arange(len(freqs))

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(freqs))
    for idx, key in enumerate(all_keys):
        ax.bar(x, values[idx], bottom=bottom, label=key)
        bottom += values[idx]

    ax.set_title(title)
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.show()

def extract_syscall_ngrams(windows, n=2):
    vectorizer = CountVectorizer(token_pattern=r'\d+', ngram_range=(n, n))
    syscall_texts = []
    for win in windows:
        syscalls = win[win['event_type'].str.contains('SYSCALL_ENTRY')]['details']
        syscall_seq = ' '.join(re.findall(r'syscall_nr=(\d+)', ' '.join(syscalls)))
        syscall_texts.append(syscall_seq)
    return vectorizer, vectorizer.fit_transform(syscall_texts)

# def compute_similarity(vec1, vec2):
#     min_len = min(vec1.shape[0], vec2.shape[0])
#     cos_sims = [cosine_similarity(vec1[i], vec2[i])[0][0] for i in range(min_len)]
#     jacc_sims = []
#     for i in range(min_len):
#         s1 = set(vec1[i].nonzero()[1])
#         s2 = set(vec2[i].nonzero()[1])
#         inter = len(s1 & s2)
#         union = len(s1 | s2)
#         jacc = inter / union if union > 0 else 0
#         jacc_sims.append(jacc)
#     return cos_sims, jacc_sims

def compute_similarity(base_vec, suspect_vec):
    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Convert sparse matrices to dense arrays if needed for processing
    if isinstance(base_vec, csr_matrix):
        base_vec_array = base_vec.toarray().flatten()  # Convert to 1D array
    else:
        base_vec_array = np.array(base_vec).flatten()
        
    if isinstance(suspect_vec, csr_matrix):
        suspect_vec_array = suspect_vec.toarray().flatten()  # Convert to 1D array
    else:
        suspect_vec_array = np.array(suspect_vec).flatten()
    
    # Get lengths after converting to array
    base_len = len(base_vec_array)
    suspect_len = len(suspect_vec_array)
    
    # Ensure both vectors are the same length (pad with zeros if necessary)
    max_len = max(base_len, suspect_len)
    
    # Pad the shorter vector with zeros
    if base_len < max_len:
        base_vec_padded = np.pad(base_vec_array, (0, max_len - base_len))
    else:
        base_vec_padded = base_vec_array
        
    if suspect_len < max_len:
        suspect_vec_padded = np.pad(suspect_vec_array, (0, max_len - suspect_len))
    else:
        suspect_vec_padded = suspect_vec_array
    
    # Reshape for cosine_similarity function (which expects 2D arrays)
    base_vec_padded = base_vec_padded.reshape(1, -1)  # Shape: (1, max_len)
    suspect_vec_padded = suspect_vec_padded.reshape(1, -1)  # Shape: (1, max_len)
    
    # Compute cosine similarity
    cos_sims = cosine_similarity(base_vec_padded, suspect_vec_padded)[0][0]
    
    # For Jaccard similarity, compute intersection and union
    # Create sets from the non-zero indices of each vector
    base_set = set(np.nonzero(base_vec_padded)[1])  # Use [1] to get column indices in 2D array
    suspect_set = set(np.nonzero(suspect_vec_padded)[1])
    
    intersection = len(base_set.intersection(suspect_set))
    union = len(base_set.union(suspect_set))
    
    jacc_sim = intersection / union if union != 0 else 0
    
    return cos_sims, jacc_sim


# --- Main analysis ---
base_log = parse_log_file('ssh_VM.log')
suspect_log = parse_log_file('ssh_VM_attack.log')

base_windows = extract_windows(base_log)
suspect_windows = extract_windows(suspect_log)

base_freqs = compute_event_frequencies(base_windows)
suspect_freqs = compute_event_frequencies(suspect_windows)

plot_event_frequencies(base_freqs, "Baseline Event Frequencies Per Window")
plot_event_frequencies(suspect_freqs, "Suspect Event Frequencies Per Window")

# n-gram analysis
vectorizer, base_vec = extract_syscall_ngrams(base_windows)
_, suspect_vec = extract_syscall_ngrams(suspect_windows)

cos_sims, jacc_sims = compute_similarity(base_vec, suspect_vec)

print("Average Cosine Similarity:", np.mean(cos_sims))
print("Average Jaccard Similarity:", np.mean(jacc_sims))

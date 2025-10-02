import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from itertools import islice

# ----------------------
# Log Parsing
# ----------------------
def parse_log_line(line):
    try:
        parts = line.strip().split(',')
        ts = parts[0]
        event_type = parts[1]
        pid = int(parts[2])
        comm = parts[3]
        extra = {}
        for p in parts[4:]:
            if '=' in p:
                k, v = p.split('=')
                extra[k] = int(v) if v.lstrip('-').isdigit() else v
        return {"timestamp": ts, "event_type": event_type, "pid": pid, "comm": comm, **extra}
    except:
        return None

# ----------------------
# Load logs
# ----------------------
def load_log_file(path):
    with open(path) as f:
        lines = f.readlines()
    parsed = [parse_log_line(line) for line in lines if line.strip()]
    df = pd.DataFrame([p for p in parsed if p is not None])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    return df

# ----------------------
# Frequency analysis
# ----------------------
def event_frequencies(df):
    # print(df["event_type"].value_counts().sort_index())
    return df["event_type"].value_counts().sort_index()

def syscall_errors(df):
    errors = df[(df["event_type"] == "SYSCALL_EXIT") & (df["ret"].astype(str).str.startswith("-"))]
    return errors["ret"].value_counts()

# ----------------------
# Extract sshd sequences
# ----------------------
def extract_sshd_syscall_sequences(df):
    sshd_df = df[(df["comm"] == "sudo") & (df["event_type"].str.contains("SYSCALL"))]
    grouped = sshd_df.groupby("pid")
    sequences = []
    for _, group in grouped:
        sequence = group.sort_values("timestamp")["syscall_nr"].tolist()
        sequences.append(sequence)
    return sequences

# ----------------------
# Build syscall transition matrix
# ----------------------
def build_transition_matrix(sequences):
    transitions = defaultdict(lambda: defaultdict(int))
    for seq in sequences:
        for a, b in zip(seq, islice(seq, 1, None)):
            transitions[a][b] += 1
    return transitions

def normalize_matrix(matrix):
    normalized = {}
    for src, targets in matrix.items():
        total = sum(targets.values())
        normalized[src] = {dst: count / total for dst, count in targets.items()}
    return normalized

# ----------------------
# Anomaly scoring
# ----------------------
def sequence_anomaly_score(sequence, transition_probs):
    score = 0.0
    for a, b in zip(sequence, islice(sequence, 1, None)):
        prob = transition_probs.get(a, {}).get(b, 1e-6)
        score += -np.log(prob)  # negative log likelihood
    return score

# ----------------------
# Main logic
# ----------------------

if __name__ == "__main__":
    baseline_df = load_log_file("sudo.log")
    suspect_df = load_log_file("sudo_attack.log")

    # Frequency Analysis
    baseline_freqs = event_frequencies(baseline_df)
    suspect_freqs = event_frequencies(suspect_df)
    freq_comparison = pd.concat([baseline_freqs, suspect_freqs], axis=1).fillna(0)
    freq_comparison.columns = ["baseline", "suspect"]
    print("\n--- Event Frequency Comparison ---")
    print(freq_comparison)

    # Syscall Error Analysis
    print("\n--- Baseline Syscall Errors ---")
    print(syscall_errors(baseline_df))
    print("\n--- Suspect Syscall Errors ---")
    print(syscall_errors(suspect_df))

    # Syscall Sequences
    baseline_sequences = extract_sshd_syscall_sequences(baseline_df)
    suspect_sequences = extract_sshd_syscall_sequences(suspect_df)

    # Markov Chain Model
    transition_matrix = build_transition_matrix(baseline_sequences)
    transition_probs = normalize_matrix(transition_matrix)

    # Score suspect sequences
    scores = [sequence_anomaly_score(seq, transition_probs) for seq in suspect_sequences]
    print("\n--- Anomaly Scores for Suspect sudo Sequences ---")
    for i, score in enumerate(scores):
        print(f"Sequence {i+1}: Score = {score:.2f}")

    # Optional: Plot frequencies
    freq_comparison.plot.bar(title="Event Type Frequency Comparison", figsize=(12,6))
    plt.tight_layout()
    plt.show()

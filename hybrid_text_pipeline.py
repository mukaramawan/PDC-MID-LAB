import csv
import re
import time
import argparse
from collections import Counter
from mpi4py import MPI
from multiprocessing import Pool, cpu_count

WORD_RE = re.compile(r"\b[a-zA-Z']+\b")

def clean_and_tokenize(text):
    if not text:
        return []
    return WORD_RE.findall(text.lower())

def process_lines_worker(lines):
    """For multiprocessing Pool: worker-level processing of a small list of reviews."""
    c = Counter()
    for text in lines:
        c.update(clean_and_tokenize(text))
    return c

def split_for_local_workers(lines, n):
    """Split list lines into n subchunks for local worker processes."""
    k, m = divmod(len(lines), n)
    chunks = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        chunks.append(lines[start:start+size])
        start += size
    return chunks

def read_reviews_with_sentiment(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r.get('review','') or '', r.get('sentiment','') or ''))
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--local-workers', type=int, default=None, help='Local multiprocessing workers per MPI rank')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        rows = read_reviews_with_sentiment(args.csv)
        # create chunks per MPI rank (equal)
        chunks = []
        k, m = divmod(len(rows), size)
        start = 0
        for i in range(size):
            sz = k + (1 if i < m else 0)
            chunks.append(rows[start:start+sz])
            start += sz
        print(f"[master] Read {len(rows)} reviews, dispatching to {size} ranks")
    else:
        chunks = None

    # Scatter chunks of (review, sentiment)
    local_rows = comm.scatter(chunks, root=0)

    # Determine local worker count
    local_workers = args.local_workers or max(1, cpu_count() // 2)
    # apply sentiment filter per spec:
    if rank == 0:
        # only positive reviews
        filtered = [r for r,s in local_rows if s.lower().startswith('pos')]
    elif rank == 1:
        # only negative reviews
        filtered = [r for r,s in local_rows if s.lower().startswith('neg')]
    else:
        filtered = [r for r,s in local_rows]

    # spawn local multiprocessing pool to process filtered reviews
    t_comm_start = time.time()
    # Split for local pool
    if len(filtered) == 0:
        local_counter = Counter()
        local_proc_time = 0.0
    else:
        subchunks = split_for_local_workers(filtered, local_workers)
        t_local_start = time.time()
        with Pool(processes=local_workers) as p:
            results = p.map(process_lines_worker, subchunks)
        t_local_end = time.time()
        local_proc_time = t_local_end - t_local_start
        local_counter = Counter()
        for r in results:
            local_counter.update(r)
    t_comm_end = time.time()
    comm_overhead = 0.0  # measured later: time to send results

    # Prepare payload to send back to master
    payload = dict(local_counter)

    # measure communication send/recv time
    t_before_comm = time.time()
    gathered = comm.gather(payload, root=0)
    t_after_comm = time.time()
    # gather also local processing times
    proc_times = comm.gather(local_proc_time, root=0)
    comm_times = comm.gather(t_after_comm - t_before_comm, root=0)

    if rank == 0:
        # master aggregates
        total_counter = Counter()
        for d in gathered:
            total_counter.update(d)
        # compute stats
        for r in range(size):
            lines_for_r = len(chunks[r])
            # count filtered per rank:
            if r == 0:
                filtered_count = sum(1 for _,s in chunks[r] if s.lower().startswith('pos'))
            elif r == 1:
                filtered_count = sum(1 for _,s in chunks[r] if s.lower().startswith('neg'))
            else:
                filtered_count = len(chunks[r])
            print(f"Node {r}: processed {filtered_count} reviews in {proc_times[r]:.3f}s (comm {comm_times[r]:.3f}s)")
        hybrid_total = max(proc_times) + max(comm_times)
        communication_overhead = sum(comm_times)
        print(f"\nCommunication overhead (sum of per-node comm times): {communication_overhead:.3f}s")
        print(f"Hybrid total (approx max proc + max comm): {hybrid_total:.3f}s")

        # For comparison run a sequential tokenization-only baseline
        t0 = time.time()
        for rev,_ in rows:
            WORD_RE.findall((rev or "").lower())
        t1 = time.time()
        seq_time = t1 - t0
        print(f"Sequential baseline (tokenization only): {seq_time:.3f}s")
        speedup = seq_time / hybrid_total if hybrid_total > 0 else float('inf')
        print(f"Hybrid speedup vs sequential (approx): {speedup:.2f}x")

        # Compute sentiment-specific top words:
        # Node 0: positive, Node1: negative. Merge both (they were counted separately).
        print("\nTop 10 words (merged):")
        for w,c in total_counter.most_common(10):
            print(f"{w}: {c}")
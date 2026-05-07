import os
import math
import random
import gc
import time
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# ---------------- vocabulary and model ----------------
DNA_VOCAB = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
VOCAB_SIZE = len(DNA_VOCAB)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):  # 增大 max_len 以支持更长序列
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerRDEModel(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, max_seq_len=1000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                 batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.05)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(self.fc(x))
        x = self.norm(x)
        return x

# ---------------- simple vector DB (占位) ----------------
class SimulatedVectorDB:
    def __init__(self, max_size=20000):
        self.max_size = max_size
    def add(self, window_chars, embedding, metadata):
        pass

# ---------------- helpers ----------------
def tokenize_sequence(seq):
    return [DNA_VOCAB.get(c.upper(), 0) for c in seq]

def build_pos_index(sequences, alphabet=('A', 'C', 'G', 'T')):
    pos_index = []
    for seq in sequences:
        idx = defaultdict(list)
        for i, c in enumerate(seq):
            if c in alphabet:
                idx[c].append(i)
        pos_index.append(idx)
    return pos_index

def next_pos_fast(char, pos_index, seq_idx, current_pos):
    positions = pos_index[seq_idx][char]
    left = 0
    right = len(positions)
    while left < right:
        mid = (left + right) // 2
        if positions[mid] < current_pos:
            left = mid + 1
        else:
            right = mid
    return positions[left] if left < len(positions) else float('inf')

def get_non_dominated_letters(current_t, sequences, pointers, pos_index):
    common_chars = set.intersection(*[set(seq[p:]) for seq, p in zip(sequences, pointers)]) - {'N'}
    nd = set()
    for a in common_chars:
        pa = [next_pos_fast(a, pos_index, i, p) for i, p in enumerate(pointers)]
        if any(p == float('inf') for p in pa):
            continue
        dominated = False
        for b in common_chars:
            if a == b: continue
            pb = [next_pos_fast(b, pos_index, i, p) for i, p in enumerate(pointers)]
            if any(x == float('inf') for x in pb):
                continue
            if all(pb_i <= pa_i for pb_i, pa_i in zip(pb, pa)) and any(pb_i < pa_i for pb_i, pa_i in zip(pb, pa)):
                dominated = True
                break
        if not dominated:
            nd.add(a)
    return nd

def build_suffix_counts(sequences, alphabet=('A', 'C', 'G', 'T')):
    alpha_index = {a: idx for idx, a in enumerate(alphabet)}
    suffix_counts = []
    for seq in sequences:
        L = len(seq)
        arr = np.zeros((L + 1, len(alphabet)), dtype=int)
        for pos in range(L - 1, -1, -1):
            arr[pos] = arr[pos + 1]
            c = seq[pos]
            if c in alpha_index:
                arr[pos, alpha_index[c]] += 1
        suffix_counts.append(arr)
    return suffix_counts

def upper_bound(t, sequences, pointers, suffix_counts=None, alphabet=('A', 'C', 'G', 'T')):
    alpha_index = {a: idx for idx, a in enumerate(alphabet)}
    sum_min_counts = 0
    for a in alphabet:
        j = alpha_index[a]
        min_count = min(suffix_counts[i][pointers[i]][j] if pointers[i] < len(sequences[i]) else 0 
                        for i in range(len(pointers)))
        sum_min_counts += min_count
    return len(t) + sum_min_counts

def assign_rank(values, reverse=False):
    if not values:
        return []
    sorted_unique = sorted(set(values), reverse=reverse)
    rank_dict = {v: rank + 1 for rank, v in enumerate(sorted_unique)}
    return [rank_dict[v] for v in values]

# ---------------- 训练加速：预计算窗口 Counter ----------------
def precompute_all_window_counters(sequences, W=20, stride=40):
    all_counters = []
    metadata = []
    for si, seq in enumerate(sequences):
        for i in range(0, max(1, len(seq) - W + 1), stride):
            wchars = seq[i:i + W]
            if len(wchars) < W:
                wchars += 'N' * (W - len(wchars))
            all_counters.append(Counter(wchars))
            metadata.append((si, i))
    return all_counters, metadata

def find_pos_neg_fast(anchor_counter, all_counters, metadata, anchor_si):
    chars = set(anchor_counter.keys())
    diffs = []
    for idx, counter in enumerate(all_counters):
        si, _ = metadata[idx]
        if si == anchor_si: continue
        diff = sum(abs(anchor_counter[ch] - counter.get(ch, 0)) for ch in chars)
        diffs.append((diff, idx))
    if not diffs: return None, None
    diffs.sort(key=lambda x: x[0])
    return metadata[diffs[0][1]], metadata[diffs[-1][1]]

def train_transformer(model, sequences, W=20, stride=40, epochs=8, lr=1e-3,
                      batch_size=256, device='cpu', margin=0.5):
    start_time = time.time()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.TripletMarginLoss(margin=margin, p=1)

    all_counters, metadata = precompute_all_window_counters(sequences, W=W, stride=stride)
    windows = [(si, start) for si, start in metadata]
    print(f"Total training windows: {len(windows)}")

    for epoch in range(epochs):
        random.shuffle(windows)
        total_loss = 0.0
        cnt = 0
        skipped = 0
        epoch_start = time.time()

        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            anchors, poss, negs = [], [], []

            for si, win_pos in batch:
                w = sequences[si][win_pos:win_pos + W]
                if len(w) < W: w += 'N' * (W - len(w))
                anchor_counter = Counter(w)
                pos_pair, neg_pair = find_pos_neg_fast(anchor_counter, all_counters, metadata, si)
                if pos_pair is None or neg_pair is None:
                    skipped += 1
                    continue
                s_pos, ppos = pos_pair
                s_neg, npos = neg_pair
                P = sequences[s_pos][ppos:ppos + W]
                if len(P) < W: P += 'N' * (W - len(P))
                N = sequences[s_neg][npos:npos + W]
                if len(N) < W: N += 'N' * (W - len(N))
                anchors.append(tokenize_sequence(w))
                poss.append(tokenize_sequence(P))
                negs.append(tokenize_sequence(N))

            if len(anchors) == 0: continue

            A_t = torch.tensor(anchors, dtype=torch.long, device=device)
            P_t = torch.tensor(poss, dtype=torch.long, device=device)
            N_t = torch.tensor(negs, dtype=torch.long, device=device)

            model.train()
            embA = model(A_t)
            embP = model(P_t)
            embN = model(N_t)
            reg_loss = 0.01 * (embA.norm(p=2) + embP.norm(p=2) + embN.norm(p=2))
            loss = loss_fn(embA, embP, embN) + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            cnt += 1

        avg_loss = total_loss / cnt if cnt > 0 else 0
        print(f"[train] epoch {epoch + 1}/{epochs}, avg_loss={avg_loss:.6f}, skipped={skipped}, time={time.time()-epoch_start:.1f}s")

    model.to('cpu')
    print(f"Total training time: {time.time() - start_time:.2f}s")
    return model

# ---------------- 预计算所有嵌入（终极修复版，支持 GPU 无报错）----------------
def precompute_all_embeddings(model, sequences, W=20, device='cpu'):
    model.eval()
    # 确保模型在正确的设备上（训练后可能还在 GPU）
    model = model.to(device)
    
    all_embeddings = {}
    print("正在预计算所有滑动窗口嵌入（stride=1）...")
    
    with torch.no_grad():
        for si, seq in enumerate(sequences):
            embeddings = []
            L = len(seq)
            batch_toks = []
            
            for i in range(max(1, L - W + 1)):
                wchars = seq[i:i + W]
                if len(wchars) < W:
                    wchars += 'N' * (W - len(wchars))
                tokenized = tokenize_sequence(wchars)
                batch_toks.append(tokenized)
                
                # 批次满 256 或到达序列末尾，执行一次 forward
                if len(batch_toks) == 256 or i == L - W:
                    if len(batch_toks) > 0:
                        # 【关键：显式指定 device】
                        batch_tensor = torch.tensor(batch_toks, dtype=torch.long, device=device)
                        # 直接在 device 上计算，只在最后转 CPU 保存
                        embs = model(batch_tensor)              # 保持在 GPU/CPU
                        embeddings.extend(embs.cpu().numpy())   # 转回 CPU 保存为 numpy
                        batch_toks = []  # 清空批次
            
            all_embeddings[si] = embeddings
    
    total_windows = sum(len(v) for v in all_embeddings.values())
    print(f"预计算完成，共 {total_windows} 个窗口嵌入（每个 {embs.shape[1]} 维）")
    return all_embeddings

# ---------------- 优先级计算（使用预计算嵌入）----------------
def compute_priority_for_letters(nd_letters, sequences, pointers, all_embeddings, W=20, pos_index=None):
    priorities = {}
    for ch in nd_letters:
        positions = [next_pos_fast(ch, pos_index, i, p) for i, p in enumerate(pointers)]
        if any(pos == float('inf') for pos in positions):
            priorities[ch] = 0.0
            continue

        embs = []
        for seq_idx, pos in enumerate(positions):
            start = max(0, pos - W // 2)
            if start >= len(all_embeddings[seq_idx]):
                start = len(all_embeddings[seq_idx]) - 1
            embs.append(all_embeddings[seq_idx][start])

        if len(embs) < 2:
            avg_sim = 0.0
        else:
            embs = np.array(embs)
            sims = []
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    d = np.sum(np.abs(embs[i] - embs[j]))
                    sims.append(1.0 / (1.0 + d))
            avg_sim = np.mean(sims) if sims else 0.0

        decays_sum = sum((1.0 - positions[i] / max(1, len(sequences[i]))) for i in range(len(positions)))
        priorities[ch] = avg_sim * decays_sum
    return priorities

# ---------------- 优化版梁搜索 ----------------
def greedy_mlcs_with_rde(sequences, model, all_embeddings, beam_width=15, W=20, max_iterations=600):
    start_time = time.time()
    pos_index = build_pos_index(sequences)
    suffix_counts = build_suffix_counts(sequences)
    pointers = [0] * len(sequences)
    beam = [("", pointers[:], 0.0)]
    best_mlcs = ""
    best_length = 0
    iteration = 0
    top_k_priority = 10

    print("\n=== Optimized Beam Search Start (beam=15) ===")
    while beam and iteration < max_iterations:
        beam.sort(key=lambda x: (-len(x[0]), -upper_bound(x[0], sequences, x[1], suffix_counts)))

        candidates = []
        for beam_idx, (mlcs, pointers, _score) in enumerate(beam):
            nd_letters = get_non_dominated_letters(mlcs, sequences, pointers, pos_index)
            if not nd_letters: continue

            if beam_idx < top_k_priority:
                priorities = compute_priority_for_letters(nd_letters, sequences, pointers, all_embeddings,
                                                         W=W, pos_index=pos_index)
            else:
                priorities = {ch: 0.0 for ch in nd_letters}

            for ch in sorted(nd_letters):
                pa = [next_pos_fast(ch, pos_index, i, pointers[i]) for i in range(len(sequences))]
                if any(p == float('inf') for p in pa): continue
                new_ptrs = [p + 1 for p in pa]
                new_mlcs = mlcs + ch
                new_ub = upper_bound(new_mlcs, sequences, new_ptrs, suffix_counts=suffix_counts)
                if new_ub <= best_length: continue
                lb_val = int(max(pa) - min(pa)) if all(p != float('inf') for p in pa) else 0
                pr_val = priorities.get(ch, 0.0)

                candidates.append({
                    'mlcs': new_mlcs, 'pointers': new_ptrs[:], 'ub': new_ub,
                    'lb': lb_val, 'pr': pr_val
                })

        if not candidates: break

        ub_vals = [c['ub'] for c in candidates]
        ub_ranks = assign_rank(ub_vals, reverse=True)
        lb_vals = [c['lb'] for c in candidates]
        lb_ranks = assign_rank(lb_vals, reverse=False)
        pr_vals = [c['pr'] for c in candidates]
        pr_ranks = assign_rank(pr_vals, reverse=True)

        for i, c in enumerate(candidates):
            score = 0.4 / (ub_ranks[i] + 1e-9) + 0.4 / (lb_ranks[i] + 1e-9) + 0.2 / (pr_ranks[i] + 1e-9)
            c['score'] = score

        candidates.sort(key=lambda x: x['score'], reverse=True)
        kept = candidates[:beam_width]

        for c in kept:
            if len(c['mlcs']) > best_length:
                best_length = len(c['mlcs'])
                best_mlcs = c['mlcs']
                print(f"  -> UPDATED best MLCS length {best_length}")

        beam = [(c['mlcs'], c['pointers'], c['score']) for c in kept]
        iteration += 1
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, beam size {len(beam)}, best length {best_length}")
        gc.collect()

    print(f"\n=== Finished: best MLCS length={best_length} ===")
    print(f"Search time: {time.time() - start_time:.2f}s")
    return best_mlcs, best_length, upper_bound(best_mlcs, sequences, pointers, suffix_counts=suffix_counts)

# ---------------- 读取单文件 FASTA ----------------
def read_sequences_from_file(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    current_seq = ""
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if current_seq:
                sequences.append(current_seq.upper())
                current_seq = ""
        else:
            current_seq += line
    if current_seq:
        sequences.append(current_seq.upper())
    return sequences

# ---------------- main experiment ----------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    NUM_GROUPS = 5
    NUM_SEQS = [75]
    SEQ_LENGTHS = 1000
    BEAM_WIDTH = 15
    W = 20
    stride = 40
    MAX_ITER = 600

    results = []  # 全局结果列表

    print("=== 开始运行 RDE-guided MLCS  ===")

    for SEQ_LEN in NUM_SEQS:
        results.clear()  # 每个长度重新开始

        for gid in range(1, NUM_GROUPS + 1):
            file_path = f"random固定长度1000×{SEQ_LEN}_group{gid}.fasta"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"数据集不存在: {file_path}，请先生成数据集文件")

            print("\n" + "=" * 80)
            print(f"正在处理长度 {SEQ_LEN} 的第 {gid}/{NUM_GROUPS} 组：{file_path}")
            print("=" * 80)

            group_start_time = time.time()

            seqs = read_sequences_from_file(file_path)
            print(f"序列加载完成，共 {len(seqs)} 条，平均长度 ≈ {SEQ_LEN}")

            model = TransformerRDEModel(vocab_size=VOCAB_SIZE, d_model=32, nhead=2, num_layers=1,
                                       dim_feedforward=64, max_seq_len=1000)

            # 训练（使用更大 stride 加速）
            model = train_transformer(model, seqs, W=W, stride=stride, epochs=8, lr=1e-3,
                                      batch_size=256, device=device, margin=0.5)

            # 预计算所有嵌入
            all_embeddings = precompute_all_embeddings(model, seqs, W=W, device=device)

            # 搜索
            res_mlcs, res_len, res_ub = greedy_mlcs_with_rde(seqs, model, all_embeddings,
                                                            beam_width=BEAM_WIDTH,
                                                            W=W, max_iterations=MAX_ITER)

            total_time = time.time() - group_start_time

            results.append({
                'group': gid,
                'mlcs_length': res_len,
                'mlcs': res_mlcs,
                'upper_bound': res_ub,
                'total_time': total_time,
                'seq_length': SEQ_LEN
            })

            print(f"第 {gid} 组运行完成！")
            print(f"→ MLCS 长度: {res_len} (Upper Bound: {res_ub})")
            print(f"→ 本组完整运行时间: {total_time:.2f} 秒\n")

        # 当前长度统计
        avg_length = np.mean([r['mlcs_length'] for r in results])
        avg_time = np.mean([r['total_time'] for r in results])

        summary_file = f"RDEBS_1000x{SEQ_LEN}.txt"
        with open(summary_file, "w") as f:
            f.write(f"RDE-guided MLCS Results ({NUM_GROUPS} groups) for Length {SEQ_LEN}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Instances: {NUM_SEQS} × ~{SEQ_LEN}, random ACGT\n")
            f.write(f"Beam width: {BEAM_WIDTH}, Window: {W}, Stride: {stride}\n")
            f.write(f"Average MLCS Length: {avg_length:.2f}\n")
            f.write(f"Average Time: {avg_time:.2f} s\n")
            f.write("=" * 50 + "\n\n")
            for r in results:
                f.write(f"Group {r['group']}:\n")
                f.write(f"  Length     : {r['mlcs_length']}\n")
                f.write(f"  Time (s)   : {r['total_time']:.2f}\n")
                f.write(f"  Upper Bound: {r['upper_bound']}\n")
                f.write(f"  MLCS       : {r['mlcs']}\n\n")

        print(f"\n长度 {SEQ_LEN} 的结果已保存至：{summary_file}")
        print(f"平均 MLCS 长度: {avg_length:.2f}，平均时间: {avg_time:.2f} 秒")

    print("\n=== 所有序列长度实验完成！===\n")

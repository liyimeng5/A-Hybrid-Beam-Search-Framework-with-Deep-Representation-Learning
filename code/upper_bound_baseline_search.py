import os
import time
import numpy as np
from collections import defaultdict
from datetime import datetime


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

def upper_bound(t, sequences, pointers, suffix_counts, alphabet=('A', 'C', 'G', 'T')):
    alpha_index = {a: idx for idx, a in enumerate(alphabet)}
    sum_min_counts = 0
    for a in alphabet:
        j = alpha_index[a]
        min_count = min(suffix_counts[i][pointers[i]][j] if pointers[i] < len(sequences[i]) else 0 
                        for i in range(len(pointers)))
        sum_min_counts += min_count
    return len(t) + sum_min_counts

# ---------------- 标准 UB-only 基线----------------
def greedy_mlcs_ub_only(sequences, beam_width=15, max_iterations=600):
    start_time = time.time()
    pos_index = build_pos_index(sequences)
    suffix_counts = build_suffix_counts(sequences)
    pointers = [0] * len(sequences)
    beam = [("", pointers[:], 0.0)]
    best_mlcs = ""
    best_length = 0
    iteration = 0

    print(f"\n=== UB-only Baseline Start (beam={beam_width}, alphabetical order extension) ===")
    while beam and iteration < max_iterations:
        candidates = []
        for _, (mlcs, pointers, _) in enumerate(beam):
            nd_letters = get_non_dominated_letters(mlcs, sequences, pointers, pos_index)
            if not nd_letters:
                continue

            # 按字母顺序扩展（A -> C -> G -> T），无智能引导
            for ch in sorted(nd_letters):
                pa = [next_pos_fast(ch, pos_index, i, pointers[i]) for i in range(len(sequences))]
                if any(p == float('inf') for p in pa):
                    continue
                new_ptrs = [p + 1 for p in pa]
                new_mlcs = mlcs + ch
                new_ub = upper_bound(new_mlcs, sequences, new_ptrs, suffix_counts)
                if new_ub <= best_length:  # 只用 UB 剪枝
                    continue

                candidates.append({
                    'mlcs': new_mlcs,
                    'pointers': new_ptrs[:],
                    'ub': new_ub,
                    'ch': ch
                })

        if not candidates:
            break

        # 关键：不使用任何启发式排序，只按字母顺序保留 beam
        candidates.sort(key=lambda x: x['ch'])  # 确定性扩展顺序
        kept = candidates[:beam_width]

        for c in kept:
            if len(c['mlcs']) > best_length:
                best_length = len(c['mlcs'])
                best_mlcs = c['mlcs']
                print(f"  -> UPDATED best MLCS length {best_length}")

        beam = [(c['mlcs'], c['pointers'], 0.0) for c in kept]
        iteration += 1
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, beam size {len(beam)}, current best {best_length}")

    final_ub = upper_bound(best_mlcs, sequences, pointers, suffix_counts) if best_mlcs else 0
    print(f"\n=== UB-only Baseline Finished: length={best_length}, final UB={final_ub} ===")
    print(f"Search time: {time.time() - start_time:.2f}s")
    return best_mlcs, best_length, final_ub


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

if __name__ == "__main__":
    NUM_GROUPS = 5
    NUM_SEQS = [15,25,35,45,55,65,75]
    SEQ_LENGTHS = 1000
    BEAM_WIDTH = 15
    MAX_ITER = 600

    print("=== 只运行 UB-only 基线算法（用于与你的 RDEBS 对比）===")

    # 外层循环：遍历每个序列长度
    for SEQ_LEN in NUM_SEQS:
        results = []  # 为当前长度新建结果列表

        print(f"\n{'='*80}")
        print(f"正在处理序列长度: {SEQ_LEN}（共 {NUM_GROUPS} 组）")
        print(f"{'='*80}")

        # 内层循环：处理该长度的 5 个 group
        for gid in range(1, NUM_GROUPS + 1):
            file_path = f"random固定长度1000×{SEQ_LEN}_group{gid}.fasta"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"数据集不存在: {file_path}")

            print(f"\n正在处理 Group {gid}/5: {file_path}")

            group_start_time = time.time()

            seqs = read_sequences_from_file(file_path)
            print(f"加载完成，共 {len(seqs)} 条序列")

            _, mlcs_len, mlcs_ub = greedy_mlcs_ub_only(seqs, beam_width=BEAM_WIDTH, max_iterations=MAX_ITER)

            total_time = time.time() - group_start_time

            results.append({
                'group': gid,
                'seq_len': SEQ_LEN,
                'mlcs_length': mlcs_len,
                'upper_bound': mlcs_ub,
                'total_time': total_time
            })

            print(f"Group {gid} 完成！MLCS 长度: {mlcs_len} (UB: {mlcs_ub}), 用时: {total_time:.2f}s")

        # ---------------- 当前长度结果汇总 ----------------
        avg_length = np.mean([r['mlcs_length'] for r in results])
        avg_time = np.mean([r['total_time'] for r in results])

        summary_file = f"UB_1000x{SEQ_LEN}.txt"
        with open(summary_file, "w") as f:
            f.write(f"UB-only Baseline Results for Length {SEQ_LEN} ({NUM_GROUPS} groups)\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Instances: {NUM_SEQS} sequences × ~{SEQ_LEN} bp, random ACGT\n")
            f.write(f"Beam width: {BEAM_WIDTH}\n")
            f.write(f"Average MLCS Length: {avg_length:.2f}\n")
            f.write(f"Average Time: {avg_time:.2f} s\n")
            f.write("=" * 50 + "\n\n")
            for r in results:
                f.write(f"Group {r['group']}:\n")
                f.write(f"  Length     : {r['mlcs_length']}\n")
                f.write(f"  Upper Bound: {r['upper_bound']}\n")
                f.write(f"  Time (s)   : {r['total_time']:.2f}\n\n")

        print(f"\n>>> 长度 {SEQ_LEN} 完成！平均 MLCS 长度: {avg_length:.2f}")
        print(f"结果已保存至：{summary_file}")

    print("\n=== 所有序列长度基线实验完成！===\n")

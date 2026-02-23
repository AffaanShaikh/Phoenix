# save as analyze_cer_by_length.py and run
import json, numpy as np, collections
fn = r"C:\Arbeit\Phoenix\LT-seq2seq\experiments\char_lstm_bi_eng2deu_20260223_005803\predictions.jsonl"
bins = [(0,10),(11,20),(21,30),(31,50),(51,999)]
agg = {b:[] for b in bins}

def levenshtein(a,b):
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): dp[i][0]=i
    for j in range(lb+1): dp[0][j]=j
    for i in range(1,la+1):
        for j in range(1,lb+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[la][lb]

with open(fn,"r",encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        ref = obj["reference"]
        hyp = obj["prediction"]
        cer = levenshtein(ref,hyp)/max(1,len(ref))
        L = len(ref)
        for b in bins:
            if b[0] <= L <= b[1]:
                agg[b].append(cer)
                break

for b in bins:
    vals = agg[b]
    print(f"len {b}: n={len(vals)} meanCER={np.mean(vals) if vals else None}")

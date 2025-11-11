import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
from collections import Counter


def jiexi(x):
    true_in_x = 'true' in x.lower()
    false_in_x = 'false' in x.lower()
    if true_in_x and false_in_x:
        return 'ambiguous'
    if (not true_in_x) and (not false_in_x):
        return 'none'
    if true_in_x:
        return 'True'
    if false_in_x:
        return 'False'


def _agg_by_sample(g):
    y_prob = np.mean(g['prediction'].astype(float))
    n = len(g)
    label = g['label'].iloc[0]
    return pd.Series({'y_prob': y_prob, 'n': n, 'label': label})


def analyze_samples(data_path, threshold=0.5, find_best_threshold=True):
    df = pd.read_csv(data_path)

    df['generation'] = df['generation'].astype(str)
    df['prediction'] = df['generation'].apply(lambda x: jiexi(x))

    df_valid = df[(df['prediction'] != 'none') & (df['prediction'] != 'ambiguous')]
    df_valid['prediction'] = df_valid['prediction'].map({'True': True, 'False': False})

    df_valid_agg = df_valid.groupby('original_tag').apply(_agg_by_sample).reset_index()
    df_valid_agg['y_pred'] = (df_valid_agg['y_prob'] >= threshold).astype(bool)

    y_prob = np.array(df_valid_agg['y_prob'].tolist())
    y_true = np.array(df_valid_agg['label'].tolist())
    y_pred = np.array(df_valid_agg['y_pred'].tolist())

    metrics = {
        "threshold": threshold,
        "Valid Size": len(df_valid),
        "Sample Size": len(df_valid_agg),
        "F1 Score": f1_score(y_true, y_pred, average='binary', pos_label=1),
        "Recall": recall_score(y_true, y_pred, average='binary', pos_label=1),
        "Precision": precision_score(y_true, y_pred, average='binary', pos_label=1),
        "Accuracy": accuracy_score(y_true, y_pred),
        "roc_AUC": roc_auc_score(y_true, y_prob),
        "pr_AUC": average_precision_score(y_true, y_prob),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "support_pos": tp + fn,
        "support_neg": tn + fp,
    })

    if find_best_threshold:
        candidate = np.unique(np.concatenate([[0.0, 1.0], y_prob]))
        print(candidate)
        print(f"Finding best threshold from {len(candidate)} candidates...")

        # 按 F1
        best_f1, best_t_f1, best_stat_f1 = -1.0, None, None
        # 按 Youden’s J（敏感度 + 特异度 - 1）
        best_j, best_t_j, best_stat_j = -1.0, None, None

        # 预先算特异度时需要 TN / (TN+FP)
        neg_mask = (y_true == 0)
        P = max((y_true == 1).sum(), 1)
        N = max(neg_mask.sum(), 1)

        for t in candidate:
            pred = (y_prob >= t).astype(int)
            tp_ = int(((pred == 1) & (y_true == 1)).sum())
            tn_ = int(((pred == 0) & (y_true == 0)).sum())
            fp_ = int(((pred == 1) & (y_true == 0)).sum())
            fn_ = int(((pred == 0) & (y_true == 1)).sum())
            rec_ = tp_ / P
            spe_ = tn_ / N
            f1_ = f1_score(y_true, pred, zero_division=0)
            J_ = rec_ + spe_ - 1

            if f1_ > best_f1:
                best_f1, best_t_f1, best_stat_f1 = f1_, float(t), {
                    "acc": accuracy_score(y_true, pred),
                    "prec": precision_score(y_true, pred, zero_division=0),
                    "rec": rec_,
                    "spe": spe_,
                    "f1": f1_,
                }
            if J_ > best_j:
                best_j, best_t_j, best_stat_j = J_, float(t), {
                    "acc": accuracy_score(y_true, pred),
                    "prec": precision_score(y_true, pred, zero_division=0),
                    "rec": rec_,
                    "spe": spe_,
                    "f1": f1_,
                    "youdenJ": J_,
                }

        bests = {
            "best_threshold_f1": best_t_f1,
            "best_stats_f1": best_stat_f1,
            "best_threshold_youdenJ": best_t_j,
            "best_stats_youdenJ": best_stat_j,
        }
        metrics.update(bests)

    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics


data_path = '/data/bwyin/project/MultiviewLLM/evaluation_results/projector_g8_t8_match_12mo_fixed_final_step18068_dgFalse_dtFalse_n5.csv'
m = analyze_samples(data_path, threshold=0.5, find_best_threshold=True)
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, brier_score_loss,
    roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
)
from hmeasure import h_score
import json
from sklearn.linear_model import LogisticRegression


def calculate_confidence(logit_data, T):
    y_true = []
    y_prob = []
    for i in range(len(logit_data)):
        # 过滤多答案情况
        if len(logit_data[i]['hit_steps'])!=1:
            continue
        # 真实标签
        y_true.append(logit_data[i]['label'])
        # 计算置信度
        true_logits = logit_data[i]['hit_info'][0]['logit_true']
        false_logits = logit_data[i]['hit_info'][0]['logit_false']
        conf_true = np.exp(true_logits / T) / (np.exp(true_logits / T) + np.exp(false_logits / T))
        y_prob.append(conf_true)
    return y_true, y_prob


def evaluate_credit_scoring_metrics(y_true, y_pred, pos_rate=None, threshold=None, b_pg=0.4, severity_ratio=1.0):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)

    # -------- 1. AUC（区分能力） ---------
    auc = roc_auc_score(y_true, y_pred)

    # -------- 2. PCC（分类正确率）--------
    # 论文做法：阈值 τ 使得 “预测为正的比例 = 训练集正类比例”
    if pos_rate is None:
        pos_rate = y_true.mean()
    # 计算阈值
    if threshold is None:
        threshold = np.quantile(y_pred, 1.0 - pos_rate) - 1e-8
    y_hat = (y_pred > threshold).astype(int)
    pcc = accuracy_score(y_true, y_hat)

    # -------- 3. BS（Brier Score，概率校准）--------
    bs = brier_score_loss(y_true, y_pred)

    # -------- 4. KS（Kolmogorov–Smirnov）--------
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ks = float(np.max(tpr - fpr))

    # -------- 5. PG（Partial Gini）--------
    # 在 y_pred <= b_pg 的子样本中计算 Gini = 2 * AUC_sub - 1
    mask = y_pred <= b_pg
    if mask.sum() > 0 and len(np.unique(y_true[mask])) == 2:
        auc_sub = roc_auc_score(y_true[mask], y_pred[mask])
        pg = 2.0 * auc_sub - 1.0
    else:
        pg = np.nan

    # -------- 6. H-measure --------
    h = h_score(y_true, y_pred, severity_ratio=severity_ratio)

    # -------- 7. Recall, Precision, F1 --------
    recall_pos = recall_score(y_true, y_hat)
    precision_pos = precision_score(y_true, y_hat)
    f1_pos = f1_score(y_true, y_hat)

    # -------- 汇总结果 --------
    meta_info = {"threshold": threshold, "pos_rate": pos_rate, "AUC_sub": auc_sub if 'auc_sub' in locals() else np.nan,
                 "b_pg": b_pg, "severity_ratio": severity_ratio}
    metrics = {"AUC": auc, "PCC": pcc, "BS": bs, "KS": ks, "PG": pg, "H-measure": h,
               "Recall_Pos": recall_pos, "Precision_Pos": precision_pos, "F1_Pos": f1_pos,
               'total_samples': len(y_true)}
    return metrics, meta_info


def get_original_index(df_pc):
    '''To retrieve the original sample index from PromptCast output dataframe.'''
    sample_index = pd.read_feather(Path(paths.processed_data_dir, 'sample_index', 'samples_min12mo_fixed_2test.feather'))
    sample_index['index'] = sample_index.index

    df_pc['pc_tag'] = df_pc.apply(lambda row: f"{row['act_idn_sky']}_{row['billing_dates'][0]}", axis=1)
    sample_index['pc_tag'] = sample_index.apply(lambda row: f"{row['act_idn_sky']}_{row['billing_dates'][0]}", axis=1)

    pc_tag_to_index = dict(zip(sample_index['pc_tag'], sample_index['index']))

    df_pc['original_tag'] = df_pc['pc_tag'].map(pc_tag_to_index)

    df_pc = df_pc.drop(columns=['pc_tag'])
    return df_pc


def logit_jiexi(logit_data):
    '''Extract predicted label from logit data.'''
    y_true = []
    y_prob = []
    for i in range(len(logit_data)):
        if 'false' in logit_data[i]['generation']:
            y_prob.append(0.0)
        elif 'true' in logit_data[i]['generation']:
            y_prob.append(1.0)
        else:
            continue
        y_true.append(logit_data[i]['label'])
    return y_true, y_prob


if __name__ == '__main__':
    from pathlib import Path
    from src.config.paths import paths

    T = 2.0
    pos_rate = None
    bg_pg = 0.4
    severity_ratio = 1.0

    # prof. zhu's promptcast
    df_promptcast = pd.read_feather(r'/home/bwyin/project/Agent/MultiviewLLM/src/temp/preds_promptcast_summary_1_20251204_181833.feather')
    df_promptcast = get_original_index(df_promptcast)
    df_promptcast = df_promptcast[df_promptcast['pred_is_delinquent'].notnull()]
    y_true, y_prob = df_promptcast['target_delinquency'], df_promptcast['pred_is_delinquent']
    metrics, meta_info = evaluate_credit_scoring_metrics(y_true, y_prob, pos_rate, 0.5, bg_pg, severity_ratio)
    promptcast_metrics = metrics
    promptcast_metrics['Model'] = 'PromptCast'
    valid_origins = df_promptcast['original_tag'].tolist()

    # xgbost结果
    df_xgboost = pd.read_csv(r'/home/bwyin/project/Agent/MultiviewLLM/src/temp/predictions_xgboost_latest.csv')
    df_xgboost['original_tag'] = df_xgboost.index
    df_xgboost = df_xgboost[df_xgboost['split']=='test']
    df_xgboost = df_xgboost[df_xgboost['original_tag'].isin(valid_origins)]
    y_true, y_prob = df_xgboost['target_delinquency'], df_xgboost['pred_prob']
    metrics, meta_info = evaluate_credit_scoring_metrics(y_true, y_prob, pos_rate, None, bg_pg, severity_ratio)
    xgboost_metrics = metrics
    xgboost_metrics['Model'] = 'XGBoost'

    # logistic回归结果
    df_logistic = pd.read_csv(r'/home/bwyin/project/Agent/MultiviewLLM/src/temp/predictions_logistic_regression_latest.csv')
    df_logistic['original_tag'] = df_logistic.index
    df_logistic = df_logistic[df_logistic['split']=='test']
    df_logistic = df_logistic[df_logistic['original_tag'].isin(valid_origins)]
    y_true, y_prob = df_logistic['target_delinquency'], df_logistic['pred_prob']
    metrics, meta_info = evaluate_credit_scoring_metrics(y_true, y_prob, pos_rate, None, bg_pg, severity_ratio)
    logistic_metrics = metrics
    logistic_metrics['Model'] = 'Logistic Regression'

    # 读取 Logit 数据
    with open(r'/data/bwyin/project/MultiviewLLM/evaluation_results/projector_g5_t1_match_12mo_fixed_exp2_final_step1076_logit.json', 'r') as f:
        logit_data = json.load(f)
    # logit_data = [item for item in logit_data if item['original_tag'] in valid_origins]
    y_true, y_prob = calculate_confidence(logit_data, T)
    metrics, meta_info = evaluate_credit_scoring_metrics(y_true, y_prob, pos_rate, None, bg_pg, severity_ratio)
    # y_true, y_prob = logit_jiexi(logit_data)
    # metrics, meta_info = evaluate_credit_scoring_metrics(y_true, y_prob, pos_rate, 0.5, bg_pg, severity_ratio)  # 记得有时候去掉阈值
    metrics['Model'] = 'MVLLM Logit-based'

    # 汇总结果
    df = pd.DataFrame([
        promptcast_metrics,
        xgboost_metrics,
        logistic_metrics,
        metrics
    ])
    df = df.round(4)
    print(df)
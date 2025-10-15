'''
This module provides evaluation utilities for graph-based models using SVM classifiers.

It is copied from the GCL library (CL.eval) to tackle unbalanced datasets effectively.

Includes:
- Data conversion functions to prepare data for SVM.
- Predefined data splitting for cross-validation.
- SVMEvaluator class for training and evaluating SVM classifiers with hyperparameter tuning.
'''

from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV


def split_to_numpy(x, y, split):
    '''
    Convert PyTorch tensors to NumPy arrays based on predefined splits.
    '''
    keys = ['train', 'test', 'valid']
    objs = [x, y]
    return [obj[split[key]].detach().cpu().numpy() for obj in objs for key in keys]


def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    '''
    Create a PredefinedSplit for cross-validation using training and validation sets.
    '''
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)])
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps


class SVMEvaluator():
    """
    针对不平衡二分类：
      - LinearSVC: 使用 class_weight='balanced'；
      - SVC: 使用 class_weight='balanced' + probability=True（用于阈值调优与AUC/PR-AUC）；
      - 寻参评分使用 F1（二分类，正类=1）；
      - 在验证集上搜索最佳阈值（使F1最大），再用于测试集评估。
    """

    def __init__(self, linear=True, params=None):
        self.linear = linear
        if linear:
            # 线性快速版（大样本/高维）——无概率
            self.evaluator = LinearSVC(class_weight='balanced', max_iter=5000)
            # 仅对 C 网格；如需调 loss/penalty/dual，可自行扩展
            default_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        else:
            # 非线性版（小样本/需要概率）
            self.evaluator = SVC(class_weight='balanced', probability=True)  # 支持 predict_proba
            # 同时支持 linear/rbf；若只需 rbf，可移除 'linear'
            default_params = {
                'kernel': ['rbf', 'linear'],
                'C': [0.01, 0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 1e-4, 1e-3, 1e-2, 1e-1]  # 仅在 rbf 时有效，sklearn会自动忽略无效组合
            }
        self.params = default_params if params is None else params

    @staticmethod
    def _best_threshold_from_val(y_true, scores):
        """
        在验证集上通过 PR 曲线搜索使 F1 最大的阈值
        """
        prec, rec, thr = precision_recall_curve(y_true, scores)
        # 注意：precision_recall_curve 返回的 thr 与 prec/rec 长度不同
        f1 = (2 * prec * rec) / np.clip(prec + rec, 1e-12, None)
        idx = np.argmax(f1)
        # 当 idx==0 时没有阈值（对应全为正/负的极端点），退回 0
        if idx == 0 or idx - 1 >= len(thr):
            return 0.0
        return float(thr[idx - 1])

    def evaluate(self, x, y, split):
        # 拆分
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)

        # 预定义 train/val 给 GridSearchCV
        ps, [x_train_val, y_train_val] = get_predefined_split(x_train, x_val, y_train, y_val)

        # 用 F1 做寻参（不平衡更合适）
        classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring='f1', verbose=0)
        classifier.fit(x_train_val, y_train_val)

        # 拿到最优模型
        best_clf = classifier.best_estimator_

        # —— 阈值调优（在验证集）——
        if self.linear:
            # LinearSVC 没有概率，用 decision_function 当分数
            val_scores = best_clf.decision_function(x_val)
        else:
            # SVC 支持概率
            val_scores = best_clf.predict_proba(x_val)[:, 1]
        best_thr = self._best_threshold_from_val(y_val, val_scores)

        # —— 在测试集评估 ——（用上面学到的最优阈值）
        if self.linear:
            test_scores = best_clf.decision_function(x_test)
        else:
            test_scores = best_clf.predict_proba(x_test)[:, 1]

        y_pred = (test_scores >= best_thr).astype(int)

        # 指标：更关注不平衡任务常用的 F1/PR-AUC/ROC-AUC
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        pos_f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        # AUC 指标使用分数而非离散预测
        try:
            rocauc = roc_auc_score(y_test, test_scores)
        except ValueError:
            rocauc = float('nan')
        try:
            prauc = average_precision_score(y_test, test_scores)
        except ValueError:
            prauc = float('nan')

        return {
            'best_params': classifier.best_params_,
            'val_best_threshold': best_thr,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'f1_pos': pos_f1,
            'precision': prec,
            'recall': rec,
            'roc_auc': rocauc,
            'pr_auc': prauc,
        }

# def evaluate(self, x, y, split):
#     x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
#     ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
#     classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring='accuracy', verbose=0)
#     classifier.fit(x_train, y_train)
#     test_macro = f1_score(y_test, classifier.predict(x_test), average='macro')
#     test_micro = f1_score(y_test, classifier.predict(x_test), average='micro')
#
#     return {
#         'micro_f1': test_micro,
#         'macro_f1': test_macro,
#     }

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve,
                           balanced_accuracy_score, cohen_kappa_score)

class ObesityPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.best_model = None
        
    def prepare_data(self, df):
        X = df.drop('Class', axis=1)
        y = df['Class'].astype(int)
        
        # 使用分层抽样进行训练集和测试集划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 计算类别权重
        unique_classes = np.unique(y_train)
        class_weights = {}
        n_samples = len(y_train)
        n_classes = len(unique_classes)
        
        for cls in unique_classes:
            class_weights[cls] = n_samples / (n_classes * np.sum(y_train == cls))
        
        # 标准化特征（X已经在load_and_preprocess_data中标准化过了）
        X_train_scaled = X_train
        X_test_scaled = X_test
        
        return X_train_scaled, X_test_scaled, y_train, y_test, class_weights
    
    def plot_learning_curve(self, X, y):
        """绘制学习曲线"""
        train_sizes, train_scores, test_scores = learning_curve(
            self.best_model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制美化的混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_roc_curves(self, X_test, y_test):
        """绘制ROC曲线"""
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        plt.figure(figsize=(10, 8))
        
        # 处理多分类情况
        n_classes = len(np.unique(y_test))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    def plot_precision_recall_curves(self, X_test, y_test):
        """绘制精确率-召回率曲线"""
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        plt.figure(figsize=(10, 8))
        
        # 处理多分类情况
        n_classes = len(np.unique(y_test))
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_test == i, y_pred_proba[:, i])
            plt.plot(recall, precision, label=f'Class {i}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()
    
    def plot_feature_importance(self, feature_names):
        """绘制特征重要性"""
        importance = self.best_model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def optimize_hyperparameters(self, X_train, y_train):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'objective': 'multi:softproba',
                'num_class': len(np.unique(y_train)),
                'tree_method': 'hist',
                'n_jobs': -1
            }
            
            model = xgb.XGBClassifier(**params)
            score = cross_val_score(
                model, X_train, y_train, 
                cv=3, 
                scoring='f1_weighted',
                n_jobs=-1
            ).mean()
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params
    
    def train(self, X_train, y_train):
        # 获取最优参数
        best_params = self.optimize_hyperparameters(X_train, y_train)
        
        # 添加早停参数
        best_params.update({
            'early_stopping_rounds': 10,
            'tree_method': 'hist',
            'n_jobs': -1
        })
        
        # 训练最终模型
        self.best_model = xgb.XGBClassifier(**best_params)
        self.best_model.fit(
            X_train, 
            y_train,
            eval_set=[(X_train, y_train)],
            verbose=0
        )
    
    def evaluate(self, X_test, y_test, feature_names=None):
        """增强的评估函数"""
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        # 打印更全面的评估指标
        print("\n=== 模型评估报告 ===")
        print("\n1. 分类报告:")
        print(classification_report(y_test, y_pred))
        
        print("\n2. 平衡准确率:", balanced_accuracy_score(y_test, y_pred))
        print("3. Cohen's Kappa:", cohen_kappa_score(y_test, y_pred))
        
        # 每个类别的详细评估
        classes = np.unique(y_test)
        for cls in classes:
            print(f"\n=== 类别 {cls} 的详细评估 ===")
            cls_mask = y_test == cls
            cls_pred_proba = y_pred_proba[:, cls]
            
            # 计算该类别的AUC
            fpr, tpr, _ = roc_curve(cls_mask, cls_pred_proba)
            print(f"AUC: {auc(fpr, tpr):.3f}")
            
            # 计算该类别的平均精确率
            precision, recall, _ = precision_recall_curve(cls_mask, cls_pred_proba)
            print(f"平均精确率: {np.mean(precision):.3f}")
    
    
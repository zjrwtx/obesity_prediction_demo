import cupy as cp
import cudf
from cuml.ensemble import RandomForestClassifier
from cuml.svm import SVC
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.metrics import classification_report
import optuna

class ObesityPredictorGPU:
    def __init__(self):
        self.models = {
            'xgb': None,
            'lgbm': None,
            'rf': None
        }
        self.best_model = None
        
    def prepare_data(self, df):
        # 转换为GPU数据格式
        df_gpu = cudf.DataFrame.from_pandas(df)
        X = df_gpu.drop('Class', axis=1)
        y = df_gpu['Class']
        
        # 使用GPU进行训练测试集划分
        from cuml.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def optimize_xgb(self, X_train, y_train):
        def objective(trial):
            params = {
                'tree_method': 'gpu_hist',  # 使用GPU
                'predictor': 'gpu_predictor',
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'multi:softproba',
                'num_class': len(cp.unique(y_train.values))
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train,
                     eval_set=[(X_train, y_train)],
                     early_stopping_rounds=10,
                     verbose=False)
            
            return model.best_score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        return study.best_params
    
    def optimize_lgbm(self, X_train, y_train):
        def objective(trial):
            params = {
                'device': 'gpu',  # 使用GPU
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'objective': 'multiclass'
            }
            
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)
            return model.score(X_train, y_train)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        return study.best_params
    
    def train(self, X_train, y_train):
        # XGBoost with GPU
        xgb_params = self.optimize_xgb(X_train, y_train)
        xgb_params.update({
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor'
        })
        self.models['xgb'] = xgb.XGBClassifier(**xgb_params)
        
        # LightGBM with GPU
        lgbm_params = self.optimize_lgbm(X_train, y_train)
        lgbm_params['device'] = 'gpu'
        self.models['lgbm'] = LGBMClassifier(**lgbm_params)
        
        # Random Forest with GPU
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_streams=4  # 使用多个CUDA流
        )
        
        # 训练所有模型
        for name, model in self.models.items():
            print(f"\n训练 {name} 模型...")
            model.fit(X_train, y_train)
        
        # 选择最佳模型
        best_score = 0
        for name, model in self.models.items():
            score = model.score(X_train, y_train)
            if score > best_score:
                best_score = score
                self.best_model = model 
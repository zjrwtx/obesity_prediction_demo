from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
class ObesityEnsemble:
    def __init__(self):
        self.models = {
            'xgb': xgb.XGBClassifier(),
            'lgbm': LGBMClassifier(),
            'catboost': CatBoostClassifier(verbose=0)
        }
        self.ensemble = None
        
    def train(self, X_train, y_train):
        # 为每个基础模型找到最佳参数
        for name, model in self.models.items():
            print(f"\n训练 {name} 模型...")
            if name == 'xgb':
                params = self.optimize_xgb(X_train, y_train)
            elif name == 'lgbm':
                params = self.optimize_lgbm(X_train, y_train)
            else:
                params = self.optimize_catboost(X_train, y_train)
            
            self.models[name].set_params(**params)
            
        # 创建投票分类器
        self.ensemble = VotingClassifier(
            estimators=[
                (name, model) for name, model in self.models.items()
            ],
            voting='soft'  # 使用概率投票
        )
        
        # 训练集成模型
        self.ensemble.fit(X_train, y_train) 
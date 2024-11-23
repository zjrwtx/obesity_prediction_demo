class Config:
    # GPU完整训练配置
    GPU_FULL_CONFIG = {
        'n_trials': 100,           # Optuna试验次数
        'cv_folds': 5,            # 交叉验证折数
        'batch_size': 2048,       # 批次大小
        'sample_size': None,      # 使用全部数据
        'n_estimators': 2000,     # 树的数量
        'early_stopping': 50,     # 早停轮数
        'max_depth': 12,          # 最大树深度
        'learning_rate_range': (0.001, 0.3)  # 学习率范围
    }
    
    # GPU快速训练配置
    GPU_FAST_CONFIG = {
        'n_trials': 30,
        'cv_folds': 3,
        'batch_size': 1024,
        'sample_size': 1000,
        'n_estimators': 500,
        'early_stopping': 10,
        'max_depth': 8,
        'learning_rate_range': (0.01, 0.3)
    }
    
    # CPU完整训练配置
    CPU_FULL_CONFIG = {
        'n_trials': 50,
        'cv_folds': 5,
        'n_estimators': 1000,
        'sample_size': None,
        'early_stopping': 30,
        'max_depth': 10,
        'learning_rate_range': (0.005, 0.3)
    }
    
    # CPU快速训练配置
    CPU_FAST_CONFIG = {
        'n_trials': 20,
        'cv_folds': 3,
        'n_estimators': 100,
        'sample_size': 1000,
        'early_stopping': 10,
        'max_depth': 6,
        'learning_rate_range': (0.01, 0.3)
    }
    
    @staticmethod
    def get_config(mode, intensity):
        if mode == 'gpu':
            return Config.GPU_FULL_CONFIG if intensity == 'full' else Config.GPU_FAST_CONFIG
        else:
            return Config.CPU_FULL_CONFIG if intensity == 'full' else Config.CPU_FAST_CONFIG
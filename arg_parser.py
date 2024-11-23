import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Obesity Prediction Training')
    
    # 添加训练模式选择
    parser.add_argument('--mode', type=str, choices=['gpu', 'cpu'], default='cpu',
                       help='选择训练模式: gpu 或 cpu (默认: cpu)')
    
    # 添加训练强度选择
    parser.add_argument('--intensity', type=str, choices=['fast', 'full'], default='fast',
                       help='选择训练强度: fast 或 full (默认: fast)')
    
    # 添加数据集大小参数
    parser.add_argument('--sample-size', type=int, default=None,
                       help='设置训练样本大小 (默认: 使用全部数据)')
    
    # 添加模型选择参数
    parser.add_argument('--models', nargs='+', 
                       choices=['xgb', 'lgbm', 'rf', 'svm', 'all'],
                       default=['xgb'],
                       help='选择要训练的模型 (默认: xgb)')
    
    return parser.parse_args() 
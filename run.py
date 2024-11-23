from data_preprocessing import load_and_preprocess_data, create_features
from model_training_gpu import ObesityPredictorGPU
from model_training import ObesityPredictor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import Config
from arg_parser import parse_args

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 获取配置
    config = Config.get_config(args.mode, args.intensity)
    
    # 如果命令行指定了样本大小，覆盖配置中的设置
    if args.sample_size is not None:
        config['sample_size'] = args.sample_size
    
    try:
        print(f"\n=== 使用 {args.mode.upper()} 模式进行{args.intensity}强度训练 ===")
        print(f"样本大小: {config['sample_size'] if config['sample_size'] else '全部数据'}")
        print(f"选择的模型: {', '.join(args.models)}")
        
        # 加载数据
        df = load_and_preprocess_data('Obesity_Dataset.xlsx', sample_size=config['sample_size'])
        df = create_features(df)
        
        # 选择GPU或CPU版本的预测器
        predictor = ObesityPredictorGPU() if args.mode == 'gpu' else ObesityPredictor()
        
        # 设置要使用的模型
        if 'all' not in args.models:
            predictor.models = {k: v for k, v in predictor.models.items() if k in args.models}
        
        # 训练模型
        X_train, X_test, y_train, y_test = predictor.prepare_data(df)
        predictor.train(X_train, y_train)
        
        # 评估模型
        predictor.evaluate(X_test, y_test)
        
        # 保存模型
        predictor.save_model(f"obesity_model_{args.mode}_{args.intensity}.pkl")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
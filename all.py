from data_preprocessing import load_and_preprocess_data, create_features
from model_training import ObesityPredictor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    try:
        df = load_and_preprocess_data('Obesity_Dataset.xlsx')
        print("成功读取数据")
        print("数据形状:", df.shape)
        df = create_features(df)
        
        # 数据探索可视化
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='Class')
        plt.title('Distribution of Obesity Classes')
        plt.show()
        
        # 相关性热图
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.show()
        
        # 训练模型
        predictor = ObesityPredictor()
        X_train, X_test, y_train, y_test = predictor.prepare_data(df)
        predictor.train(X_train, y_train)
        
        # 综合评估和可视化
        feature_names = df.drop('Class', axis=1).columns.tolist()
        predictor.evaluate(X_test, y_test, feature_names)
        
    except Exception as e:
        print("执行过程中出错:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
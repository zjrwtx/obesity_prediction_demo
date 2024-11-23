import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(file_path, sample_size=None):
    # 定义列名
    columns = ['Sex', 'Age', 'Height', 'Overweight_Obese_Family', 'Consumption_of_Fast_Food',
               'Frequency_of_Consuming_Vegetables', 'Number_of_Main_Meals_Daily',
               'Food_Intake_Between_Meals', 'Smoking', 'Liquid_Intake_Daily',
               'Calculation_of_Calorie_Intake', 'Physical_Exercise',
               'Schedule_Dedicated_to_Technology', 'Type_of_Transportation_Used', 'Class']
    
    # 根据文件扩展名选择不同的读取方法
    if file_path.endswith('.xlsx'):
        # 读取Excel文件
        df = pd.read_excel(file_path, names=columns)
    else:
        # 读取CSV/TXT文件
        try:
            df = pd.read_csv(file_path, sep='\t', names=columns, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, sep='\t', names=columns, encoding='gbk')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, sep='\t', names=columns, encoding='latin-1')
    
    # 处理缺失值
    for col in df.columns:
        if df[col].dtype != 'object':
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 特征编码
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    
    # 重要：将Class列的值减1，使其从0开始
    df['Class'] = df['Class'] - 1
    
    # 添加异常值处理
    def handle_outliers(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        return df
    
    # 对数值型列进行异常值处理，但排除 'Class' 列
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_columns = numeric_columns.drop('Class')  # 排除 Class 列
    df = handle_outliers(df, numeric_columns)
    
    # 添加特征缩放，但排除 'Class' 列
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    # 如果指定了样本大小，进行随机采样
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    return df

def create_features(df):
    # 增加更多有意义的特征
    
    # 使用身高作为体型指标（移除对Weight的依赖）
    df['Height_Category'] = pd.qcut(df['Height'], q=4, labels=['Short', 'Medium', 'Tall', 'Very Tall'])
    df['Height_Category'] = LabelEncoder().fit_transform(df['Height_Category'])
    
    # 生活习惯得分
    df['Healthy_Habits_Score'] = (
        df['Physical_Exercise'] * 2 +  # 体育锻炼权重加倍
        df['Frequency_of_Consuming_Vegetables'] -
        df['Consumption_of_Fast_Food'] * 1.5 +  # 快餐消费负面影响加权
        (df['Number_of_Main_Meals_Daily'] == 3).astype(int) * 2  # 规律三餐得分
    )
    
    # 压力指数
    df['Stress_Index'] = (
        df['Schedule_Dedicated_to_Technology'] * 0.5 +
        (df['Number_of_Main_Meals_Daily'] < 3).astype(int) +
        df['Smoking']
    )
    
    # 饮食习惯指数
    df['Eating_Habits_Score'] = (
        df['Frequency_of_Consuming_Vegetables'] * 2 -
        df['Consumption_of_Fast_Food'] * 1.5 +
        df['Calculation_of_Calorie_Intake'] +
        df['Number_of_Main_Meals_Daily']
    )
    
    # 生活方式综合评分
    df['Lifestyle_Score'] = (
        df['Physical_Exercise'] * 2 +
        df['Healthy_Habits_Score'] -
        df['Stress_Index'] +
        df['Eating_Habits_Score']
    )
    
    return df 
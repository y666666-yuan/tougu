import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from core import sys


class MABTSLRRecommender:
    def __init__(self, investor_features=None, product_features=None, user_id=None, iterations=100,
                 opt_method='L-BFGS-B', opt_options=None):
        self.full_df = None
        self.feature_cols = None
        self.m = None
        self.q = None
        self.iterations = iterations
        self.opt_method = opt_method
        self.opt_options = opt_options if opt_options is not None else {}
        self.investor_features = investor_features if investor_features is not None else []
        self.product_features = product_features if product_features is not None else []
        self.id_to_userno = {}
        self.userno_to_id = {}
        self.output_path = sys.get_file_path(sys.BASE_TASK_PATH,
                                             str(sys.ALGORITHM_TYPE_BASE),
                                             str(user_id) + '_' + sys.ALGORITHM_BASE_NAME_MABTSLR,
                                             str(int(time.time())))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def load_and_preprocess(self, data_path, filter_conditions=None):
        """加载并预处理数据"""
        if not os.path.exists(data_path):
            print(f"错误：指定的数据路径 {data_path} 不存在。")
            return None
        csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
        if len(csv_files) != 1:
            print(f"错误：路径 {data_path} 下应只有一个 .csv 文件，当前找到 {len(csv_files)} 个。")
            return None
        csv_file = csv_files[0]

        try:
            self.full_df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"读取文件 {csv_file} 时出错：{e}")
            return None

        if filter_conditions is None:
            filter_conditions = {}

        def filter_dataframe(df, feature, value):
            if isinstance(value, (int, float)):
                return df[df[feature] == value]
            elif isinstance(value, dict):
                try:
                    max_value = int(value['max'])
                    df = df[df[feature] < max_value]
                except Exception as e:
                    print(e)
                    print(f"\n过滤值错误! {feature}:{value} ")
                try:
                    min_value = int(value['min'])
                    df = df[df[feature] >= min_value]
                except Exception as e:
                    print(e)
                    print(f"\n过滤值错误! {feature}:{value} ")
                return df
            elif isinstance(value, list):
                try:
                    df = df[df[feature].isin(values=value)]
                except Exception as e:
                    print(e)
                    print(f"\n过滤值错误! {feature}:{value} ")
                return df
            else:
                return df[df[feature].astype(str) == str(value)]

        # 根据用户输入的特征进行数据筛选
        for feature, value in filter_conditions.items():
            if value is None:
                continue
            self.full_df = filter_dataframe(self.full_df, feature, value)

        # 打印数据信息
        print("筛选出来的标准化前的数据集信息：")
        self.full_df.info()
        print("标准化前的数据集前几行：")
        print(self.full_df.head().to_csv(sep='\t', na_rep='nan'))

        # 输出处理后的 DataFrame 为 CSV 文件到 output 文件夹
        output_csv_path = os.path.join(self.output_path, "processed_data.csv")
        self.full_df.to_csv(output_csv_path, index=False)
        print(f"处理后的数据集已保存到 {output_csv_path}")

        # 标准化连续字段
        continuous_cols = [
            'age', 'borrowingcredit', 'loancredit', 'weightedbidrate_percent',
            'baddebts_percent', 'user_invest_count', 'total', 'apr_percent',
            'term', 'level', 'project_invest_count'
        ]
        scaler = StandardScaler()
        self.full_df[continuous_cols] = scaler.fit_transform(self.full_df[continuous_cols])
        # 计算 credit 特征
        self.full_df['credit'] = (self.full_df['loancredit'] + self.full_df['borrowingcredit']) / 2

        # 定义特征列
        self.feature_cols = self.investor_features + self.product_features
        self.full_df[self.feature_cols] = self.full_df[self.feature_cols].astype(float)

        # 生成映射关系
        self.full_df['investor_id'] = self.full_df['userno'].astype('category').cat.codes
        self.id_to_userno = self.full_df[['investor_id', 'userno']].drop_duplicates().set_index('investor_id')[
            'userno'].to_dict()
        self.userno_to_id = {v: k for k, v in self.id_to_userno.items()}

        # 初始化模型参数
        self.m = np.zeros(len(self.feature_cols))
        self.q = np.ones(len(self.feature_cols))

        return self.full_df, self.feature_cols

    def update_parameters(self, X, y):
        """更新模型参数"""
        # 计算概率向量（添加维度处理）
        p = 1 / (1 + np.exp(-X.dot(self.m)))  # shape (n_samples,)
        p = p.reshape(-1, 1)  # 转换为列向量 shape (n_samples, 1)

        # 定义优化目标函数
        def loss(u):
            reg_term = 0.5 * np.sum(self.q * (u - self.m) ** 2)
            log_loss = np.sum(np.log(1 + np.exp(-y * X.dot(u))))
            return reg_term + log_loss

        # 约束优化（m_i >= 0）
        bounds = [(0, None)] * len(self.feature_cols)
        res = minimize(loss, self.m, method=self.opt_method, bounds=bounds, options=self.opt_options)
        self.m = res.x

        # 更新精度参数（修正广播问题）
        self.q += np.sum(X ** 2 * p * (1 - p), axis=0)  # 正确广播

    def train_model(self, metrics_to_calculate):
        """训练模型"""
        if self.full_df is None or self.feature_cols is None:
            print("请先加载并预处理数据。")
            return None, None

        # 假设特征和标签
        X = self.full_df[self.feature_cols].values
        y = self.full_df['reward'].values  # 假设存在 'reward' 列

        for _ in range(self.iterations):
            self.update_parameters(X, y)

        # 模型评估
        probas = 1 / (1 + np.exp(-self.full_df[self.feature_cols].values.dot(self.m)))
        preds = (probas > 0.5).astype(int)

        # 计算评估指标
        metrics = {}
        if 'accuracy' in metrics_to_calculate:
            metrics['accuracy'] = accuracy_score(y, preds)
        if 'precision' in metrics_to_calculate:
            metrics['precision'] = precision_score(y, preds)
        if 'recall' in metrics_to_calculate:
            metrics['recall'] = recall_score(y, preds)
        if 'f1' in metrics_to_calculate:
            metrics['f1'] = f1_score(y, preds)
        if 'auc' in metrics_to_calculate:
            metrics['auc'] = roc_auc_score(y, probas)

        print("\n模型评估指标:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        return self.m, self.q, metrics

    def get_priority_vector(self):
        """获取归一化的优先级向量"""
        priority = self.m.copy()
        priority[priority < 0] = 0
        return priority / (priority.sum() + 1e-8)

    def generate_recommendations(self, target_investor_id):
        """生成推荐结果"""
        target_userno = self.id_to_userno[target_investor_id]

        # 提取目标用户特征
        target_mask = self.full_df['userno'] == target_userno
        target_features = self.full_df.loc[target_mask, self.feature_cols].mean(axis=0).values

        # 计算所有用户的优先级向量
        user_priorities = {}
        for user in self.full_df['userno'].unique():
            user_mask = self.full_df['userno'] == user
            user_features = self.full_df.loc[user_mask, self.feature_cols].mean(axis=0).values
            user_priorities[user] = user_features

        # 计算余弦相似度
        similarities = {}
        target_priority = self.get_priority_vector()
        for user, features in user_priorities.items():
            if user == target_userno:
                continue
            dot_product = np.dot(target_priority, features)
            norm_product = np.linalg.norm(target_priority) * np.linalg.norm(features)
            similarities[user] = dot_product / (norm_product + 1e-8)

        # 获取Top-5相似用户
        top_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]

        # 计算产品得分
        product_scores = {}
        all_products = self.full_df['PROJECTNO'].unique()

        for product in all_products:
            product_features = self.full_df[self.full_df['PROJECTNO'] == product][self.feature_cols].iloc[0].values
            base_score = np.dot(self.get_priority_vector(), product_features)

            total_weight = sum(sim for _, sim in top_users)
            for user, sim in top_users:
                user_purchased = self.full_df[(self.full_df['userno'] == user) &
                                              (self.full_df['PROJECTNO'] == product) &
                                              (self.full_df['reward'] == 1)]
                if not user_purchased.empty:
                    product_scores[product] = product_scores.get(product, 0) + (sim / total_weight) * base_score

        sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        # 表1：Top-5相似投资者
        table1 = pd.DataFrame({
            'Userno': [u[0] for u in top_users],
            'Similarity': [u[1] for u in top_users]
        })

        # 表2：Top-5推荐产品
        table2 = pd.DataFrame({
            'Projectno': [p[0] for p in sorted_products],
            'Score': [p[1] for p in sorted_products]
        })

        # 表3：特征贡献值
        # added by jiang for 计算每个客户的特征贡献值
        user_weights = self.get_priority_vector() * target_features  # 根据客户特征调整贡献值
        user_weights = user_weights / (user_weights.sum() + 1e-8)  # 归一化
        feature_names = self.feature_cols
        table3 = pd.DataFrame({
            'Feature': feature_names,
            'Contribution': user_weights
        }).sort_values('Contribution', ascending=False)

        # added by jiang for 筛选非零特征贡献值的特征
        non_zero_table3 = table3[table3['Contribution'] > 0]
        if non_zero_table3.empty:
            # 若所有特征贡献值都为零，创建一个默认的 DataFrame
            non_zero_table3 = pd.DataFrame({
                'Feature': ['No significant feature'],
                'Contribution': [1.0]
            })

        return table1, table2, non_zero_table3, target_userno

    def run_algorithm(self, filter_conditions, data_path="E:/桌面/研一下/导师任务/智能投顾平台/Data", metrics_to_calculate=None):
        """主函数封装"""
        if metrics_to_calculate is None:
            metrics_to_calculate = ['accuracy', 'precision', 'recall', 'f1', 'auc']

        # 1. 数据预处理
        print("开始加载数据...")
        result = self.load_and_preprocess(data_path, filter_conditions)
        if result is None:
            print("数据加载失败，请检查数据路径和文件。")
            return None, "数据加载失败"
        print("数据加载完成")

        # 2. 模型训练
        self.m, self.q, metrics = self.train_model(metrics_to_calculate)
        print("模型训练完成")

        # 3. 获取所有投资者 ID，针对每个投资者生成特定的结果
        all_investor_ids = self.full_df['investor_id'].unique()
        all_table1_list = []
        all_table2_list = []
        all_table3_list = []

        for target_investor_id in all_investor_ids:
            try:
                target_userno = self.id_to_userno[target_investor_id]
            except KeyError:
                print(f"错误：投资者 investor_id '{target_investor_id}' 对应的 userno 不存在")
                continue

            # 4. 生成结果
            table1, table2, table3, verified_userno = self.generate_recommendations(target_investor_id)
            # 计算该用户在筛选数据集中的样本个数
            sample_count = len(self.full_df[self.full_df['investor_id'] == target_investor_id])

            # 在 table1 前添加 userno 和 sample_count 列
            table1.insert(0, 'userno', int(target_userno))
            table1.insert(1, 'sample_count', sample_count)

            # 在 table2 前添加 userno 和 sample_count 列
            table2.insert(0, 'userno', int(target_userno))
            table2.insert(1, 'sample_count', sample_count)

            # 在 table3 前添加 userno 和 sample_count 列
            table3.insert(0, 'userno', int(target_userno))
            table3.insert(1, 'sample_count', sample_count)

            # 5. 输出结果
            print("=" * 50)
            print(f"目标投资者验证信息")
            print(f"输入userno: {target_userno}")
            print(f"映射investor_id: {target_investor_id}")
            print(f"数据库验证userno: {verified_userno}")
            print("=" * 50 + "\n")

            print("表1：Top-5相似投资者")
            print(table1.to_string(index=False))
            print("\n表2：Top-5推荐产品")
            print(table2.to_string(index=False))
            print("\n表3：特征贡献值")
            print(table3.to_string(index=False))

            table1.to_csv(os.path.join(self.output_path, f"userno_{target_userno}_similar_investors.csv"), index=False)
            table2.to_csv(os.path.join(self.output_path, f"userno_{target_userno}_recommended_products.csv"),
                          index=False)
            table3.to_csv(os.path.join(self.output_path, f"userno_{target_userno}_feature_contributions.csv"),
                          index=False)

            # 将每次生成的表格添加到对应的列表中
            all_table1_list.append(table1)
            all_table2_list.append(table2)
            all_table3_list.append(table3)

        # 纵向合并所有表格
        all_table1 = pd.concat(all_table1_list, ignore_index=True) if all_table1_list else pd.DataFrame()
        all_table2 = pd.concat(all_table2_list, ignore_index=True) if all_table2_list else pd.DataFrame()
        all_table3 = pd.concat(all_table3_list, ignore_index=True) if all_table3_list else pd.DataFrame()

        # 保存合并后的表格
        all_table1.to_csv(os.path.join(self.output_path, "all_similar_investors.csv"), index=False)
        all_table2.to_csv(os.path.join(self.output_path, "all_recommended_products.csv"), index=False)
        all_table3.to_csv(os.path.join(self.output_path, "all_feature_contributions.csv"), index=False)

        # 将模型评估指标保存为 CSV 文件，保留四位小数
        metrics = {k: float(f"{v:.4f}") for k, v in metrics.items()}
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_path = os.path.join(self.output_path, "model_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"模型评估指标已保存到: {metrics_path}")

        print("所有结果保存完成")
        return self.output_path, None


def main():
    parser = argparse.ArgumentParser(description='运行 MAB-TS-LR 算法并选择输出的评估指标')
    parser.add_argument('--metrics', nargs='+',
                        choices=['accuracy', 'precision', 'recall', 'f1', 'auc'],
                        default=['accuracy', 'precision', 'recall', 'f1', 'auc'],
                        # 'accuracy':准确率; 'precision':查准率（精确率）; 'recall':查全率（召回率）; 'f1':F1值; 'auc':AUC.
                        help='选择要输出的评估指标，默认输出所有指标')
    args = parser.parse_args()

    # 定义样本筛选维度
    filter_conditions = {
        'gender': 0,
        'age': 30,
        'edu_level': None,
        'CURRENTIDENTITY': None,
        'user_invest_count': None,
        'total': None,
        'apr_percent': None,
        'term': None,
        'REPAYMENT': None,
        'level': None,
        'project_invest_count': None
    }
    # 可选投资者特征
    custom_investor_features = [
        'gender', 'age', 'edu_level', 'University', 'CURRENTIDENTITY',
        'credit', 'weightedbidrate_percent',
        'baddebts_percent', 'user_invest_count'
    ]
    # 可选产品特征
    custom_product_features = [
        'total', 'apr_percent', 'term', 'REPAYMENT', 'level',
        'project_invest_count'
    ]

    mabtslr_runner = MABTSLRRecommender(
        investor_features=custom_investor_features,
        product_features=custom_product_features,
        iterations=100,  # 超参数名称:迭代次数; 取值范围:1到正无穷的整数; 常见取值:50到1000之间的整数.
        opt_method='L-BFGS-B',  # 超参数名称:优化方法; 常见取值选项:'L-BFGS-B'、'BFGS'、'Newton-CG'、'SLSQP'.
        opt_options={'maxiter': 1000}  # 超参数名称:优化选项; 常见取值选项:{'maxiter': 1000}、{'ftol': 1000} .
    )
    data_path = sys.get_file_path('data/dataset/default/')
    result_path, err_msg = mabtslr_runner.run_algorithm(filter_conditions, data_path, args.metrics)
    if err_msg is None:
        print(f"实验结果跑完，结果路径：{result_path}")
    else:
        print("实验报错")


if __name__ == '__main__':
    main()

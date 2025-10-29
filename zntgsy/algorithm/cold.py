import os
import time
import numpy as np
import pandas as pd
import shap
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from core import sys


class ColdRecommender:
    def __init__(self, investor_features=None, product_features=None, user_id=None, iterations=100, depth=6,
                 learning_rate=0.1):
        self.full_df = None
        self.user_weights = None
        self.model = None
        self.id_to_userno = None
        self.userno_to_id = None
        # 默认投资者特征
        default_investor_features = [
            'gender', 'age', 'edu_level', 'University', 'CURRENTIDENTITY',
            'credit', 'weightedbidrate_percent',
            'baddebts_percent', 'user_invest_count'
        ]
        # 默认产品特征
        default_product_features = [
            'total', 'apr_percent', 'term', 'REPAYMENT', 'level',
            'project_invest_count'
        ]
        self.investor_features = investor_features if investor_features is not None else default_investor_features
        self.product_features = product_features if product_features is not None else default_product_features
        self.output_path = sys.get_file_path(sys.BASE_TASK_PATH,
                                             str(sys.ALGORITHM_TYPE_BASE),
                                             str(user_id) + '_' + sys.ALGORITHM_BASE_NAME_COLD,
                                             str(int(time.time())))

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate

    def load_and_preprocess_data(self, data_path, filter_conditions=None):
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

        # 标准化连续特征
        cont_features = [
            'age', 'borrowingcredit', 'loancredit', 'weightedbidrate_percent',
            'baddebts_percent', 'user_invest_count', 'total', 'apr_percent',
            'term', 'level', 'project_invest_count'
        ]
        scaler = StandardScaler()
        self.full_df[cont_features] = scaler.fit_transform(self.full_df[cont_features])
        # 计算 credit 特征
        self.full_df['credit'] = (self.full_df['loancredit'] + self.full_df['borrowingcredit']) / 2

        # 生成映射关系
        self.full_df['investor_id'] = self.full_df['userno'].astype('category').cat.codes
        self.full_df['product_id'] = self.full_df['PROJECTNO'].astype('category').cat.codes

        # 创建双向映射字典
        self.id_to_userno = self.full_df[['investor_id', 'userno']].drop_duplicates().set_index('investor_id')[
            'userno'].to_dict()
        self.userno_to_id = {v: k for k, v in self.id_to_userno.items()}
        # clod算法需要将小数字段转为字符串
        self.full_df[cont_features] = self.full_df[
            cont_features].astype(str)
        return self.full_df

    def train_model(self, metrics_to_calculate):
        """训练 CatBoost 模型并生成用户优先级向量"""
        if self.full_df is None:
            print("请先加载并预处理数据。")
            return None, None

        # 假设特征和标签
        features = self.investor_features + self.product_features
        label = 'reward'
        continuous_features = [
            'age', 'credit', 'weightedbidrate_percent', 'baddebts_percent',
            'user_invest_count', 'total', 'apr_percent',
            'term', 'level', 'project_invest_count',
        ]
        # 计算分类特征，即 features 中不属于连续变量的特征
        cat_features = [feature for feature in features if feature not in continuous_features]

        train_pool = Pool(data=self.full_df[features], label=self.full_df[label], cat_features=cat_features)
        self.model = CatBoostClassifier(iterations=self.iterations, depth=self.depth,  # 使用类属性设置参数
                                        learning_rate=self.learning_rate, verbose=0)
        self.model.fit(train_pool)

        # 计算 Shapley 值
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(train_pool)

        # 生成用户优先级向量
        self.user_weights = {}
        # 创建一个临时 DataFrame 用于后续索引匹配
        temp_df = self.full_df.reset_index(drop=True)
        for userno in temp_df['userno'].unique():
            user_data = temp_df[temp_df['userno'] == userno]
            # 使用重置后的索引访问 shap_values
            user_shap = shap_values[user_data.index].mean(axis=0)
            total_importance = np.abs(user_shap).sum()
            self.user_weights[userno] = user_shap / total_importance  # 归一化

        # 模型评估
        # with torch.no_grad():
        outputs = self.model.predict_proba(self.full_df[features])[:, 1]
        preds = (outputs > 0.5).astype(int)

        # 计算评估指标
        metrics = {}
        if 'accuracy' in metrics_to_calculate:
            metrics['accuracy'] = accuracy_score(self.full_df[label], preds)
        if 'precision' in metrics_to_calculate:
            metrics['precision'] = precision_score(self.full_df[label], preds)
        if 'recall' in metrics_to_calculate:
            metrics['recall'] = recall_score(self.full_df[label], preds)
        if 'f1' in metrics_to_calculate:
            metrics['f1'] = f1_score(self.full_df[label], preds)
        if 'auc' in metrics_to_calculate:
            metrics['auc'] = roc_auc_score(self.full_df[label], outputs)

        print("\n模型评估指标:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        return self.model, metrics

    def recommend_products(self, target_userno, top_k=5):
        """为目标用户推荐产品"""
        if target_userno not in self.user_weights:
            raise ValueError(f"Target user {target_userno} not found in user weights.")

        # 获取目标用户优先级向量
        target_weight = self.user_weights[target_userno]

        # 计算所有用户的余弦相似度
        similarity_scores = {}
        for userno in self.user_weights:
            if userno == target_userno:
                continue
            sim = cosine_similarity([target_weight], [self.user_weights[userno]])[0][0]
            similarity_scores[userno] = sim

        # 选取 Top-k 相似用户
        top_users = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 获取目标用户已投资的产品
        target_purchased = set(self.full_df[(self.full_df['userno'] == target_userno) & (self.full_df['reward'] == 1)][
                                   'PROJECTNO'])

        # 收集相似用户投资但目标用户未投资的产品
        product_scores = {}
        for userno, sim in top_users:
            # 获取相似用户投资的产品（reward=1）
            purchased = self.full_df[(self.full_df['userno'] == userno) & (self.full_df['reward'] == 1)][
                'PROJECTNO'].unique()
            for project in purchased:
                if project not in target_purchased:
                    # 加权得分：相似度作为权重
                    product_scores[project] = product_scores.get(project, 0) + sim

        # 按得分排序，推荐 Top-k
        top_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return top_users, top_products

    def generate_results(self, target_investor_id):
        """生成推荐结果表格"""
        if self.full_df is None or self.model is None or self.user_weights is None:
            print("请先加载并预处理数据，训练模型。")
            return None

        target_userno = self.id_to_userno[target_investor_id]
        top_users, top_products = self.recommend_products(target_userno)

        # 表1：Top-5相似投资者
        table1 = pd.DataFrame({
            'Userno': [u[0] for u in top_users],
            'Similarity': [u[1] for u in top_users]
        })

        # 表2：Top-5推荐产品
        table2 = pd.DataFrame({
            'Projectno': [p[0] for p in top_products],
            'Score': [p[1] for p in top_products]
        })

        # 表3：特征贡献值
        user_shap = self.user_weights[target_userno]
        feature_names = self.investor_features + self.product_features
        table3 = pd.DataFrame({
            'Feature': feature_names,
            'Contribution': user_shap
        }).sort_values('Contribution', ascending=False)

        return table1, table2, table3, target_userno

    def run_cold(self, filter_conditions, data_path="E:/桌面/研一下/导师任务/智能投顾平台/Data",
                 metrics_to_calculate=None):
        """主函数封装"""
        if metrics_to_calculate is None:
            metrics_to_calculate = ['accuracy', 'precision', 'recall', 'f1', 'auc']

        # 1. 数据预处理
        print("开始加载数据...")
        result = self.load_and_preprocess_data(data_path, filter_conditions)
        if result is None:
            print("数据加载失败，请检查数据路径和文件。")
            return None, "数据加载失败"
        print("数据加载完成")

        # 2. 模型训练
        self.model, metrics = self.train_model(metrics_to_calculate)

        if metrics is None:
            return None, "模型训练失败"

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
            table1, table2, table3, verified_userno = self.generate_results(target_investor_id)
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
    parser = argparse.ArgumentParser(description='运行 Cold 推荐模型并选择输出的评估指标')
    parser.add_argument('--metrics', nargs='+',
                        choices=['accuracy', 'precision', 'recall', 'f1', 'auc'],
                        default=['accuracy', 'precision', 'recall', 'f1', 'auc'],
                        # 'accuracy':准确率; 'precision':查准率（精确率）; 'recall':查全率（召回率）; 'f1':F1值; 'auc':AUC.
                        help='选择要输出的评估指标，默认输出所有指标')

    args = parser.parse_args()

    # 定义样本筛选维度
    filter_conditions = {
        'gender': 0,
        'age': {
            "max": 30,
            "min": 20
        },
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
    cold_runner = ColdRecommender(
        investor_features=custom_investor_features,
        product_features=custom_product_features,
        iterations=100,  # 超参数名称:迭代次数; 取值范围:1到正无穷的整数; 常见取值:100到1000之间的整数.
        depth=6,  # 树的深度:; 取值范围:1到16的整数; 常见取值:4到10之间的整数.
        learning_rate=0.1,  # 超参数名称:学习率; 取值范围:0到1之间的浮点数; 常见取值:0.01到0.3之间.
        user_id=2
    )
    data_path = sys.get_file_path('data/dataset/default/')
    cold_runner.run_cold(filter_conditions, data_path, args.metrics)


if __name__ == '__main__':
    main()

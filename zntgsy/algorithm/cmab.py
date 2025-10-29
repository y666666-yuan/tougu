import math
import os
import time
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve_triangular
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from core import sys
import json
# 在run_cmab方法中使用多线程处理用户
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

ALGORITHM_TYPE = 'cmab'


class CMABRunner:
    def __init__(self, investor_features=None, product_features=None, user_id=None,
                 alpha=0.7, gamma=0.0009, lambd=0.0009, delta=0.0009,
                 m=5, T=1000, init_theta=20.0, output_suffix=""):
        self.seed = 42
        np.random.seed(self.seed)
        self.full_df = None
        default_investor_features = [
            'gender', 'age', 'edu_level', 'University', 'CURRENTIDENTITY',
            'credit', 'weightedbidrate_percent', 'baddebts_percent', 'user_invest_count'
        ]
        default_product_features = [
            'total', 'apr_percent', 'term', 'REPAYMENT', 'level', 'project_invest_count'
        ]
        self.investor_features = investor_features or default_investor_features
        self.product_features = product_features or default_product_features

        # 创建带时间戳的结果路径
        # self.output_path = sys.get_file_path("data/task_result", str(user_id) + '_' + ALGORITHM_TYPE,
        #                                      str(int(time.time())))
        #
        # if not os.path.exists(self.output_path):
        #     os.makedirs(self.output_path)
        timestamp = str(int(time.time()))
        self.output_path = sys.get_file_path(sys.BASE_TASK_PATH,
                                             str(sys.ALGORITHM_TYPE_CONSTRAINT),
                                             f"{user_id}_{sys.ALGORITHM_CONSTRAINT_NAME_CMAB}",
                                             timestamp)
        os.makedirs(self.output_path, exist_ok=True)
        # 算法超参数配置
        self.alpha = alpha  # 探索系数
        self.gamma = gamma  # q的步长
        self.lambd = lambd  # Q的步长
        self.delta = delta  # theta的步长
        self.m = m  # 每轮推荐数
        self.T = T  # 总训练轮数
        self.init_theta = init_theta  # theta初始值

    def manual_equal_size_binning(self, data_series, q=4):
        """等额分箱函数"""
        sorted_indices = np.argsort(data_series)
        n = len(data_series)
        bins = np.empty(n, dtype=int)
        bin_size = n // q
        remainder = n % q

        start = 0
        for i in range(q):
            end = start + bin_size + (1 if i < remainder else 0)
            bins[sorted_indices[start:end]] = i
            start = end
        return bins

    def load_and_preprocess(self, data_path, filter_conditions=None):
        if not os.path.exists(data_path):
            print(f"错误：指定的数据路径 {data_path} 不存在。")
            return None
        csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
        if len(csv_files) != 1:
            print(f"错误：路径 {data_path} 下应只有一个 .csv 文件，当前找到 {len(csv_files)} 个。")
            return None
        csv_file = csv_files[0]
        self.full_df = pd.read_csv(csv_file)

        if filter_conditions:
            for feature, value in filter_conditions.items():
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    self.full_df = self.full_df[self.full_df[feature] == value]
                elif isinstance(value, dict):
                    if 'max' in value:
                        self.full_df = self.full_df[self.full_df[feature] < int(value['max'])]
                    if 'min' in value:
                        self.full_df = self.full_df[self.full_df[feature] >= int(value['min'])]
                elif isinstance(value, list):
                    self.full_df = self.full_df[self.full_df[feature].isin(value)]
                else:
                    self.full_df = self.full_df[self.full_df[feature].astype(str) == str(value)]

        raw_data_path = os.path.join(self.output_path, "processed_data.csv")
        self.full_df.to_csv(raw_data_path, index=False)

        # 定义所有连续特征（包含投资者和产品特征）
        cont_features = [
            'age', 'total', 'apr_percent', 'term',
            'project_invest_count', 'user_invest_count',
            'weightedbidrate_percent', 'baddebts_percent'
        ]
        scaler = StandardScaler()
        self.full_df[cont_features] = scaler.fit_transform(self.full_df[cont_features])

        # 生成映射关系
        self.full_df['investor_id'] = self.full_df['userno'].astype('category').cat.codes
        self.id_to_userno = self.full_df[['investor_id', 'userno']].drop_duplicates() \
            .set_index('investor_id')['userno'].to_dict()

        return self.full_df

    class RoboAdvisor:
        def __init__(self, alpha, d, K, m, T, gamma, lambd, delta, theta):
            self.alpha = alpha
            self.d = d
            self.K = K
            self.m = m
            self.T = T
            self.A = np.eye(d)
            self.b = np.zeros(d)
            self.beta = np.zeros(d)
            self.theta = theta
            self.Q = np.array([0.0])
            self.gamma = gamma
            self.lambd = lambd
            self.delta = delta
            self.q = np.zeros(K)
            self.q[np.random.choice(K, m, replace=False)] = 1
            # 添加参数校验
            assert 0 < alpha <= 1, "探索系数alpha应在(0,1]范围内"
            assert m > 0, "推荐数量m应大于0"

            # 初始化方法改进
            self.q = np.zeros(K)
            indices = np.random.choice(K, m, replace=False)
            self.q[indices] = 1  # 更安全的初始化方式

        def recommend(self, X_features, constraint_feature):
            # 添加维度校验
            assert X_features.shape[0] == len(constraint_feature), "特征矩阵与约束向量维度不匹配"

            # 改进后的UCB计算（向量化）
            try:
                L = cholesky(self.A, lower=True)
                L_inv_X = solve_triangular(L, X_features.T, lower=True)
                ucb = self.alpha * np.sqrt(np.sum(L_inv_X ** 2, axis=0))
                r_hat = X_features @ self.beta + ucb
            except np.linalg.LinAlgError:
                r_hat = X_features @ self.beta

            # 改进梯度计算
            grad_L = self.theta * r_hat - self.Q[0] * constraint_feature
            q_new = self.q - self.gamma * grad_L
            self.q = self.project_C(q_new)

            # 改进推荐结果生成
            selected = np.argpartition(self.q, -self.m)[-self.m:]
            return selected[np.argsort(self.q[selected])[::-1]], r_hat

        def project_C(self, q_new):
            # 改进投影函数
            q_clipped = np.clip(q_new, 0, 1)
            if np.sum(q_clipped) <= self.m:
                return q_clipped
            return (q_clipped >= np.partition(q_clipped, -self.m)[-self.m]).astype(float)

    class MABTSLRRecommender:
        def __init__(self, output_path: str, iterations=100, opt_method='L-BFGS-B', opt_options=None,
                     investor_features=None, product_features=None):
            self.full_df = None
            self.feature_cols = None
            self.m = None
            self.q = None
            self.iterations = iterations
            self.opt_method = opt_method
            self.opt_options = opt_options if opt_options is not None else {}
            self.id_to_userno = {}
            self.userno_to_id = {}
            self.investor_features = investor_features if investor_features is not None else []
            self.product_features = product_features if product_features is not None else []

            self.processed_data = sys.get_file_path(output_path, "processed_data.csv")
            self.output_path = sys.get_file_path(output_path, "mabtslr")
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

        def load_and_preprocess(self):

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

        def generate_recommendations(self, target_investor_id, target_priority):
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

            for user, features in user_priorities.items():
                if user == target_userno:
                    continue
                dot_product = np.dot(target_priority, features)
                norm_product = np.linalg.norm(target_priority) * np.linalg.norm(features)
                if norm_product < 1e-8:  # 很小的值，避免除以零
                    similarity = 0
                else:
                    similarity = dot_product / (norm_product + 1e-8)
                similarities[user] = similarity
                # similarities[user] = dot_product / (norm_product + 1e-8)

            # 获取Top-5相似用户
            top_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]

            # 计算产品得分
            product_scores = {}
            all_products = self.full_df['PROJECTNO'].unique()

            for product in all_products:
                product_features = self.full_df[self.full_df['PROJECTNO'] == product][self.feature_cols].iloc[0].values
                base_score = np.dot(target_priority, product_features)

                total_weight = sum(sim for _, sim in top_users)
                if total_weight < 1e-8:  # 如果总权重很小，则跳过
                    continue
                for user, sim in top_users:
                    user_purchased = self.full_df[(self.full_df['userno'] == user) &
                                                  (self.full_df['PROJECTNO'] == product) &
                                                  (self.full_df['reward'] == 1)]
                    if not user_purchased.empty:
                        if not np.isfinite(sim) or not np.isfinite(base_score):
                            continue
                        try:
                            product_scores[product] = product_scores.get(product, 0) + (sim / total_weight) * base_score
                        except (ZeroDivisionError, FloatingPointError):
                            # 如果出现错误，跳过这个产品
                            continue

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
            user_weights = target_priority * target_features  # 根据客户特征调整贡献值
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
            sample_count = len(self.full_df[self.full_df['investor_id'] == target_investor_id])
            return table1, table2, non_zero_table3, target_userno, sample_count

        # 在MABTSLRRecommender中同样使用多线程处理用户推荐
        def generate_recommendations_parallel(self, target_investor_ids):
            """并行生成推荐结果"""
            target_priority = self.get_priority_vector()
            print("开始并行触发")
            with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
                futures = {
                    executor.submit(self.generate_recommendations, tid, target_priority): tid
                    for tid in target_investor_ids
                }

                results = []
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"生成推荐时发生错误: {str(e)}")

                return results

        def run_algorithm(self, metrics_to_calculate=None):
            """主函数封装"""
            if metrics_to_calculate is None:
                metrics_to_calculate = ['accuracy', 'precision', 'recall', 'f1', 'auc']

            # 1. 数据预处理
            print("开始加载数据...")
            try:
                self.full_df = pd.read_csv(self.processed_data)
            except Exception as e:
                print(f"读取processed_data.csv 时出错：{e}")
                return None
            self.load_and_preprocess()
            # 2. 模型训练
            self.m, self.q, metrics = self.train_model(metrics_to_calculate)
            print("模型训练完成")

            # 3. 获取所有投资者 ID，针对每个投资者生成特定的结果
            all_investor_ids = self.full_df['investor_id'].unique()
            all_table1_list = []
            all_table2_list = []
            all_table3_list = []
            # 并行生成推荐
            results = self.generate_recommendations_parallel(all_investor_ids)

            for table1, table2, table3, verified_userno, sample_count in results:
                # 4. 生成结果

                # 在 table1 前添加 userno 和 sample_count 列
                table1.insert(0, 'userno', int(verified_userno))
                table1.insert(1, 'sample_count', sample_count)

                # 在 table2 前添加 userno 和 sample_count 列
                table2.insert(0, 'userno', int(verified_userno))
                table2.insert(1, 'sample_count', sample_count)

                # 在 table3 前添加 userno 和 sample_count 列
                table3.insert(0, 'userno', int(verified_userno))
                table3.insert(1, 'sample_count', sample_count)

                # 将每次生成的表格添加到对应的列表中
                all_table1_list.append(table1)
                all_table2_list.append(table2)
                all_table3_list.append(table3)

            print("******************************")
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"纵向合并所有表格:{now}")
            # 纵向合并所有表格
            all_table1 = pd.concat(all_table1_list, ignore_index=True) if all_table1_list else pd.DataFrame()
            all_table2 = pd.concat(all_table2_list, ignore_index=True) if all_table2_list else pd.DataFrame()
            all_table3 = pd.concat(all_table3_list, ignore_index=True) if all_table3_list else pd.DataFrame()
            print("******************************")
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"保存合并后的表格:{now}")
            # 保存合并后的表格
            all_table1.to_csv(os.path.join(self.output_path, "all_similar_investors.csv"), index=False)
            all_table2.to_csv(os.path.join(self.output_path, "all_recommended_products.csv"), index=False)
            all_table3.to_csv(os.path.join(self.output_path, "all_feature_contributions.csv"), index=False)
            print("******************************")
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"将模型评估指标保存为 CSV 文件，保留四位小数:{now}")
            # 将模型评估指标保存为 CSV 文件，保留四位小数
            metrics = {k: float(f"{v:.4f}") for k, v in metrics.items()}
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            metrics_path = os.path.join(self.output_path, "model_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
            print(f"模型评估指标已保存到: {metrics_path}")

            print("所有结果保存完成")
            return self.output_path, None

    def run_cmab(self, filter_conditions, data_path, constraint_config, metrics_to_calculate):
        print("******************************")
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"算法开始运行:{now}")
        if metrics_to_calculate is None:
            metrics_to_calculate = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        # 获取当前约束配置
        current_constraint = next(iter(constraint_config.keys()))
        config = constraint_config[current_constraint]
        self.full_df = None
        # 数据预处理
        df = self.load_and_preprocess(data_path, filter_conditions)
        print(f"总用户数: {len(df['userno'].unique())}")

        # 初始化结果存储（仅当前约束）
        constraint_results = {
            current_constraint: {
                'recommendations': [],
                'similar_investors': [],
                'features': []
            }
        }
        y_true = []  # 真实标签
        y_pred = []  # 预测分数


        users = df['userno'].unique()
        results = []
        with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            futures = [executor.submit(self.process_user, userno, df, config, current_constraint,
                                       self.m, self.T, self.alpha, self.gamma,
                                       self.lambd, self.delta, self.init_theta,
                                       self.product_features)
                       for userno in users]
            for f in as_completed(futures):
                res = f.result()
                if res is not None:
                    results.append(res)
        if not results:
            return None, "没有生成任何推荐结果"
        for df, yt, yp in results:
            constraint_results[current_constraint]['recommendations'].append(df)
            y_true.extend(yt)
            y_pred.extend(yp)

        # 合并推荐结果
        rec_df = pd.concat(constraint_results[current_constraint]['recommendations'])
        rec_df.to_csv(os.path.join(self.output_path, f"all_recommended_products.csv"), index=False)

        # 评估指标计算（保持原有逻辑）
        metrics = {}
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        valid_samples = len(y_true_np) > 0
        has_positive = np.any(y_true_np == 1)
        has_negative = np.any(y_true_np == 0)

        if 'accuracy' in metrics_to_calculate and valid_samples:
            metrics['accuracy'] = accuracy_score(y_true_np, np.round(y_pred_np))
        if 'precision' in metrics_to_calculate and valid_samples:
            metrics['precision'] = precision_score(y_true_np, np.round(y_pred_np), zero_division=0)
        if 'recall' in metrics_to_calculate and valid_samples and has_positive:
            metrics['recall'] = recall_score(y_true_np, np.round(y_pred_np))
        if 'f1' in metrics_to_calculate and valid_samples and has_positive:
            metrics['f1'] = f1_score(y_true_np, np.round(y_pred_np))
        if 'auc' in metrics_to_calculate and valid_samples and has_positive and has_negative:
            metrics['auc'] = roc_auc_score(y_true_np, y_pred_np)

        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_path = os.path.join(self.output_path, "model_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"算法评估指标已保存到: {metrics_path}")
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"cmab算法结束运行:{now}")
        print("******************************")
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"mabtslr算法开始运行:{now}")
        mabtslr_runner = self.MABTSLRRecommender(output_path=self.output_path,
                                                 iterations=100,
                                                 opt_method='L-BFGS-B',
                                                 opt_options={'maxiter': 1000},
                                                 investor_features=self.investor_features,
                                                 product_features=self.product_features)
        result_path, err_msg = mabtslr_runner.run_algorithm()
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"mabtslr算法结束运行:{now}")
        print("******************************")
        if err_msg is not None:
            return None, err_msg
        try:
            self.build_radar_data()
            return self.output_path, None
        except Exception as e:
            return None, "雷达数据生成文件失败"

    def process_user(self, userno, df, config, constraint, m, T, alpha, gamma, lambd, delta, init_theta,
                     product_features):
        try:
            user_df = df[df['userno'] == userno].copy()
            if user_df.empty:
                return None, None, None
            if constraint in ['apr_percent', 'term']:
                user_df[f'{constraint}_rank'] = user_df[constraint].rank(method='first')
                user_df[f'{constraint}_category'] = self.manual_equal_size_binning(
                    user_df[f'{constraint}_rank'], q=4)
                constraint_mask = user_df[f'{constraint}_category'].isin([v - 1 for v in config['category']])
            elif constraint in ['REPAYMENT', 'level']:
                user_df[constraint] = user_df[constraint].astype(int)
                constraint_mask = user_df[constraint].isin(config['category'])
            else:
                constraint_mask = np.ones(len(user_df), dtype=bool)

            user_df = user_df[constraint_mask].copy()
            if user_df.empty or len(user_df) < m:
                return None, None, None

            X_features = user_df[product_features].values
            robo = self.RoboAdvisor(
                alpha=alpha, d=X_features.shape[1], K=X_features.shape[0], m=m,
                T=T, gamma=gamma, lambd=lambd, delta=delta, theta=init_theta
            )
            constraint_feature = np.ones(len(user_df), dtype=int)
            selected, _ = robo.recommend(X_features, constraint_feature)
            selected = selected.ravel()[:5]
            recommended_projects = user_df.iloc[selected]['PROJECTNO'].tolist()
            # 收集预测数据（保留核心数据收集）
            y_true = user_df['reward'].values
            valid_indices = np.arange(len(user_df))
            y_pred = robo.q[valid_indices]
            result_df = pd.DataFrame({
                'userno': [userno] * len(recommended_projects),
                'sample_count': [len(user_df)] * len(recommended_projects),
                'Projectno': recommended_projects
            })
            return result_df, y_true, y_pred
        except Exception:
            return None, None, None

    def build_radar_data(self):
        """
        计算雷达图数据并生成对应的数据文件
        """
        try:
            fields = ['level', 'total', 'apr_percent', 'term', 'REPAYMENT', 'project_invest_count']
            project_field = "PROJECTNO"
            # 读取 筛选完的全部数据
            df = pd.read_csv(sys.get_file_path(self.output_path, "processed_data.csv"))
            # 读取 cmab算法的推荐结果
            df_cmab_recommend = pd.read_csv(sys.get_file_path(self.output_path, "all_recommended_products.csv"))
            df_cmab_recommend = df_cmab_recommend.rename(columns={"Projectno": project_field})

            # 读取 mabtslr算法的推荐结果
            df_mabtslr_recommend = pd.read_csv(
                sys.get_file_path(self.output_path, "mabtslr", "all_recommended_products.csv"))
            df_mabtslr_recommend = df_mabtslr_recommend.rename(columns={"Projectno": project_field})
            mabtslr_total = len(df_mabtslr_recommend)

            radar_field_dict = {}
            for field in fields:
                new_key = field + "_group_radar"
                # 按 field 排序
                group_num = 4
                if field == 'REPAYMENT':
                    group_num = 3
                elif field == 'level':
                    group_num = 8
                df_sorted = self.assign_equal_parts(df, field, new_key, group_num)

                # 构建 PROJECTNO -> 等级 的映射字典
                project_level_dict = dict(zip(df_sorted[project_field], df_sorted[new_key]))
                # level 4等分 group by 分组数据
                # level_group_count = df_sorted.groupby(new_key)[project_field].count().to_dict()

                radar_data = {}
                radar_data['cmab'], cmab_max = self.build_radar_data_by_recommend_df(df_cmab_recommend,
                                                                                     project_level_dict,
                                                                                     project_field, new_key,
                                                                                     group_num=group_num)

                radar_data['mabtslr'], mabtslr_max = self.build_radar_data_by_recommend_df(df_mabtslr_recommend,
                                                                                           project_level_dict,
                                                                                           project_field, new_key,
                                                                                           group_num)
                max = cmab_max
                if mabtslr_max > max:
                    max = mabtslr_max
                radar_data['total'] = {}
                for num in range(group_num):
                    radar_data['total'][str(num + 1)] = math.ceil(1.1 * max)
                radar_field_dict[field] = radar_data

            with open(self.output_path + '/radar.json', 'w', encoding='utf-8') as f:
                json.dump(radar_field_dict, f, ensure_ascii=False)
        except Exception as e:
            print(e)
            print("雷达数据生成文件失败")

    @staticmethod
    def build_radar_data_by_recommend_df(recommend_df, project_level_dict, project_field, new_key, group_num: int = 4):
        '''
        根据推荐产品列表df，产品分组编号列表 计算推荐产品列表的雷达数据
        :param recommend_df:
        :param project_level_dict:
        :param project_field:
        :param new_key:
        :return:
        '''
        recommend_df[new_key] = recommend_df[project_field].map(project_level_dict)
        radar_dict = recommend_df.groupby(new_key)[project_field].count().to_dict()
        radar_keys = list(range(1, group_num + 1))
        # 分组不足4个的 自动补零
        new_dict = {}
        max = 0
        for key in radar_keys:
            if key not in radar_dict:
                new_dict[key] = 0
            else:
                new_dict[key] = radar_dict[key]
                if radar_dict[key] > max:
                    max = radar_dict[key]
        return new_dict, max

    @staticmethod
    def assign_equal_parts(df, key: str, sort_key: str, n=4):
        """
        按从小到大排序，直接将数据等分成 n 份
        忽略边界值是否相等
        """
        # 排序
        df_sorted = df.sort_values(by=key).reset_index(drop=True).copy()

        # 每份的大小（向上取整）
        part_size = math.ceil(len(df_sorted) / n)

        # 直接用行号分组
        df_sorted[sort_key] = (df_sorted.index // part_size) + 1

        # 超过 n 的部分修正为 n
        df_sorted.loc[df_sorted[sort_key] > n, sort_key] = n

        return df_sorted


def main():
    parser = argparse.ArgumentParser(description='运行CMAB推荐算法')
    parser.add_argument('--metrics', nargs='+',
                        choices=['accuracy', 'precision', 'recall', 'f1', 'auc'],
                        default=['accuracy', 'precision', 'recall', 'f1', 'auc'],
                        help='选择评估指标，默认输出所有指标')
    args = parser.parse_args()

    # 明确定义约束配置（EHGNN风格）
    constraint_config = {
        'apr_percent': {'category': [1, 2]},  # 连续特征（利率）用分位数约束，四等分，可选填1、2、3、4
        # 'term': {'category': [1, 2]} #连续特征（借款期限）用分位数约束，四等分，可选填1、2、3、4
        # 'REPAYMENT': {'category': [0, 1]} #分类特征（还款方式）直接约束，可选填0、1、2
        # 'level': {'category': [1, 8]}  # 分类特征（风险等级）直接约束，可选填1、2、3、4、5、6、7、8
    }

    # 明确定义筛选条件（EHGNN风格）
    filter_conditions = {
        'gender': None,
        'age': None,
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

    # 明确定义特征体系
    custom_investor_features = [
        'gender', 'age', 'edu_level', 'University', 'CURRENTIDENTITY',
        'credit', 'weightedbidrate_percent', 'baddebts_percent', 'user_invest_count'
    ]
    custom_product_features = [
        'total', 'apr_percent', 'term', 'REPAYMENT', 'level', 'project_invest_count'
    ]

    # 修改CMABRunner初始化（添加约束特征到输出路径）
    constraint_feature = next(iter(constraint_config.keys()))
    config_values = list(constraint_config[constraint_feature].values())[0]

    # 直接定义算法参数（EHGNN风格）
    runner = CMABRunner(
        investor_features=custom_investor_features,
        product_features=custom_product_features,
        user_id=1001,  # 直接指定用户ID
        # 新增约束特征到输出路径
        output_suffix=f"{constraint_feature}_{'_'.join(map(str, config_values))}",

        # 算法超参数配置
        alpha=0.7,  # 探索系数，取值范围(0,1]
        gamma=0.0009,  # q的步长，取值范围(0,0.01]
        lambd=0.0009,  # Q的步长，取值范围(0,0.01]
        delta=0.0009,  # theta的步长，取值范围(0,0.01]
        m=5,  # 每轮推荐数，取值范围[1,50]
        T=1000,  # 总训练轮数，取值范围[100,10000]
        init_theta=20.0  # theta初始值，取值范围[1,100]
    )
    data_path = sys.get_file_path('data/dataset/default/')
    # 运行算法
    result_path = runner.run_cmab(
        filter_conditions,
        data_path=data_path,
        constraint_config=constraint_config,
        metrics_to_calculate=args.metrics
    )

    print(f"结果保存至: {result_path}")


def load_and_prepare(file):
    """
    计算雷达图数据并生成对应的数据文件
    """
    # 读取数据
    df = pd.read_csv(sys.get_file_path(file))

    # 按 level 排序
    df_sorted = df.sort_values(by="level").reset_index(drop=True)

    # 使用 qcut 将 level 分为 4 等分，并打上标签 1,2,3,4
    df_sorted["level_group"] = pd.qcut(df_sorted["level"], q=4, labels=[1, 2, 3, 4])

    # 构建 PROJECTNO -> 等级 的映射字典
    project_level_dict = dict(zip(df_sorted["PROJECTNO"], df_sorted["level_group"]))
    count = df_sorted.groupby("level_group")["PROJECTNO"].count().to_dict()


if __name__ == "__main__":
    main()

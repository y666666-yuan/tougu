import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from core import sys

class EHGNNRunner:
    def __init__(self, investor_features, product_features,
                 hidden_dim=64, lr=0.001, epochs=50, user_id="default_user"):
        self.investor_features = investor_features
        self.product_features = product_features
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.user_id = user_id
        self.full_df = None
        self.output_path =sys.get_file_path(sys.BASE_TASK_PATH,
                          str(sys.ALGORITHM_TYPE_NEW_INVESTOR),
                          str(user_id) + '_' + sys.ALGORITHM_NEW_INVESTOR_COLD_RUN,
                          str(int(time.time())))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.id_to_userno = {}
        self.userno_to_id = {}

    def load_and_preprocess_data(self, data_path, filter_conditions):
        # 数据加载和预处理逻辑保持不变
        if not os.path.exists(data_path):
            print(f"数据路径不存在: {data_path}")
            return None

        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if len(csv_files) != 1:
            print(f"发现 {len(csv_files)} 个CSV文件，应仅存在1个")
            return None

        try:
            self.full_df = pd.read_csv(os.path.join(data_path, csv_files[0]))
        except Exception as e:
            print(f"读取数据失败: {e}")
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
            else:
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

        # 修改需要标准化的连续特征
        cont_features = [
            'age', 'borrowingcredit', 'loancredit', 'weightedbidrate_percent',
            'baddebts_percent', 'user_invest_count', 'total', 'apr_percent',
            'term', 'level', 'project_invest_count'
        ]

        scaler = StandardScaler()
        self.full_df[cont_features] = scaler.fit_transform(self.full_df[cont_features])
        # 计算 credit 特征
        self.full_df['credit'] = (self.full_df['loancredit'] + self.full_df['borrowingcredit']) / 2

        # 创建映射关系
        self.full_df['investor_id'] = self.full_df['userno'].astype('category').cat.codes
        self.full_df['product_id'] = self.full_df['PROJECTNO'].astype('category').cat.codes
        # 创建双向映射字典
        self.id_to_userno = self.full_df[['investor_id', 'userno']].drop_duplicates().set_index('investor_id')[
            'userno'].to_dict()
        self.userno_to_id = {v: k for k, v in self.id_to_userno.items()}
        return self.full_df

    def build_heterogeneous_graph(self):
        # 投资者-产品交互矩阵
        interaction_matrix = pd.pivot_table(
            self.full_df, values='reward', index='investor_id',
            columns='product_id', fill_value=0
        )

        # 投资者社交图
        investor_social = np.zeros((len(interaction_matrix), len(interaction_matrix)))
        for i in range(len(interaction_matrix)):
            vec_i = interaction_matrix.iloc[i].values
            for j in range(i + 1, len(interaction_matrix)):
                vec_j = interaction_matrix.iloc[j].values
                intersection = np.sum(np.minimum(vec_i, vec_j))
                union = np.sum(np.maximum(vec_i, vec_j))
                jaccard = intersection / union if union > 0 else 0
                if jaccard > 0.3:  # 相似度阈值
                    investor_social[i][j] = investor_social[j][i] = 1

        # 产品相似图
        product_similarity = cosine_similarity(interaction_matrix.T)

        return interaction_matrix, investor_social, product_similarity

    # 算法实现部分
    class EHGNN(nn.Module):
        def __init__(self, num_investors, num_products, investor_feat_dim, product_feat_dim, hidden_dim=64,
                     runner=None):
            super().__init__()
            # 投资者嵌入层
            self.investor_embed = nn.Embedding(num_investors, hidden_dim)
            self.investor_fc = nn.Linear(investor_feat_dim, hidden_dim)

            # 产品嵌入层
            self.product_embed = nn.Embedding(num_products, hidden_dim)
            self.product_fc = nn.Linear(product_feat_dim, hidden_dim)

            # 预测网络
            self.predictor = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.runner = runner
            self.hidden_dim = hidden_dim

        def forward(self, investor_ids, product_ids, investor_feats, product_feats):
            # 获取异构图数据
            interaction_matrix, investor_social, product_similarity = self.runner.build_heterogeneous_graph()

            # 投资者特征融合（加入社交信息）
            social_influence = torch.FloatTensor(investor_social[investor_ids.cpu().numpy()].sum(axis=1))
            h_investor = self.investor_embed(investor_ids).squeeze() + self.investor_fc(investor_feats).squeeze()
            h_investor = h_investor + social_influence[:h_investor.size(0)].view(-1, 1).expand(-1, 64).to(
                investor_ids.device)

            # 产品特征融合（加入相似信息）
            similarity_influence = torch.FloatTensor(product_similarity[product_ids.cpu().numpy()].sum(axis=1))
            h_product = self.product_embed(product_ids).squeeze() + self.product_fc(product_feats).squeeze()
            h_product = h_product + similarity_influence[:h_product.size(0)].view(-1, 1).expand(-1, 64).to(
                product_ids.device)

            # 拼接特征
            h_investor = h_investor.unsqueeze(1)
            h_product = h_product.unsqueeze(1)
            combined = torch.cat([h_investor, h_product], dim=1)
            combined = combined.view(-1, 2 * self.hidden_dim)

            return self.predictor(combined)

    class CatBoostNN(nn.Module):
        def __init__(self, input_dim, hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid()
            )

        def forward(self, inv_feats, pro_feats):
            return self.net(torch.cat([inv_feats, pro_feats], 1))

    class NCF(nn.Module):
        def __init__(self, num_users, num_items, hidden):
            super().__init__()
            self.user_embed = nn.Embedding(num_users, hidden)
            self.item_embed = nn.Embedding(num_items, hidden)
            self.mlp = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid()
            )

        def forward(self, user_ids, item_ids):
            u = self.user_embed(user_ids)
            i = self.item_embed(item_ids)
            return self.mlp(torch.cat([u, i], 1))

    def investor_based_cf(self, df):
        """修正后的投资者协同过滤"""
        user_matrix = df.pivot_table(index='investor_id',
                                     columns='product_id',
                                     values='reward').fillna(0)

        # 计算用户相似度
        user_sim = cosine_similarity(user_matrix)
        user_sim = pd.DataFrame(user_sim,
                                index=user_matrix.index,
                                columns=user_matrix.index)

        # 生成推荐
        recommendations = {}
        for user in user_matrix.index:
            # 获取相似用户
            similar_users = user_sim[user].sort_values(ascending=False)[1:6]

            # 加权平均相似用户的偏好
            rec_scores = user_matrix.loc[similar_users.index].mean(axis=0)
            top5 = rec_scores.sort_values(ascending=False).index[:5].tolist()

            recommendations[user] = top5

        return recommendations

    def product_based_cf(self, df):
        """
        产品协同过滤 - 推荐相似产品
        输入: 包含投资者-产品交互数据的DataFrame
        输出: 字典{产品ID: [推荐产品ID1, 推荐产品ID2, ...]}
        """
        # 创建产品-投资者交互矩阵
        item_matrix = df.pivot_table(
            index='product_id',
            columns='investor_id',
            values='reward'
        ).fillna(0)

        # 计算产品相似度矩阵
        similarity = cosine_similarity(item_matrix)

        # 为每个产品生成推荐
        recommendations = {}
        for i, product_id in enumerate(item_matrix.index):
            # 获取相似度分数并排序
            sim_scores = list(enumerate(similarity[i]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # 获取前5个相似产品(排除自己)
            top_similar = [item_matrix.index[j] for j, score in sim_scores[1:6]]
            recommendations[product_id] = top_similar

        return recommendations

    def _get_common_data(self):
        inv_ids = torch.LongTensor(self.full_df['investor_id'].values)
        pro_ids = torch.LongTensor(self.full_df['product_id'].values)
        inv_feats = torch.FloatTensor(self.full_df[self.investor_features].values)
        pro_feats = torch.FloatTensor(self.full_df[self.product_features].values)
        labels = torch.FloatTensor(self.full_df['reward'].values)
        return inv_ids, pro_ids, inv_feats, pro_feats, labels

    def _save_recommendations(self, recommendations, test_set, algo_name, result_dir):
        # 获取原始userno和PROJECTNO映射
        user_map = self.full_df[['investor_id', 'userno']].drop_duplicates()
        product_map = self.full_df[['product_id', 'PROJECTNO']].drop_duplicates()
        missing_products = set()
        for user_id, products in recommendations.items():
            for product_id in products[:5]:
                if product_id not in product_map['product_id'].values:
                    missing_products.add(product_id)

        if missing_products:
            print(f"Warning: {len(missing_products)} unique product_ids not found in product_map")
            print(f"Missing product_ids: {sorted(missing_products)}")

        output = []

        for user_id, products in recommendations.items():
            actual = test_set[test_set['investor_id'] == user_id]['product_id'].tolist()
            try:
                userno = user_map[user_map['investor_id'] == user_id]['userno'].values[0]
                for product_id in products:
                    try:
                        projectno = product_map[product_map['product_id'] == product_id]['PROJECTNO'].values[0]
                        output.append({
                            'userno': userno,
                            'Projectno': projectno
                        })
                    except IndexError:
                        print(f"Warning: product_id {product_id} not found in product_map")
            except IndexError:
                print(f"Warning: investor_id {user_id} not found in user_map")

        # 保存到CSV
        if output:
            pd.DataFrame(output).to_csv(
                os.path.join(result_dir, f'{algo_name}_recommendations.csv'),
                index=False
            )
        print(f'所有结果已保存至: {result_dir}')

    def _train_neural_model(self, model, inv_ids, pro_ids, inv_feats, pro_feats, labels):
        interaction_matrix, investor_social, product_similarity = self.build_heterogeneous_graph()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = model(inv_ids, pro_ids, inv_feats, pro_feats) if isinstance(model, self.EHGNN) \
                else model(inv_feats, pro_feats) if isinstance(model, self.CatBoostNN) \
                else model(inv_ids, pro_ids)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(inv_ids, pro_ids, inv_feats, pro_feats).squeeze() if isinstance(model, self.EHGNN) \
                else model(inv_feats, pro_feats).squeeze() if isinstance(model, self.CatBoostNN) \
                else model(inv_ids, pro_ids).squeeze()
        return preds

    def _run_cf_model(self, cf_type):
        interaction = pd.pivot_table(self.full_df, values='reward',
                                     index='investor_id', columns='product_id',
                                     fill_value=0)
        if cf_type == 'investor':
            sim = cosine_similarity(interaction)
            preds = np.dot(sim, interaction.values) / (np.sum(np.abs(sim), axis=1) + 1e-9)[:, None]
        else:  # product
            sim = cosine_similarity(interaction.T)
            preds = np.dot(interaction.values, sim) / (np.sum(np.abs(sim), axis=1) + 1e-9)
        return (preds.flatten() > 0.5).astype(int)

    def execute_algorithm(self, algo_type):
        inv_ids, pro_ids, inv_feats, pro_feats, labels = self._get_common_data()

        if algo_type == 'ehgnn':
            model = self.EHGNN(
                num_investors=self.full_df['investor_id'].nunique(),
                num_products=self.full_df['product_id'].nunique(),
                investor_feat_dim=len(self.investor_features),
                product_feat_dim=len(self.product_features),
                hidden_dim=self.hidden_dim,
                runner=self
            )
            raw_output = self._train_neural_model(model, inv_ids, pro_ids, inv_feats, pro_feats, labels)
            pred_scores = raw_output.detach().numpy()
        elif algo_type == 'catboostnn':
            model = self.CatBoostNN(
                input_dim=len(self.investor_features) + len(self.product_features),
                hidden=self.hidden_dim
            )
            raw_output = self._train_neural_model(model, inv_ids, pro_ids, inv_feats, pro_feats, labels)
            pred_scores = raw_output.detach().numpy()
        elif algo_type == 'ncf':
            model = self.NCF(
                num_users=self.full_df['investor_id'].nunique(),
                num_items=self.full_df['product_id'].nunique(),
                hidden=self.hidden_dim
            )
            raw_output = self._train_neural_model(model, inv_ids, pro_ids, inv_feats, pro_feats, labels)
            pred_scores = raw_output.detach().numpy()
        elif 'cf' in algo_type:
            preds = self._run_cf_model(algo_type.split('_')[0])
            pred_scores = preds.astype(float)

        recommendations = {}
        for investor_id in self.full_df['investor_id'].unique():
            investor_mask = (self.full_df['investor_id'] == investor_id)
            investor_scores = pred_scores[investor_mask]
            sorted_indices = np.argsort(-investor_scores)[:5]
            recommended_projects = self.full_df[investor_mask].iloc[sorted_indices]['product_id'].tolist()
            recommendations[investor_id] = recommended_projects

        return recommendations

    def preprocess_data(self, raw_df, filter_conditions):
        """
        数据预处理:
        1. 应用筛选条件
        2. 处理缺失值
        3. 特征工程
        """
        # 应用筛选条件
        filtered_df = raw_df.query(filter_conditions) if filter_conditions else raw_df

        # 简单示例: 填充缺失值
        processed_df = filtered_df.fillna({
            'reward': 0,
            'investor_feature1': filtered_df['investor_feature1'].mean(),
            'product_feature1': filtered_df['product_feature1'].median()
        })
        cont_features = [
            'age', 'borrowingcredit', 'loancredit', 'weightedbidrate_percent',
            'baddebts_percent', 'user_invest_count', 'total', 'apr_percent',
            'term', 'level', 'project_invest_count'
        ]
        scaler = StandardScaler()
        self.full_df[cont_features] = scaler.fit_transform(self.full_df[cont_features])
        print(f"原始数据中的产品ID数量: {raw_df['PROJECTNO'].nunique()}")
        print(f"处理后数据中的产品ID数量: {processed_df['PROJECTNO'].nunique()}")
        self.interaction_matrix, self.investor_social, self.product_similarity = self.build_heterogeneous_graph()
        return self.full_df

    def run_comparison(self, data_path, filter_conditions):
        algorithms = [
            'ehgnn',
            'catboostnn',
            'investor_cf',
            'product_cf',
            'ncf'
        ]

        if self.load_and_preprocess_data(data_path, filter_conditions) is None:
            return None
        # 1. 读取并处理数据
        processed_df = self.full_df

        # 2. 划分训练集和测试集
        test_set = processed_df.sample(frac=0.2)
        train_set = processed_df.drop(test_set.index)
        results = {}
        # 定义算法列表
        algo_list = [
            ('ehgnn', lambda df: self.execute_algorithm('ehgnn')),
            ('catboostnn', lambda df: self.execute_algorithm('catboostnn')),
            ('investor_cf', self.investor_based_cf),
            ('product_cf', self.product_based_cf),
            ('ncf', lambda df: self.execute_algorithm('ncf'))]
        raw_scores = {}
        for algo_name, algo_func in algo_list:
            # 获取推荐结果
            recs = algo_func(processed_df)
            self._save_recommendations(recs, test_set, algo_name, self.output_path)
            # 计算加权命中率
            weighted_score = 0
            for user, top5 in recs.items():
                actual = test_set[test_set['investor_id'] == user]['product_id'].tolist()
                for rank, product in enumerate(top5, 1):
                    if product in actual:
                        weighted_score += 1 / rank
            raw_scores[algo_name] = weighted_score / len(recs) if recs else 0

            # 找出最大加权命中率
            max_hit_rate = max(raw_scores.values()) if raw_scores else 0

            # 根据最大命中率确定共用缩放因子
            if max_hit_rate < 0.5:
                scale_factor = 2.0
            elif max_hit_rate < 0.666:
                scale_factor = 1.5
            elif max_hit_rate < 0.8:
                scale_factor = 1.25
            else:
                scale_factor = 1.0

            # 应用缩放因子到所有算法
            for algo_name in raw_scores:
                scaled_score = min(raw_scores[algo_name] * scale_factor, 1.0)
                results[algo_name] = {
                    'Algorithm': algo_name,
                    'Weighted_HitRate': scaled_score,
                    'Original_HitRate': raw_scores[algo_name],
                    'Scale_Factor': scale_factor
                }
        comparison_df = pd.DataFrame(results.values())
        comparison_df.to_csv(os.path.join(self.output_path, 'comparison_results.csv'), index=False)

        print(f'所有结果已保存至: {self.output_path}')
        return self.output_path,None


if __name__ == '__main__':
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

    runner = EHGNNRunner(
        investor_features=custom_investor_features,
        product_features=custom_product_features,
        user_id="3",
        hidden_dim=64,
        lr=0.001,
        epochs=200
    )

    result_file = runner.run_comparison(
        data_path=sys.get_file_path("data/dataset/default"),
        filter_conditions=filter_conditions  # 这里传入filter_conditions
    )
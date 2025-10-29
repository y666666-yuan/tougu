import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from core import sys


class EHGNNRunner:
    def __init__(self, investor_features=None, product_features=None, user_id=None,
                 hidden_dim=64, steps=50, lr=0.001, epochs=200):
        self.full_df = None
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
        self.id_to_userno = None
        self.userno_to_id = None
        self.model = None

        self.output_path = sys.get_file_path(sys.BASE_TASK_PATH,
                                             str(sys.ALGORITHM_TYPE_BASE),
                                             str(user_id) + '_' + sys.ALGORITHM_BASE_NAME_EHGNN,
                                             str(int(time.time())))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.lr = lr
        self.epochs = epochs

    # ----------------------
    # 1. 数据加载与预处理
    # ----------------------
    def load_and_preprocess_data(self, data_path, filter_conditions=None):
        """加载并预处理数据，返回合并后的DataFrame和映射字典"""
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

        # 生成映射关系
        self.full_df['investor_id'] = self.full_df['userno'].astype('category').cat.codes
        self.full_df['product_id'] = self.full_df['PROJECTNO'].astype('category').cat.codes

        # 创建双向映射字典
        self.id_to_userno = self.full_df[['investor_id', 'userno']].drop_duplicates().set_index('investor_id')[
            'userno'].to_dict()
        self.userno_to_id = {v: k for k, v in self.id_to_userno.items()}

        return self.full_df

    # ----------------------
    # 2. 异构图构建
    # ----------------------
    def build_heterogeneous_graph(self):
        """构建投资者-产品交互矩阵、投资者社交图和产品相似图"""
        if self.full_df is None:
            print("请先加载并预处理数据。")
            return None
        # 投资者-产品交互矩阵
        interaction_matrix = pd.pivot_table(
            self.full_df, values='reward', index='investor_id',
            columns='product_id', fill_value=0
        )

        # 投资者社交图（基于共同购买）
        investor_social = np.zeros((len(interaction_matrix), len(interaction_matrix)))
        for i in tqdm(range(len(interaction_matrix)), desc="构建投资者社交图"):
            vec_i = interaction_matrix.iloc[i].values
            for j in range(i + 1, len(interaction_matrix)):
                vec_j = interaction_matrix.iloc[j].values
                jaccard = np.sum(vec_i & vec_j) / np.sum(vec_i | vec_j)
                if jaccard > 0.3:  # 相似度阈值
                    investor_social[i][j] = investor_social[j][i] = 1

        # 产品相似图（余弦相似度）
        product_similarity = cosine_similarity(interaction_matrix.T)

        return interaction_matrix, investor_social, product_similarity

    # ----------------------
    # 3. EHGNN模型定义
    # ----------------------
    class EHGNN(nn.Module):
        def __init__(self, num_investors, num_products, investor_feat_dim, product_feat_dim, hidden_dim=64):
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
                nn.Linear(hidden_dim, 1),  # 保持输出维度为(batch_size, 1)
                nn.Sigmoid()
            )

        def forward(self, investor_ids, product_ids, investor_feats, product_feats):
            # 投资者特征融合
            h_investor = self.investor_embed(investor_ids) + self.investor_fc(investor_feats)

            # 产品特征融合
            h_product = self.product_embed(product_ids) + self.product_fc(product_feats)

            # 拼接特征
            combined = torch.cat([h_investor, h_product], dim=1)

            # 预测输出（保持二维）
            output = self.predictor(combined)

            # 维度验证
            assert output.ndim == 2, f"输出应为二维张量，当前维度：{output.ndim}"
            assert output.shape[1] == 1, f"输出第二维应为1，当前形状：{output.shape}"

            return output

    # ----------------------
    # 4. 积分梯度计算类
    # ----------------------
    class IntegratedGradients:
        def __init__(self, model, steps=50):
            self.model = model
            self.model.eval()  # 固定模型为评估模式
            self.steps = steps

        def compute_integrated_gradients(self, investor_ids, product_ids, investor_feats, product_feats):
            """计算投资者和产品特征的积分梯度"""
            # 生成基线输入
            baseline_inv = torch.zeros_like(investor_feats)
            baseline_pro = torch.zeros_like(product_feats)

            # 存储梯度
            gradients = []

            # 生成插值路径
            for alpha in tqdm(np.linspace(0, 1, self.steps + 1), desc="计算积分梯度"):
                # 插值输入
                current_inv = baseline_inv + alpha * (investor_feats - baseline_inv)
                current_pro = baseline_pro + alpha * (product_feats - baseline_pro)
                current_inv.requires_grad_(True)
                current_pro.requires_grad_(True)

                # 前向传播
                output = self.model(
                    investor_ids=investor_ids,
                    product_ids=product_ids,
                    investor_feats=current_inv,
                    product_feats=current_pro
                )

                # 反向传播
                self.model.zero_grad()
                output.sum().backward()  # 对输出求和

                # 收集梯度
                grad_inv = current_inv.grad.detach().cpu().numpy()
                grad_pro = current_pro.grad.detach().cpu().numpy()
                gradients.append(np.concatenate([grad_inv, grad_pro], axis=1))

            # 计算积分梯度
            avg_grad = np.mean(gradients[:-1], axis=0)
            delta_inv = (investor_feats - baseline_inv).detach().cpu().numpy()
            delta_pro = (product_feats - baseline_pro).detach().cpu().numpy()
            integrated_grad = np.concatenate([delta_inv, delta_pro], axis=1) * avg_grad

            return integrated_grad.mean(axis=0)  # 返回特征平均贡献

    # ----------------------
    # 5. 训练与评估流程
    # ----------------------
    def train_model(self, metrics_to_calculate):
        """模型训练流程"""
        if self.full_df is None:
            print("请先加载并预处理数据。")
            return None
        num_investors = self.full_df['investor_id'].nunique()
        num_products = self.full_df['product_id'].nunique()

        # 初始化模型
        self.model = self.EHGNN(
            num_investors=num_investors,
            num_products=num_products,
            investor_feat_dim=len(self.investor_features),
            product_feat_dim=len(self.product_features),
            hidden_dim=self.hidden_dim
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        # # added by jiang for 先转换为 numpy数组再创建张量避免性能警告
        investor_ids = torch.LongTensor(np.array(self.full_df['investor_id'].values))
        product_ids = torch.LongTensor(np.array(self.full_df['product_id'].values))
        investor_feats = torch.FloatTensor(np.array(self.full_df[self.investor_features].values))
        product_feats = torch.FloatTensor(np.array(self.full_df[self.product_features].values))
        labels = torch.FloatTensor(np.array(self.full_df['reward'].values))

        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(investor_ids, product_ids, investor_feats, product_feats)
            loss = criterion(outputs.squeeze(), labels)  # 输出维度调整
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # 模型评估
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(investor_ids, product_ids, investor_feats, product_feats)
            preds = (outputs.squeeze() > 0.5).int().numpy()
            probs = outputs.squeeze().numpy()

        # 计算评估指标
        metrics = {}
        if 'accuracy' in metrics_to_calculate:
            metrics['accuracy'] = accuracy_score(labels.numpy(), preds)
        if 'precision' in metrics_to_calculate:
            metrics['precision'] = precision_score(labels.numpy(), preds)
        if 'recall' in metrics_to_calculate:
            metrics['recall'] = recall_score(labels.numpy(), preds)
        if 'f1' in metrics_to_calculate:
            metrics['f1'] = f1_score(labels.numpy(), preds)
        if 'auc' in metrics_to_calculate:
            metrics['auc'] = roc_auc_score(labels.numpy(), probs)
        print("\n模型评估指标:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        return self.model, metrics

    # ----------------------
    # 6. 结果生成（修改映射关系）
    # ----------------------
    def generate_results(self, target_investor_id):
        """生成三个结果表格，使用原始userno"""
        if self.full_df is None or self.model is None:
            print("请先加载并预处理数据，训练模型。")
            return None
        # 获取目标投资者信息
        target_userno = self.id_to_userno[target_investor_id]
        target_data = self.full_df[self.full_df['investor_id'] == target_investor_id]

        # 表1：Top-5相似投资者（显示原始userno）
        investor_matrix = self.full_df.groupby('investor_id')[self.investor_features].mean()
        similarity = cosine_similarity(investor_matrix)
        top5_indices = np.argsort(-similarity[target_investor_id])[1:6]

        table1 = pd.DataFrame({
            'Userno': [self.id_to_userno[i] for i in top5_indices],
            'Similarity': [similarity[target_investor_id][i] for i in top5_indices]
        })

        # 表2：Top-5推荐产品（保持不变）
        all_products = self.full_df['product_id'].unique()
        with torch.no_grad():
            investor_ids = torch.LongTensor(
                np.array([target_investor_id] * len(all_products)))  # added by jiang for 先转换为 numpy数组再创建张量避免性能警告
            product_ids = torch.LongTensor(all_products)
            investor_feat_mean = target_data[self.investor_features].mean().values
            investor_feats = torch.FloatTensor(
                np.array([investor_feat_mean] * len(all_products)))  # added by jiang for 先转换为 numpy数组再创建张量避免性能警告
            product_feats = torch.FloatTensor(
                self.full_df.groupby('product_id')[self.product_features].mean().loc[all_products].values)
            scores = self.model(investor_ids, product_ids, investor_feats, product_feats).squeeze().numpy()

        # 过滤已购买产品
        purchased = self.full_df[(self.full_df['investor_id'] == target_investor_id) & (self.full_df['reward'] == 1)][
            'product_id'].unique()
        mask = ~np.isin(all_products, purchased)
        top5_products = all_products[mask][np.argsort(-scores[mask])[:5]]
        table2 = pd.DataFrame({
            'Projectno': [self.full_df[self.full_df['product_id'] == p]['PROJECTNO'].iloc[0] for p in top5_products],
            'Score': scores[mask][np.argsort(-scores[mask])[:5]]
        })

        # 表3：特征贡献值（保持不变）
        ig = self.IntegratedGradients(self.model)
        investor_ids = torch.LongTensor(target_data['investor_id'].values)
        product_ids = torch.LongTensor(target_data['product_id'].values)
        investor_feats = torch.FloatTensor(target_data[self.investor_features].values)
        product_feats = torch.FloatTensor(target_data[self.product_features].values)

        attributions = ig.compute_integrated_gradients(
            investor_ids=investor_ids,
            product_ids=product_ids,
            investor_feats=investor_feats,
            product_feats=product_feats
        )

        feature_names = self.investor_features + self.product_features
        table3 = pd.DataFrame({
            'Feature': feature_names,
            'Contribution': attributions
        }).sort_values('Contribution', ascending=False)

        return table1, table2, table3, target_userno

    # 主函数封装
    def run_ehgnn(self, filter_conditions, data_path="dataset/default/", metrics_to_calculate=None):

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

        # 将模型评估指标保存为 CSV 文件
        # 将所有指标值转换为浮点数并保留 4 位小数
        metrics = {k: float(f"{v:.4f}") for k, v in metrics.items()}
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_path = os.path.join(self.output_path, "model_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"模型评估指标已保存到: {metrics_path}")

        print("所有结果保存完成")
        return self.output_path, None


def main1():
    parser = argparse.ArgumentParser(description='运行 EHGNN 模型并选择输出的评估指标')
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
    ehgnn_runner = EHGNNRunner(
        investor_features=custom_investor_features,
        product_features=custom_product_features,
        hidden_dim=64,  # 超参数名称:隐藏层维度; 取值范围:正整数; 常见取值:16、32、64、128、256等.
        steps=50,  # 超参数名称:积分梯度计算步数; 取值范围:正整数; 常见取值:20到100之间的整数.
        lr=0.001,  # 超参数名称:学习率; 取值范围:0到1之间的浮点数; 常见取值:0.001到0.1之间.
        epochs=200,  # 超参数名称:训练轮数; 取值范围:1到正无穷的整数; 常见取值:10到1000之间的整数.
        user_id=2
    )
    data_path = sys.get_file_path('dataset/default/')
    result_path, err_msg = ehgnn_runner.run_ehgnn(filter_conditions=filter_conditions,
                                                  data_path=data_path)
    if err_msg is None:
        print(f"实验结果跑完，结果路径：{result_path}")
    else:
        print("实验报错")


if __name__ == '__main__':
    main1()

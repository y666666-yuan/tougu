import pandas as pd
import os
import json
from core import sys
from core.base_network import BaseNetworkNode, BaseNetworkEdge, BaseNetwork
import numpy as np
"""
此模块主要用于处理实验前后的文件和实验所用数据集，
"""


class TaskTool:
    def __init__(self, task_id: str, user_id: int, algorithm_type: int, algorithm_name: str):
        self.task_id = task_id
        self.user_id = user_id
        self.algorithm_type = algorithm_type
        self.algorithm_name = algorithm_name
        self.task_path = sys.BASE_TASK_PATH + str(self.algorithm_type) + "/" + str(
            self.user_id) + "_" + self.algorithm_name + "/" + self.task_id
        if not os.path.exists(self.task_path):
            raise Exception("任务未找到")
        self._check_algorithm_type()

    def _check_algorithm_type(self):
        """
        检查算法类型是否支持
        :return:
        """
        if self.algorithm_name is None:
            raise Exception(f"系统不支持的算法{str(self.algorithm_type)}_{self.algorithm_name}")
        if self.algorithm_type < 1 or self.algorithm_type > 4:
            raise Exception(f"系统不支持的算法{str(self.algorithm_type)}_{self.algorithm_name}")
        if self.algorithm_type == sys.ALGORITHM_TYPE_BASE:
            if self.algorithm_name not in sys.ALGORITHM_BASE_NAMES:
                raise Exception(f"系统不支持的算法{str(self.algorithm_type)}_{self.algorithm_name}")
        elif self.algorithm_type == sys.ALGORITHM_TYPE_CONSTRAINT:
            if self.algorithm_name not in sys.ALGORITHM_CONSTRAINT_NAMES:
                raise Exception(f"系统不支持的算法{str(self.algorithm_type)}_{self.algorithm_name}")
        elif self.algorithm_type == sys.ALGORITHM_TYPE_NEW_INVESTOR:
            if self.algorithm_name not in sys.ALGORITHM_NEW_INVESTOR_NAMES:
                raise Exception(f"系统不支持的算法{str(self.algorithm_type)}_{self.algorithm_name}")
        else:
            raise Exception(f"系统不支持的算法{str(self.algorithm_type)}_{self.algorithm_name}")

    def get_processed_investors(self, page_size: int, page_num: int) -> (dict, str):
        """
        分页获取数据集中的投资者编号列表
        :param page_size 页大小
        :param page_num 页码
        :return:
        """
        file_path = sys.get_file_path(
            sys.BASE_TASK_PATH,
            str(self.algorithm_type),
            str(self.user_id) + "_" + self.algorithm_name,
            self.task_id,
            "processed_data.csv"
        )
        if not os.path.exists(file_path):
            return None, "数据未找到"
        df = pd.read_csv(file_path)
        unique_users = df['userno'].unique()
        if unique_users is None:
            return None, "数据未找到"

        user_no_list = unique_users.tolist()

        result_data = {
            "total": len(user_no_list),
            "items": []
        }
        start = (page_num - 1) * page_size
        end = page_num * page_size
        if start < 0:
            start = 0
        if end > len(user_no_list):
            end = len(user_no_list)

        user_no_list = user_no_list[start:end]
        for user_no in user_no_list:
            item = {
                "user_no": user_no,

            }
            recommend_file = sys.get_file_path(
                self.task_path,
                "userno_" + str(user_no) + "_recommended_products.csv"
            )
            df = pd.read_csv(recommend_file)
            recommend_products = df['Projectno'].unique()

            if recommend_products is not None:
                item["recommend_products"] = recommend_products.tolist()
                result_data["items"].append(item)
        return result_data, None

    def get_processed_investor_detail(self, user_no: int):
        """
        获取用户某一次实验其中一个投资者详情
        :param user_no: 投资者编号
        :return:
        """
        # 获取基本信息
        base_file = sys.get_file_path(
            self.task_path,
            "processed_data.csv")
        base_df = pd.read_csv(base_file)
        base_df = base_df[base_df['userno'] == user_no]
        base_first_row = base_df.iloc[0]
        base = {
            "user_no": str(int(base_first_row[0])),
            "gender": '男' if base_first_row[8] == 0 else '女',
            "age": str(int(base_first_row[4])),
            "total_bid_number": str(int(base_first_row[5])),
            "edu_level": DataSetTool.query_cn_name("edu_level", int(base_first_row[9])),
            "university": DataSetTool.query_cn_name("University", int(base_first_row[1])),
            "current_identity": DataSetTool.query_cn_name("CURRENTIDENTITY", int(base_first_row[10]))
        }

        # 获取相似投资者
        similar_investors = []
        if self.algorithm_type == sys.ALGORITHM_TYPE_BASE:
            similar_file = sys.get_file_path(
                self.task_path,
                "userno_" + str(user_no) + "_similar_investors.csv")
            similar_df = pd.read_csv(similar_file)

            for row in similar_df.itertuples(index=False):
                similar_investors.append({"user_no": str(int(row.Userno)), "score": row.Similarity})

        # 获取推荐产品
        recommend_file = sys.get_file_path(
            self.task_path,
            "userno_" + str(user_no) + "_recommended_products.csv")
        recommend_df = pd.read_csv(recommend_file)
        recommended_products = []
        for row in recommend_df.itertuples(index=False):
            product = {}
            if self.algorithm_type == sys.ALGORITHM_TYPE_BASE:
                product = {
                    "product_no": int(row.Projectno),
                    "score": row.Score,
                }
            elif self.algorithm_type == sys.ALGORITHM_TYPE_CONSTRAINT:
                product = {
                    "product_no": int(row.Projectno),
                }
            product_row = base_df[base_df['PROJECTNO'] == row.Projectno].iloc[0]
            product['total'] = str(int(product_row[14]))
            product['apr_percent'] = str(product_row[15])
            product['term'] = str(int(product_row[17]))
            product['level'] = str(int(product_row[18]))
            product['bidders'] = str(int(product_row[16]))
            product['repayment'] = DataSetTool.query_cn_name("REPAYMENT", int(base_first_row[13]))
            recommended_products.append(product)

        choose_features = []
        if not os.path.exists(self.task_path):
            return None, "任务未找到"
        try:
            with open(os.path.join(self.task_path, "request_params.json"), 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                choose_features.extend(json_data.get("investor_features"))
                choose_features.extend(json_data.get("product_features"))
        except Exception as e:
            print(e)
            print("请求参数文件未找到")

        # 投资者属性评分
        product_feature = []
        investor_feature = []
        if self.algorithm_type == sys.ALGORITHM_TYPE_BASE:
            feature_file = sys.get_file_path(
                self.task_path,
                "userno_" + str(user_no) + "_feature_contributions.csv"
            )
            feature_df = pd.read_csv(feature_file)
            # 可选投资者特征
            all_investor_features = [
                'gender', 'age', 'edu_level', 'University', 'CURRENTIDENTITY',
                'credit', 'weightedbidrate_percent',
                'baddebts_percent', 'user_invest_count'
            ]
            # 可选产品特征
            all_product_features = [
                'total', 'apr_percent', 'term', 'REPAYMENT', 'level',
                'project_invest_count'
            ]
            for row in feature_df.itertuples(index=False):
                if row.Feature in all_investor_features and row.Feature in choose_features:
                    investor_feature.append(
                        {"feature": row.Feature, "score": row.Contribution,
                         "label": DataSetTool.query_feature_cn(row.Feature)})
                if row.Feature in all_product_features and row.Feature in choose_features:
                    product_feature.append(
                        {"feature": row.Feature, "score": row.Contribution,
                         "label": DataSetTool.query_feature_cn(row.Feature)})

        detail = {
            "similar_investors": similar_investors,
            "base": base,
            "recommended_products": recommended_products,
            "product_features": product_feature,
            "investor_features": investor_feature
        }
        return detail

    @classmethod
    def get_algorithm_list(cls, algorithm_type):
        """
        算法参数
        :return:
        """
        if algorithm_type == sys.ALGORITHM_TYPE_BASE:
            return sys.ALGORITHM_BASE_PARAMS
        elif algorithm_type == sys.ALGORITHM_TYPE_CONSTRAINT:
            return sys.ALGORITHM_CONSTRAINT_PARAMS
        elif algorithm_type == sys.ALGORITHM_TYPE_NEW_INVESTOR:
            return sys.ALGORITHM_NEW_INVESTOR_PARAMS

    def get_model_metrics(self):
        """
        获取某一次任务算法评价属性
        :return:
        """
        if self.algorithm_type == sys.ALGORITHM_TYPE_BASE:
            params, error = self.get_task_params()
            if error is not None:
                return None, error
            result_data = {}
            result_data['algorithm_params'] = params.get("algorithm_params")
            metrics_df = pd.read_csv(os.path.join(self.task_path, "model_metrics.csv"))
            metrics = {}
            for row in metrics_df.itertuples(index=False):
                metrics[str(row.Metric)] = row.Value
            result_data['metrics'] = metrics
            return result_data, None
        elif self.algorithm_type == sys.ALGORITHM_TYPE_CONSTRAINT:
            params = {}
            try:
                with open(os.path.join(self.task_path, "radar.json"), 'r', encoding='utf-8') as file:
                    params = json.load(file)
            except Exception as e:
                print(e)
                print("请求参数文件未找到")
                return None, "请求参数文件未找到"

            return params, None
        elif self.algorithm_type == sys.ALGORITHM_TYPE_NEW_INVESTOR:
            params = {}
            metrics_df = pd.read_csv(sys.get_file_path(self.task_path, "comparison_results.csv"))
            rows_as_dicts = metrics_df.to_dict(orient='records')
            return rows_as_dicts, None
        else:
            return None, "不支持的算法"

    def get_task_params(self):
        """
        获取任务输入参数
        :return:
        """

        params = {}
        try:
            with open(os.path.join(self.task_path, "request_params.json"), 'r', encoding='utf-8') as file:
                params = json.load(file)
        except Exception as e:
            print(e)
            print("请求参数文件未找到")
            return None, "请求参数文件未找到"

        return params, None


class DataSetTool:
    def __init__(self, dataset: str, filter_conditions: dict):
        self.dataset = dataset
        self.filter_conditions = filter_conditions
        self.data_path = sys.get_file_path(sys.BASE_DATA_SET_PATH, self.dataset)
        if not os.path.exists(self.data_path):
            raise Exception(f"数据集不存在{self.dataset}")

    def build_investor_product_network(self):
        """
        构建投资者-产品网络
        :return:
        """
        full_df, err_msg = self._filter_dataset_fast()
        if err_msg is not None:
            return None, err_msg
        node_ids = set()
        nodes = []
        edge_ids = set()
        edges = []
        if full_df is not None:
            for row in full_df.itertuples(index=False):
                u_id = "u_" + str(int(row.userno))
                if not u_id in node_ids:
                    nodes.append(self._build_network_node_investor(row=row, id=u_id))
                    node_ids.append(u_id)

                p_id = "p_" + str(int(row.PROJECTNO))
                if not p_id in node_ids:
                    nodes.append(self._build_network_node_product(row=row, id=p_id))
                    node_ids.append(p_id)

                e_id = u_id + "_" + p_id
                if not e_id in edge_ids:
                    edges.append(BaseNetworkEdge(id=e_id, edge_from=u_id, edge_to=p_id, label="投资").__dict__)
                    edge_ids.append(e_id)
        return BaseNetwork(nodes=nodes, edges=edges).__dict__, None

    def build_investor_product_network_fast(self):
        """
        构建投资者-产品网络
        :return:
        """
        full_df, err_msg = self._filter_dataset_fast()
        if err_msg: return None, err_msg
        if full_df is None or full_df.empty: return {"nodes": [], "edges": []}, None

        # 生成唯一 user、project
        unique_users = full_df.drop_duplicates('userno')
        unique_projects = full_df.drop_duplicates('PROJECTNO')

        # 投资者节点
        nodes_user = [
            self._build_network_node_investor(row, f"u_{int(row.userno)}")
            for row in unique_users.itertuples(index=False)
        ]

        # 3. 产品节点
        nodes_product = [
            self._build_network_node_product(row, f"p_{int(row.PROJECTNO)}")
            for row in unique_projects.itertuples(index=False)
        ]

        # 边：只保留唯一 (user, project)
        upairs = full_df[['userno', 'PROJECTNO']].drop_duplicates()
        edges = [
            BaseNetworkEdge(id=f"u_{int(r.userno)}_p_{int(r.PROJECTNO)}",
                            edge_from=f"u_{int(r.userno)}",
                            edge_to=f"p_{int(r.PROJECTNO)}",
                            label="投资").__dict__
            for r in upairs.itertuples(index=False)
        ]
        return BaseNetwork(nodes=[*nodes_user, *nodes_product], edges=edges).__dict__, None

    def _filter_dataset(self):
        """
        根据 属性 和属性值进行过滤
        :return:
        """

        csv_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith('.csv')]
        df = pd.read_csv(csv_files[0])
        for feature, value in self.filter_conditions.items():
            if value is None:
                continue

            if isinstance(value, (int, float)):
                df = df[df[feature] == value]
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
                df = df
            elif isinstance(value, list):
                try:
                    df = df[df[feature].isin(values=value)]
                except Exception as e:
                    print(e)
                    print(f"\n过滤值错误! {feature}:{value} ")
                df = df
            else:
                df = df[df[feature].astype(str) == str(value)]
        return df, None

    def _filter_dataset_fast(self):
        """一次性链式过滤 + 指定dtype"""
        csv_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith('.csv')]
        if not csv_files:
            return None, f"未找到 CSV: {self.dataset}"

        # 只加载必要列 & 指定dtype
        dtypes = {
            "userno": "int64", "PROJECTNO": "int64",
            # 其他可能的列自行加上
        }
        df = pd.read_csv(csv_files[0],  engine="c")

        # 构造query表达式
        exprs = []
        for feature, value in self.filter_conditions.items():
            if value is None: continue
            if isinstance(value, (int, float, str)):
                exprs.append(f"`{feature}` == @value")
            elif isinstance(value, dict):
                if 'min' in value: exprs.append(f"`{feature}` >= {int(value['min'])}")
                if 'max' in value: exprs.append(f"`{feature}` < {int(value['max'])}")
            elif isinstance(value, list):
                exprs.append(f"`{feature}` in @value")

        if exprs:
            df = df.query(" & ".join(exprs))

        return df, None

    def build_investor_product_network(self):
        """
        构建投资者-产品网络
        :return:
        """
        full_df, err_msg = self._filter_dataset_fast()
        if err_msg is not None:
            return None, err_msg
        node_ids = []
        nodes = []
        edge_ids = []
        edges = []
        if full_df is not None:
            for row in full_df.itertuples(index=False):
                u_id = "u_" + str(int(row.userno))
                if u_id not in node_ids:
                    nodes.append(self._build_network_node_investor(row=row, id=u_id))
                    node_ids.append(u_id)

                p_id = "p_" + str(int(row.PROJECTNO))
                if p_id not in node_ids:
                    nodes.append(self._build_network_node_product(row=row, id=p_id))
                    node_ids.append(p_id)

                e_id = u_id + "_" + p_id
                if e_id not in edge_ids:
                    edges.append(BaseNetworkEdge(id=e_id, edge_from=u_id, edge_to=p_id, label="投资").__dict__)
                    edge_ids.append(e_id)
        return BaseNetwork(nodes=nodes, edges=edges).__dict__, None

    def build_investor_investor_network(self):
        """
        构建投资者-投资者网络
        :return:
        """
        similar_path = sys.get_file_path(sys.BASE_SYS_FILE_PATH, self.dataset + "_investor_similar.csv")
        if not os.path.exists(similar_path):
            return None, f"数据集不存在{self.dataset + '_investor_similar.csv'}"
        full_df, err_msg = self._filter_dataset_fast()
        if err_msg is not None:
            return None, err_msg
        unique_users = full_df['userno'].unique()
        node_ids = []
        nodes = []
        edges = []
        if unique_users is not None:
            user_no_list = unique_users.tolist()
            for row in full_df.itertuples(index=False):
                u_id = "u_" + str(int(row.userno))
                if not u_id in node_ids:
                    nodes.append(self._build_network_node_investor(row=row, id=u_id))
                    node_ids.append(u_id)

            edges = self._get_similar_edges(similar_path, user_no_list, 'u_')
        return BaseNetwork(nodes=nodes, edges=edges).__dict__, None

    def build_product_product_network(self):
        """
        构建产品-产品网络
        :return:
        """
        similar_path = sys.get_file_path(sys.BASE_SYS_FILE_PATH + self.dataset + "_project_similar.csv")
        if not os.path.exists(similar_path):
            return None, f"数据集不存在{self.dataset + '_project_similar.csv'}"
        full_df, err_msg = self._filter_dataset()
        if err_msg is not None:
            return None, err_msg
        unique_users = full_df['PROJECTNO'].unique()
        node_ids = []
        nodes = []
        edges = []
        if unique_users is not None:
            user_no_list = unique_users.tolist()
            for row in full_df.itertuples(index=False):
                u_id = "p_" + str(int(row.PROJECTNO))
                if not u_id in node_ids:
                    nodes.append(self._build_network_node_product(row=row, id=u_id))
                    node_ids.append(u_id)

            edges = self._get_similar_edges(similar_path, user_no_list, 'p_')
        return BaseNetwork(nodes=nodes, edges=edges).__dict__, None

    @classmethod
    def _build_network_node_investor(cls, row, id):
        """
        根据csv行数据构建 投资者 网络节点
        :param row:
        :param id:
        :return: 返回网络节点数组
        """
        properties = {
            "id": int(row.userno),
            "gender": cls.query_cn_name("gender", int(row.gender)),
            "age": int(row.age),
            "edu_level": cls.query_cn_name("edu_level", int(row.edu_level)),
            "university": cls.query_cn_name("University", int(row.University)),
            "current_identity": cls.query_cn_name("CURRENTIDENTITY", int(row.CURRENTIDENTITY)),
            "borrowing_credit": int(row.borrowingcredit),
            "loan_credit": int(row.loancredit),
            "total_bid_number": int(row.totalbidnumber),
            "weighte_dbidrate_percent": row.weightedbidrate_percent,
            "baddebts_percent": row.baddebts_percent
        }
        return BaseNetworkNode.build_investor(id, properties).__dict__

    @classmethod
    def _build_network_node_product(cls, row, id):
        """
        根据csv行数据构建 产品 网络节点
        :param row:
        :param id:
        :return: 返回网络节点数组
        """
        properties = {
            "id": int(row.PROJECTNO),
            "repayment": cls.query_cn_name("REPAYMENT", int(row.REPAYMENT)),
            "total": int(row.total),
            "apr_percent": row.apr_percent,
            "bidders": int(row.bidders),
            "term": int(row.term),
            "level": int(row.level),
            "project_invest_count": int(row.project_invest_count),
        }
        return BaseNetworkNode.build_product(id, properties).__dict__

    @classmethod
    def _get_similar_edges(cls, similar_file_path, user_no_list, node_prefix):
        """
        根据上三角相似矩阵csv文件 和 筛选出来的 id 构建相似网络中的边
        :param similar_file_path: 上三角相似矩阵csv文件地址
        :param user_no_list: 筛选出来的 id
        :return: 返回网络边的数组
        """
        df = pd.read_csv(similar_file_path, index_col=0, dtype=str)
        # 将可转为数值的单元格转为 numeric，不能转的变为 NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # 统一 index 和 columns 类型为 str（和上面 dtype=str 保持一致）
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)

        # 把需要保留的 user ids 也转为 str，按 df 的 index 顺序挑选出存在的 id
        wanted_set = set(map(str, user_no_list))
        keep_ids = [idx for idx in df.index if idx in wanted_set]

        if not keep_ids:
            return []

        # 取子矩阵并转换为 numpy 矩阵
        mat = df.loc[keep_ids, keep_ids].to_numpy()

        # 获取上三角（不含对角线）索引
        tri_i, tri_j = np.triu_indices_from(mat, k=1)

        edges = []
        for i, j in zip(tri_i, tri_j):
            sim = mat[i, j]
            if pd.isna(sim):  # 跳过空值
                continue
            id1 = keep_ids[i]
            id2 = keep_ids[j]
            edges.append(
                BaseNetworkEdge(
                    id=f"{node_prefix}{id1}{node_prefix}{id2}",
                    edge_from=f"{node_prefix}{id1}",
                    edge_to=f"{node_prefix}{id2}",
                    label='相似',
                    value=float(sim)
                ).__dict__
            )
        return edges

    @classmethod
    def query_cn_name(cls, field: str, value: int):
        """
        获取枚举值
        :param field:
        :param value:
        :return:
        """
        label = '其他'
        if field == 'gender':
            return '男' if value == 0 else '女'
        elif field == 'edu_level':
            label = '初中及以下'
            if value == 1:
                label = '高中'
            elif value == 2:
                label = '本科'
            elif value == 3:
                label = '研究生及以上'
            return label
        elif field == 'University':
            if value == 1:
                label = '二本'
            elif value == 2:
                label = '一本'
            elif value == 3:
                label = '211'
            elif value == 4:
                label = '985'
            elif value == 5:
                label = 'C9'
            return label
        elif field == 'CURRENTIDENTITY':
            if value == 1:
                label = '工薪族'
            elif value == 2:
                label = '私营业主'
            elif value == 4:
                label = '网店卖家'
            elif value == 5:
                label = '学生'
            return label
        elif field == 'REPAYMENT':
            if value == 0:
                label = '等额本息'
            elif value == 1:
                label = '一次性还本付息'
            elif value == 2:
                label = '月还息，季还1/4本金'
            return label

    @classmethod
    def query_feature_cn(cls, feature):
        if feature == 'gender':
            return '性别'
        elif feature == 'age':
            return '年龄'
        elif feature == 'edu_level':
            return '教育程度'
        elif feature == 'University':
            return '毕业院校'
        elif feature == 'CURRENTIDENTITY':
            return '身份'
        elif feature == 'borrowingcredit':
            return '借入信用总分'
        elif feature == 'loancredit':
            return '借出信用总得分'
        elif feature == 'totalbidnumber':
            return '总投标次数'
        elif feature == 'weightedbidrate_percent':
            return '加权投资利率'
        elif feature == 'baddebts_percent':
            return '坏账比例'
        elif feature == 'user_invest_count':
            return '投资次数'
        elif feature == 'total':
            return '规模'
        elif feature == 'apr_percent':
            return '利率'
        elif feature == 'term':
            return '借款期限'
        elif feature == 'REPAYMENT':
            return '还款方式'
        elif feature == 'level':
            return '风险等级'
        elif feature == 'bidders':
            return '实际累计次数（累计被投资人数）'
        elif feature == 'project_invest_count':
            return '被投资次数'
        elif feature == 'credit':
            return '信用总分'
        else:
            return feature

    @classmethod
    def dataset_file_check(cls, file):
        df = pd.read_csv(file)
        columns = ['userno', 'University', 'borrowingcredit', 'loancredit', 'age', 'totalbidnumber',
                   'baddebts_percent', 'weightedbidrate_percent', 'gender', 'edu_level', 'CURRENTIDENTITY',
                   'user_invest_count', 'PROJECTNO', 'REPAYMENT', 'total', 'apr_percent', 'bidders', 'term',
                   'level', 'project_invest_count', 'reward']
        for column in columns:
            if column not in df.columns:
                return False, "缺失数据列：" + column + ",数据集需要包含" + str(columns)
        return True, None


if __name__ == '__main__':
    # filter = {"age": {"min": 31, "max": 46}, "user_invest_count": {"min": 1, "max": 30},
    #           "total": {"min": 1000, "max": 100000}, "apr_percent": {"min": 0, "max": 1}, "term": {"min": 1, "max": 24},
    #           "level": {"min": 1, "max": 8}, "project_invest_count": {"min": 0, "max": 200}}
    # tool = DataSetTool(dataset="default", filter_conditions=filter)
    # network, err_msg = tool.build_investor_investor_network()
    # if err_msg is not None:
    #     print(err_msg)
    # json_str = json.dumps(network, indent=4, ensure_ascii=False)
    # with open('C:\\code\\zntgsy\\data\\network_all_2.json', 'w', encoding='utf-8') as file:
    #     file.write(json_str)
    df = pd.read_csv('C:\\code\\zntgsy\\data\\dataset\\all\\all.csv')


    # 使用value_counts计算每个唯一值的出现次数
    # value_counts = df['PROJECTNO'].value_counts()

    print(len(df))


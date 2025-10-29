import os
from flask import Flask

app = Flask(__name__)


# 算法分类 基础智能投顾实验
ALGORITHM_TYPE_BASE = 1
# 算法分类 带约束条件的智能投顾实验
ALGORITHM_TYPE_CONSTRAINT = 2
# 算法分类 对新投资者的智能投顾实验
ALGORITHM_TYPE_NEW_INVESTOR = 3
# 算法分类 可解释增强的智能投顾实验
ALGORITHM_TYPE_EXPLAIN = 4
# 基础智能投顾实验支持的算法
ALGORITHM_BASE_NAME_EHGNN = 'ehgnn'
ALGORITHM_BASE_NAME_COLD = 'cold'
ALGORITHM_BASE_NAME_MISS = 'miss'
ALGORITHM_BASE_NAME_MABTSLR = 'MAB-TS-LR'
ALGORITHM_BASE_NAMES = [ALGORITHM_BASE_NAME_EHGNN,
                        ALGORITHM_BASE_NAME_COLD,
                        ALGORITHM_BASE_NAME_MISS,
                        ALGORITHM_BASE_NAME_MABTSLR]
ALGORITHM_BASE_PARAMS = [
    {
        "name": "异构图神经网络",
        "value": "ehgnn",
        "property": [
            {
                "label": "隐藏层维度",
                "value": "hidden_dim",
                "commit": "取值范围:正整数; 常见取值:16、32、64、128、256等",
                "default_value": 64
            },
            {
                "label": "积分梯度计算步数",
                "value": "steps",
                "commit": "取值范围:正整数; 常见取值:20到100之间的整数",
                "default_value": 50
            },
            {
                "label": "学习率",
                "value": "lr",
                "commit": "取值范围:0到1之间的浮点数; 常见取值:0.001到0.1之间",
                "default_value": 0.001
            },
            {
                "label": "训练轮数",
                "value": "epochs",
                "commit": "取值范围:1到正无穷的整数; 常见取值:10到1000之间的整数",
                "default_value": 200
            }
        ]
    },
    {
        "name": "基于Logit的在线学习",
        "value": "miss",
        "property": [

            {
                "label": "学习率",
                "value": "lr",
                "commit": "取值范围:0到1之间的浮点数; 常见取值:0.001到0.1之间",
                "default_value": 0.01
            },
            {
                "label": "训练轮数",
                "value": "epochs",
                "commit": "取值范围:1到正无穷的整数; 常见取值:10到1000之间的整数",
                "default_value": 10

            }
        ]
    },
    {
        "name": "CatboostNN",
        "value": "cold",
        "property": [

            {
                "label": "学习率",
                "value": "lr",
                "commit": "取值范围:0到1之间的浮点数; 常见取值:0.001到0.1之间",
                "default_value": 0.1
            },
            {
                "label": "迭代次数",
                "value": "iterations",
                "commit": "取值范围:1到正无穷的整数; 常见取值:100到1000之间的整数.",
                "default_value": 100
            },
            {
                "label": "树的深度",
                "value": "depth",
                "commit": "取值范围:1到16的整数; 常见取值:4到10之间的整数",
                "default_value": 6
            }
        ]
    },
    {
        "name": "强化学习",
        "value": "MAB-TS-LR",
        "property": [

            {
                "label": "迭代次数",
                "value": "iterations",
                "commit": "取值范围:1到正无穷的整数; 常见取值:50到1000之间的整数.",
                "default_value": 100
            },
            {
                "label": "优化方法",
                "value": "opt_method",
                "commit": "常见取值选项:'L-BFGS-B'、'BFGS'、'Newton-CG'、'SLSQP'.",
                "default_value": 'L-BFGS-B'
            },
            {
                "label": "优化选项",
                "value": "opt_options",
                "commit": "常见取值选项:{'maxiter': 1000}、{'ftol': 1000} ",
                "default_value": {'maxiter': 1000}
            }
        ]
    }
]

ALGORITHM_CONSTRAINT_NAME_CMAB = 'cmab'
ALGORITHM_CONSTRAINT_NAMES = [ALGORITHM_CONSTRAINT_NAME_CMAB]
ALGORITHM_CONSTRAINT_PARAMS = [
    {
        "name": "全局约束强化学习框架",
        "value": "cmab",
        "property": [
            {
                "label": "探索系数",
                "value": "alpha",
                "commit": "取值范围(0,1]",
                "default_value": 0.7
            },
            {
                "label": "q的步长",
                "value": "gamma",
                "commit": "取值范围(0,0.01]",
                "default_value": 0.0009
            },
            {
                "label": "Q的步长",
                "value": "lambd",
                "commit": "取值范围(0,0.01]",
                "default_value": 0.0009
            },
            {
                "label": "theta的步长",
                "value": "delta",
                "commit": "取值范围(0,0.01]",
                "default_value": 0.0009
            },
            {
                "label": "每轮推荐数",
                "value": "m",
                "commit": "取值范围[1,50]",
                "default_value": 5
            },
            {
                "label": "总训练轮数",
                "value": "T",
                "commit": "取值范围[100,10000]",
                "default_value": 1000
            },
            {
                "label": "theta初始值",
                "value": "init_theta",
                "commit": "取值范围[1,100]",
                "default_value": 20.0
            }
        ],
        "constraint": [
            {
                "label": "利率",
                "value": "apr_percent",
                "commit": "连续特征（利率）用分位数约束，四等分，可选填1、2、3、4",
                "default_value": [{"label": '前25%', 'value': 1},
                                  {"label": '25%-50%', 'value': 2},
                                  {"label": '50%-75%', 'value': 3},
                                  {"label": '75%-100%', 'value': 4}]
            },
            {
                "label": "借款期限",
                "value": "term",
                "commit": "连续特征（借款期限）用分位数约束，四等分，可选填1、2、3、4",
                "default_value": [{"label": '前25%', 'value': 1},
                                  {"label": '25%-50%', 'value': 2},
                                  {"label": '50%-75%', 'value': 3},
                                  {"label": '75%-100%', 'value': 4}]
            },
            {
                "label": "还款方式",
                "value": "REPAYMENT",
                "commit": "分类特征（还款方式）直接约束，可选填0、1、2",
                "default_value": [{"label": '等额本息', 'value': 0},
                                  {"label": '一次性还本付息', 'value': 1},
                                  {"label": '月还息，季还1/4本金', 'value': 2}]
            },
            {
                "label": "风险等级",
                "value": "level",
                "commit": "分类特征（风险等级）直接约束，可选填1、2、3、4、5、6、7、8",
                "default_value": [{"label": '1', 'value': 1},
                                  {"label": '2', 'value': 2},
                                  {"label": '3', 'value': 3},
                                  {"label": '4', 'value': 4},
                                  {"label": '5', 'value': 5},
                                  {"label": '6', 'value': 6},
                                  {"label": '7', 'value': 7},
                                  {"label": '8', 'value': 8}
                                  ]
            }

        ]

    },
]

ALGORITHM_NEW_INVESTOR_COLD_RUN = 'cold-run'
ALGORITHM_NEW_INVESTOR_NAMES = [ALGORITHM_NEW_INVESTOR_COLD_RUN]
ALGORITHM_NEW_INVESTOR_PARAMS = [
    {
        "name": "异构神经图网络",
        "value": "cold-run",
        "property": [

        ]

    },
    {
        "name": "相似矩阵填充",
        "value": "cold-run1",
        "property": [

        ]

    },
    {
        "name": "社会网络",
        "value": "cold-run2",
        "property": [

        ]

    },
]

DATASET_FEATURES = {
    "investor_features": {
        'gender': '性别',
        'age': '年龄',
        'edu_level': '教育程度',
        'University': '毕业院校',
        'CURRENTIDENTITY': '身份',
        'weightedbidrate_percent': '加权投资利率',
        'baddebts_percent': '坏账比例',
        'user_invest_count': '投资次数',
        'credit': '投资者信用'
    },
    "product_features": {
        'total': '规模',
        'apr_percent': '利率',
        'term': '借款期限',
        'REPAYMENT': '还款方式',
        'level': '风险等级',
        'project_invest_count': '被投资次数'
    }
}
# 数据存放位置基本目录
BASE_DATA_PATH = 'data/'
# 算法数据集文件夹
BASE_DATA_SET_PATH = BASE_DATA_PATH + 'dataset/'
# 系统文件存放位文件夹
BASE_SYS_FILE_PATH = BASE_DATA_PATH + 'sys_file/'
# 算法结果文件夹
BASE_TASK_PATH = BASE_DATA_PATH + "task_result/"


def get_file_path(*args):
    '''
    获取文件绝对路径
    :param args:
    :return:
    '''
    rootPath = os.path.dirname(__file__)
    filePath = os.path.join(rootPath, '..', *args)
    return filePath


def delete_dir(path):
    folder_path = get_file_path(path)
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
    os.removedirs(folder_path)

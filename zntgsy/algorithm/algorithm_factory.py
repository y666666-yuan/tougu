from . import ehgnn
from . import cold
from . import miss
from . import mabtslr
from . import cmab
from . import ehgnn_lqd
from core import sys


def algorithm_run(dataset_path: str,
                  user_id: str,
                  algorithm_name: str,
                  algorithm_type: int = 1,
                  filter_conditions=None,
                  custom_investor_features: list = None,
                  custom_product_features: list = None,
                  algorithm_params: dict = None,
                  constraint_params: dict = None
                  ):
    '''
    算法跑任务
    :param constraint_params: 约束条件 当algorithm_type=2，此参数生效
    :param dataset_path: 数据集路径 项目根路径  data/dataset/default
    :param user_id: 系统用户id
    :param algorithm_type: 算法类型
    :param algorithm_name: 算法名称
    :param filter_conditions: 过滤条件
    :param custom_investor_features: 投资者属性
    :param custom_product_features: 产品属性
    :param algorithm_params: 算法参数
    :return:
    '''
    if algorithm_type == sys.ALGORITHM_TYPE_BASE:
        return _algorithm_run_base(dataset_path=dataset_path,
                                   user_id=user_id,
                                   algorithm_name=algorithm_name,
                                   filter_conditions=filter_conditions,
                                   custom_investor_features=custom_investor_features,
                                   custom_product_features=custom_product_features,
                                   algorithm_params=algorithm_params)
    elif algorithm_type == sys.ALGORITHM_TYPE_CONSTRAINT:
        return _algorithm_run_constraint(dataset_path=dataset_path,
                                         user_id=user_id,
                                         algorithm_name=algorithm_name,
                                         filter_conditions=filter_conditions,
                                         custom_investor_features=custom_investor_features,
                                         custom_product_features=custom_product_features,
                                         algorithm_params=algorithm_params,
                                         constraint_params=constraint_params)
    elif algorithm_type == sys.ALGORITHM_TYPE_NEW_INVESTOR:
        return _algorithm_run_new_investor(dataset_path=dataset_path,
                                           user_id=user_id,
                                           algorithm_name=algorithm_name,
                                           filter_conditions=filter_conditions,
                                           custom_investor_features=custom_investor_features,
                                           custom_product_features=custom_product_features)
    else:
        print(f'算法类型:{algorithm_type} 暂不支持')
        return None, f'算法类型:{algorithm_type} 暂不支持'


def _algorithm_run_base(dataset_path: str,
                        user_id: str,
                        algorithm_name: str,
                        filter_conditions=None,
                        custom_investor_features: list = None,
                        custom_product_features: list = None,
                        algorithm_params: dict = None):
    if algorithm_name == sys.ALGORITHM_BASE_NAME_EHGNN:
        ehgnn_runner = ehgnn.EHGNNRunner(
            investor_features=custom_investor_features,
            product_features=custom_product_features,
            hidden_dim=int(algorithm_params.get('hidden_dim')),  # 超参数名称:隐藏层维度; 取值范围:正整数; 常见取值:16、32、64、128、256等.
            steps=int(algorithm_params.get('steps')),  # 超参数名称:积分梯度计算步数; 取值范围:正整数; 常见取值:20到100之间的整数.
            lr=float(algorithm_params.get('lr')),  # 超参数名称:学习率; 取值范围:0到1之间的浮点数; 常见取值:0.001到0.1之间.
            epochs=int(algorithm_params.get('epochs')),  # 超参数名称:训练轮数; 取值范围:1到正无穷的整数; 常见取值:10到1000之间的整数.
            user_id=user_id
        )
        data_path = sys.get_file_path(dataset_path)
        return ehgnn_runner.run_ehgnn(filter_conditions=filter_conditions,
                                      data_path=data_path)

    elif algorithm_name == sys.ALGORITHM_BASE_NAME_COLD:
        cold_runner = cold.ColdRecommender(
            investor_features=custom_investor_features,
            product_features=custom_product_features,
            iterations=int(algorithm_params.get('iterations')),  # 超参数名称:迭代次数; 取值范围:1到正无穷的整数; 常见取值:100到1000之间的整数.
            depth=int(algorithm_params.get('depth')),  # 树的深度:; 取值范围:1到16的整数; 常见取值:4到10之间的整数.
            learning_rate=float(algorithm_params.get('lr')),  # 超参数名称:学习率; 取值范围:0到1之间的浮点数; 常见取值:0.01到0.3之间.
            user_id=user_id
        )
        data_path = sys.get_file_path(dataset_path)
        return cold_runner.run_cold(filter_conditions, data_path, None)
    elif algorithm_name == sys.ALGORITHM_BASE_NAME_MISS:
        miss_runner = miss.MissRecommender(
            user_id=user_id,
            investor_features=custom_investor_features,
            product_features=custom_product_features,
            learning_rate=float(algorithm_params.get('lr')),  # 超参数名称:学习率; 取值范围:0到1之间的浮点数; 常见取值:0.001到0.1之间.
            epochs=int(algorithm_params.get('epochs'))  # 超参数名称:训练轮数; 取值范围:1到正无穷的整数; 常见取值:10到1000之间的整数.
        )
        data_path = sys.get_file_path(dataset_path)
        return miss_runner.run_miss(filter_conditions, data_path, None)
    elif algorithm_name == sys.ALGORITHM_BASE_NAME_MABTSLR:
        mabtslr_runner = mabtslr.MABTSLRRecommender(
            user_id=user_id,
            investor_features=custom_investor_features,
            product_features=custom_product_features,
            iterations=int(algorithm_params.get('iterations')),  # 超参数名称:迭代次数; 取值范围:1到正无穷的整数; 常见取值:50到1000之间的整数.
            opt_method=str(algorithm_params.get('opt_method')),  # 超参数名称:训练轮数; 取值范围:1到正无穷的整数; 常见取值:10到1000之间的整数.
            opt_options=algorithm_params.get("opt_options")  # 超参数名称:优化选项; 常见取值选项:{'maxiter': 1000}、{'ftol': 1000} .
        )
        data_path = sys.get_file_path(dataset_path)
        return mabtslr_runner.run_algorithm(filter_conditions, data_path, None)
    else:
        print(f'算法:{algorithm_name} 暂不支持')
        return None, f'算法:{algorithm_name} 暂不支持'


def _algorithm_run_constraint(dataset_path: str,
                              user_id: str,
                              algorithm_name: str,
                              filter_conditions=None,
                              custom_investor_features: list = None,
                              custom_product_features: list = None,
                              algorithm_params: dict = None,
                              constraint_params: dict = None
                              ):
    if algorithm_name == sys.ALGORITHM_CONSTRAINT_NAME_CMAB:
        cmab_runner = cmab.CMABRunner(
            investor_features=custom_investor_features,
            product_features=custom_product_features,
            user_id=user_id,
            alpha=float(algorithm_params.get('alpha')),  # 探索系数，取值范围(0,1]
            gamma=float(algorithm_params.get('gamma')),  # q的步长，取值范围(0,0.01]
            lambd=float(algorithm_params.get('lambd')),  # Q的步长，取值范围(0,0.01]
            delta=float(algorithm_params.get('delta')),  # theta的步长，取值范围(0,0.01]
            m=int(algorithm_params.get('m')),  # 每轮推荐数，取值范围[1,50]
            T=int(algorithm_params.get('T')),  # 总训练轮数，取值范围[100,10000]
            init_theta=int(algorithm_params.get('init_theta'))  # theta初始值，取值范围[1,100]
        )
        data_path = sys.get_file_path(dataset_path)
        return cmab_runner.run_cmab(filter_conditions=filter_conditions,
                                    data_path=data_path,
                                    constraint_config=constraint_params,
                                    metrics_to_calculate=None)
    else:
        print(f'算法:{algorithm_name} 暂不支持')
        return None, f'算法:{algorithm_name} 暂不支持'


def _algorithm_run_new_investor(dataset_path: str,
                                user_id: str,
                                algorithm_name: str,
                                filter_conditions=None,
                                custom_investor_features: list = None,
                                custom_product_features: list = None
                                ):
    if algorithm_name == sys.ALGORITHM_NEW_INVESTOR_COLD_RUN:
        runner = ehgnn_lqd.EHGNNRunner(
            investor_features=custom_investor_features,
            product_features=custom_product_features,
            user_id=user_id,
            hidden_dim=64,
            lr=0.001,
            epochs=200
        )
        data_path = sys.get_file_path(dataset_path)
        return runner.run_comparison(filter_conditions=filter_conditions,
                                     data_path=data_path)
    else:
        print(f'算法:{algorithm_name} 暂不支持')
        return None, f'算法:{algorithm_name} 暂不支持'

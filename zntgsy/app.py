import io
import json
import os
import sys

from flask import Flask, request, jsonify

from algorithm import algorithm_factory
from core import sys as csys
from core.sys import app
from core.result import Result
from utils.ex_data_utils import DataSetTool
from utils.ex_data_utils import TaskTool

@app.route('/run-task', methods=['POST'])
def run_task():
    '''
    跑实验
    body数据：
    {
        "user_id": 1,
        "dataset_name":"default",
        "filter":{
            "gender":0,
            "age":1
        }
    }

    :return:
    '''
    body = request.get_json()
    if body is None:
        return jsonify(Result.fail(None, '请求参数错误，请检查！')), 200
    filter = body.get('filter')
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = body.get('user_id')
    dataset_name = body.get('dataset_name')
    algorithm_name = body.get('algorithm_name')
    algorithm_type = body.get('algorithm_type')
    if algorithm_type is None or int(algorithm_type) < 1 or int(algorithm_type) > 4:
        return jsonify(Result.fail(None, '请求参数错误，请检查！').__dict__), 200
    if user_id is None or int(user_id) <= 0 or dataset_name is None or algorithm_name is None:
        return jsonify(Result.fail(None, '请求参数错误，请检查！').__dict__), 200
    if not os.path.exists("data/dataset/" + dataset_name + "/"):
        return jsonify(Result.fail(None, '数据集不存在！').__dict__), 200
    constraint_params = body.get("constraint_params")
    if int(algorithm_type) == csys.ALGORITHM_TYPE_CONSTRAINT and (
            constraint_params is None or len(constraint_params) != 1):
        return jsonify(Result.fail(None, '请求参数(约束条件)错误，请检查！').__dict__), 200
    user_id = int(user_id)
    filter_conditions = {}
    filter_conditions['gender'] = filter.get('gender')
    filter_conditions['age'] = filter.get('age')
    filter_conditions['edu_level'] = filter.get('edu_level')
    filter_conditions['CURRENTIDENTITY'] = filter.get('current_identity')
    filter_conditions['user_invest_count'] = filter.get('user_invest_count')
    filter_conditions['total'] = filter.get('total')
    filter_conditions['apr_percent'] = filter.get('apr_percent')
    filter_conditions['term'] = filter.get('term')
    filter_conditions['REPAYMENT'] = filter.get('repayment')
    filter_conditions['level'] = filter.get('level')
    filter_conditions['project_invest_count'] = filter.get('project_invest_count')

    investor_features = body.get("investor_features")
    product_features = body.get("product_features")
    algorithm_params = body.get("algorithm_params")

    if (not isinstance(investor_features, list)) or (
            not isinstance(product_features, list) or (not isinstance(algorithm_params, dict))):
        return jsonify(Result.fail(None, '请求参数错误，请检查！').__dict__), 200
    if 'credit' in investor_features:
        investor_features.append("loancredit")
        investor_features.append("borrowingcredit")
    data_path = csys.get_file_path("data/dataset/" + dataset_name)
    result_path, err_msg = algorithm_factory.algorithm_run(dataset_path=data_path,
                                                           user_id=str(user_id),
                                                           algorithm_type=int(algorithm_type),
                                                           algorithm_name=algorithm_name,
                                                           filter_conditions=filter_conditions,
                                                           custom_investor_features=investor_features,
                                                           custom_product_features=product_features,
                                                           algorithm_params=algorithm_params,
                                                           constraint_params=constraint_params)
    print(f"实验结果跑完，结果路径：{result_path}")
    if err_msg is None:
        task_id = os.path.basename(result_path)
        try:
            with open(result_path + '/request_params.json', 'w', encoding='utf-8') as f:
                json.dump(body, f, ensure_ascii=False)
        except Exception as e:
            print(e)
            print("生成请求参数文件失败")
        return jsonify(Result.ok(task_id).__dict__), 200
    elif err_msg == '产品个数不足5个无法推荐,请重新选择数据。':
        return jsonify(Result(code=201, msg=err_msg, data=None).__dict__), 200
    else:
        return jsonify(Result.fail(None, err_msg).__dict__), 200


@app.route('/query-network', methods=['POST'])
def query_network():
    '''
    根据筛选条件获取数据网状结构

    :return:
    '''
    body = request.get_json()
    if body is None:
        return jsonify(Result.fail(None, '请求参数错误，请检查！')), 200
    filter = body.get('filter')
    dataset_name = body.get('dataset_name')
    network_type = body.get('network_type')
    if network_type is None:
        network_type = 0
    if dataset_name is None or (not isinstance(network_type, int)):
        return jsonify(Result.fail(None, '请求参数错误，请检查！').__dict__), 200

    filter_conditions = {}
    filter_conditions['gender'] = filter.get('gender')
    filter_conditions['age'] = filter.get('age')
    filter_conditions['edu_level'] = filter.get('edu_level')
    filter_conditions['CURRENTIDENTITY'] = filter.get('current_identity')
    filter_conditions['user_invest_count'] = filter.get('user_invest_count')
    filter_conditions['total'] = filter.get('total')
    filter_conditions['apr_percent'] = filter.get('apr_percent')
    filter_conditions['term'] = filter.get('term')
    filter_conditions['REPAYMENT'] = filter.get('repayment')
    filter_conditions['level'] = filter.get('level')
    filter_conditions['project_invest_count'] = filter.get('project_invest_count')

    network = None
    err_msg = None
    tool = DataSetTool(dataset=dataset_name, filter_conditions=filter_conditions)
    # network_type: 0-投资者-产品网络  1- 投资者网络  2-产品网络
    if int(network_type) == 0:
        network, err_msg = tool.build_investor_product_network_fast()
    elif int(network_type) == 1:
        network, err_msg = tool.build_investor_investor_network()
    elif int(network_type) == 2:
        network, err_msg = tool.build_product_product_network()

    if err_msg is not None:
        return jsonify(Result.fail(None, err_msg).__dict__), 200

    return jsonify(Result.ok(network).__dict__), 200


@app.route('/algorithm-list', methods=['GET'])
def query_algorithm_list():
    '''
    获取所有的算法类型

    :return:
    '''
    algorithm_type = request.args.get("algorithm_type")
    if algorithm_type is None or int(algorithm_type) < 1 or int(algorithm_type) > 4:
        return jsonify(Result.fail(None, '请求参数错误，请检查！').__dict__), 200
    return jsonify(Result.ok(TaskTool.get_algorithm_list(int(algorithm_type))).__dict__), 200


@app.route('/ex-data/list', methods=['GET'])
def query_ex_data_path():
    '''
    获取实验数据集
    :return:
    '''
    file_list = os.listdir(csys.get_file_path("data/dataset"))
    return jsonify(Result.ok(file_list).__dict__), 200


@app.route('/ex-data/investors', methods=['GET'])
def query_ex_data_investors():
    '''
    获取实验使用的投资者编号
    :return:
    '''
    user_id = request.args.get("user_id")
    task_id = request.args.get("task_id")
    page_num = request.args.get("page_num")
    page_size = request.args.get("page_size")
    if page_num is None or page_size is None:
        return jsonify(Result.fail(None, "参数缺失，请检查").__dict__), 200

    page_num = int(page_num)
    page_size = int(page_size)

    algorithm_type = request.args.get("algorithm_type")
    if algorithm_type is None or int(algorithm_type) < 1 or int(algorithm_type) > 4:
        return jsonify(Result.fail(None, '请求参数错误，请检查！').__dict__), 200
    algorithm_name = request.args.get("algorithm_name")
    if user_id is None or task_id is None or algorithm_name is None:
        return jsonify(Result.fail(None, "参数缺失，请检查").__dict__), 200
    result_data = {}
    err = None
    try:
        tool = TaskTool(task_id=task_id, user_id=int(user_id), algorithm_name=algorithm_name,
                        algorithm_type=int(algorithm_type))
        result_data, err = tool.get_processed_investors(page_size=page_size, page_num=page_num)
        if err is not None:
            return jsonify(Result.fail(None, err).__dict__), 200
        return jsonify(Result.ok(result_data).__dict__), 200
    except Exception as e:
        return jsonify(Result.fail(None, str(e)).__dict__), 200


@app.route('/ex-data/investor-detail', methods=['GET'])
def query_ex_data_investor_detail():
    '''
    获取用户某一次实验其中一个投资者详情
    :return:
    '''
    user_id = request.args.get("user_id")
    task_id = request.args.get("task_id")
    investor_no = request.args.get("investor_no")
    algorithm_type = request.args.get("algorithm_type")
    algorithm_name = request.args.get("algorithm_name")
    if algorithm_type is None or int(algorithm_type) < 1 or int(algorithm_type) > 4:
        return jsonify(Result.fail(None, '请求参数错误，请检查！').__dict__), 200
    if user_id is None or task_id is None or investor_no is None or algorithm_name is None:
        return jsonify(Result.fail(None, "参数缺失，请检查").__dict__), 200
    try:
        tool = TaskTool(task_id=task_id, user_id=int(user_id), algorithm_name=algorithm_name,
                        algorithm_type=int(algorithm_type))
        detail = tool.get_processed_investor_detail(int(investor_no))
        return jsonify(Result.ok(detail).__dict__), 200
    except Exception as e:
        return jsonify(Result.fail(None, str(e)).__dict__), 200


@app.route('/ex-data/query-task-metrics', methods=['GET'])
def query_task_metrics():
    '''
    获取某一次任务的算法评价结果
    :return:
    '''
    user_id = request.args.get("user_id")
    task_id = request.args.get("task_id")
    algorithm_type = request.args.get("algorithm_type")
    algorithm_name = request.args.get("algorithm_name")
    if algorithm_type is None or int(algorithm_type) < 1 or int(algorithm_type) > 4:
        return jsonify(Result.fail(None, '请求参数错误，请检查！').__dict__), 200
    if user_id is None or task_id is None or algorithm_name is None:
        return jsonify(Result.fail(None, "参数缺失，请检查").__dict__), 200
    try:
        tool = TaskTool(task_id=task_id, user_id=int(user_id), algorithm_name=algorithm_name,
                        algorithm_type=int(algorithm_type))
        result_data, err = tool.get_model_metrics()
        if err is None:
            return jsonify(Result.ok(result_data, err).__dict__), 200
        else:
            return jsonify(Result.fail(None, err).__dict__), 200
    except Exception as e:
        return jsonify(Result.fail(None, str(e)).__dict__), 200


@app.route('/ex-data/query-features', methods=['GET'])
def query_features():
    '''
    获取数据集 属性介绍
    :return:
    '''
    dataset_name = request.args.get("dataset_name")
    if not os.path.exists(csys.get_file_path("data/dataset", dataset_name)):
        return jsonify(Result.fail(None, '数据集不存在！').__dict__), 200

    return jsonify(Result.ok(csys.DATASET_FEATURES).__dict__), 200


@app.route('/ex-data/upload/dataset', methods=['POST'])
def upload_dataset_file():
    '''
    上传数据集
    :return:
    '''
    if 'file' not in request.files:
        return jsonify(Result.fail('没有文件上传').__dict__), 200
    file = request.files['file']
    # 如果用户没有选择文件，则file将为None
    if file.filename == '':
        return jsonify(Result.fail('没有文件上传').__dict__), 200
    if not file.filename.endswith(".csv"):
        return jsonify(Result.fail('文件格式不正确，只支持csv文件上传').__dict__), 200

    save_path = csys.get_file_path("data/dataset/" + file.filename.replace('.csv', ''))
    if os.path.exists(save_path):
        return jsonify(Result.fail('文件名称已经存在，请重命名').__dict__), 200
    os.makedirs(save_path)
    file.save(save_path + '/' + file.filename)

    flag, error_msg = DataSetTool.dataset_file_check(
        csys.get_file_path("data/dataset/" + file.filename.replace('.csv', '')) + "/" + file.filename)
    if not flag:
        csys.delete_dir("data/dataset/" + file.filename.replace('.csv', ''))
        return jsonify(Result.fail(error_msg).__dict__), 200
    return jsonify(Result.ok().__dict__), 200


@app.route('/ex-data/query-task-params', methods=['GET'])
def query_task_params():
    '''
    获取某一次任务的入参
    :return:
    '''
    user_id = request.args.get("user_id")
    task_id = request.args.get("task_id")
    algorithm_type = request.args.get("algorithm_type")
    algorithm_name = request.args.get("algorithm_name")
    if algorithm_type is None or int(algorithm_type) < 1 or int(algorithm_type) > 4:
        return jsonify(Result.fail(None, '请求参数错误，请检查！').__dict__), 200
    if user_id is None or task_id is None or algorithm_name is None:
        return jsonify(Result.fail(None, "参数缺失，请检查").__dict__), 200
    try:
        tool = TaskTool(task_id=task_id, user_id=int(user_id), algorithm_name=algorithm_name,
                        algorithm_type=int(algorithm_type))
        result_data, err = tool.get_task_params()
        if err is None:
            return jsonify(Result.ok(result_data, err).__dict__), 200
        else:
            return jsonify(Result.fail(None, err).__dict__), 200
    except Exception as e:
        return jsonify(Result.fail(None, str(e)).__dict__), 200


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
    app.run()

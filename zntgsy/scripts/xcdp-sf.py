import requests
import json
import time
from datetime import datetime, timedelta

result_data = {
    "hysffxgl": '行业研究以涉诉企业公开案件作为研究基础，将企业分为至32大行业，经行业司法风险评价体系评估，其中房地产行业高司法风险的企业占比最高，达到14.33%，行业或面临较高的司法风险；银行行业公司经营侧案件占比最高，需重点关注行业经营风险；非银金融行业公司财务类案件占比最高，行业或面临较大的违约风险；综合行业公司治理类案件占比最高，行业公司治理能力或有待进一步提升。\n\n 近一年案件量增速最快，案件均值最高的行业分别是综合和美容护理，需多关注相关行业风险的变化。',
    "industry": []
}

API_TOKEN_URL = 'http://bdx.cjbdi.com/jrkjpt/bdx-api/token/dynamic'
API_GROUP_BY_JAAY_URL = 'http://bdx.cjbdi.com/jrkjpt/leto/caserating/getCaseRatingGroupJaayPercent'
API_GROUP_BY_INDUSTRY_URL = 'http://bdx.cjbdi.com/jrkjpt/leto/caserating/getCaseRatingTypeByIndustry'
API_CASE_RATING_ALL_URL = 'http://bdx.cjbdi.com/jrkjpt/leto/datastatistics/getCaseRatingTypeByAllIndustry'


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data


def get_dynamic_token(token: str):
    headers = {'Authorization': token}
    response = requests.get(API_TOKEN_URL, headers=headers)
    if response.status_code == 200:
        # 成功获取数据，解析JSON响应
        data = response.json()
        if data.get('code') == 200:
            return data.get('data')
        else:
            return None
    else:
        return None


def get_group_by_industry_data(token: str, inustry: str):
    headers = {'Authorization': token}
    url = API_GROUP_BY_INDUSTRY_URL + "?industry=" + inustry
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # 成功获取数据，解析JSON响应
        data = response.json()
        if data.get('code') == '00000':
            return data.get('data')
        else:
            return None
    else:
        return None


def get_group_by_jaay_data(token: str, inustry: str):
    headers = {'Authorization': token}
    endTime = datetime.now()
    startTime = endTime.replace(year=endTime.year - 1)
    url = API_GROUP_BY_JAAY_URL + "?industry=" + inustry + "&startTime=" + startTime.strftime(
        '%Y-%m-%d %H:%M:%S') + "&endTime=" + endTime.strftime('%Y-%m-%d %H:%M:%S')
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # 成功获取数据，解析JSON响应
        data = response.json()
        if data.get('code') == '00000':
            return data.get('data')
        else:
            return None
    else:
        return None


def get_case_rating_by_all_industry(token: str):
    headers = {'Authorization': token}

    response = requests.get(API_CASE_RATING_ALL_URL, headers=headers)
    if response.status_code == 200:
        # 成功获取数据，解析JSON响应
        data = response.json()
        if data.get('code') == '00000':
            return data.get('data')
        else:
            return None
    else:
        return None


if __name__ == '__main__':
    token = 'eyJhbGciOiJIUzUxMiJ9.eyJhcGlfdXNlcl90b2tlbl9rZXkiOjIzfQ.Li_G15EHbimVqyf_qqWKwGoOWjnI37kplR-1jjjM9viNMMyG2BZALU5IIcd_vaKL3fqxd5DeW-COEzv-BjDRCg'
    dynamic_token = get_dynamic_token(token)
    if dynamic_token is None:
        print("获取动态token出错")
    print("获取动态token成功：" + dynamic_token)
    all_industry_case_rating = get_case_rating_by_all_industry(dynamic_token)
    industry_map = {}
    if all_industry_case_rating is not None:
        for item in all_industry_case_rating:
            industry_map[item.get("industrie")] = item
    print(all_industry_case_rating)
    json_path = "sf.json"
    sf_json_ui_data = list(read_json_file(json_path))
    industrys = []
    for item in sf_json_ui_data:
        industry_item = {
            "name": item.get("name"),
            "overall_risk": item.get("overall_risk")
        }
        industry_item['case_rating'] = industry_map.get(item.get("name"))
        print("开始获取行业数据：" + item.get("name"))
        industry_data = get_group_by_industry_data(dynamic_token, item.get("name"))
        if industry_data is not None:
            industry_item['group_by_industry'] = industry_data
        else:
            print(f"***{item.get('name')}:获取行业风险分布数据失败")
        jaay_data = get_group_by_jaay_data(dynamic_token, item.get("name"))
        if jaay_data is not None:
            industry_item['group_by_jaay'] = jaay_data
        else:
            print(f"***{item.get('name')}:获取结案案由分布数据失败")
        industrys.append(industry_item)
        time.sleep(1)
    result_data["industry"] = industrys
    with open('sf_data.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False)

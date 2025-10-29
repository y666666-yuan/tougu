CODE_SUCCESS = 200
CODE_FAIL = 500
MSG_SUCCESS = "success"
MSG_FAIL = "fail"

'''接口返回对象
对象包含 code ，msg ，data 三个属性
'''


class Result:

    def __init__(self, code: int, msg: str, data):
        self.code = code
        self.msg = msg
        self.data = data

    @staticmethod
    def ok(data=None, msg: str = MSG_SUCCESS):
        return Result(CODE_SUCCESS, msg, data)

    @staticmethod
    def fail(data=None, msg: str = MSG_FAIL):
        return Result(CODE_FAIL, msg, data)

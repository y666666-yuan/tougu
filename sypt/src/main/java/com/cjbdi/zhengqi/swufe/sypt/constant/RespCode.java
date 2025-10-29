package com.cjbdi.zhengqi.swufe.sypt.constant;

public enum RespCode {

//    public static final String SUCCESS_CODE = "200";

    SUCCESS("200", "成功"),
    FAIL("500", "失败"),
    UNAUTHORIZED("401", "未授权"),
    FORBIDDEN("403", "禁止访问"),
    NOT_FOUND("404", "未找到"),
    BAD_REQUEST("400", "请求错误"),
    UNPROCESSABLE_ENTITY("422", "请求参数错误"),
    INTERNAL_SERVER_ERROR("500", "服务器错误"),
    SERVICE_UNAVAILABLE("503", "服务不可用"),
    GATEWAY_TIMEOUT("504", "网关超时"),
    GATEWAY_TIMEOUT_DESC("504", "网关超时"),
    TOO_MANY_REQUESTS("429", "请求过于频繁"),
    REQUEST_TIMEOUT("408", "请求超时"),
    USER_NOT_FOUND("404", "用户不存在"),
    USER_ALREADY_EXISTS("409", "用户已存在"),
    USER_PASSWORD_ERROR("401", "用户名或密码错误"),
    USER_NOT_LOGIN("401", "用户未登录"),
    USER_NOT_ACTIVATED("401", "用户未激活")
    ;

    private final String code;
    private final String msg;

    RespCode(String code, String msg) {
        this.code = code;
        this.msg = msg;
    }

    public String getCode() {
        return code;
    }

    public static RespCode getByCode(String code) {
        for (RespCode respCode : values()) {
            if (respCode.getCode().equals(code)) {
                return respCode;
            }
        }
        return null;
    }

}

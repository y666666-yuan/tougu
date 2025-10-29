package com.cjbdi.zhengqi.swufe.sypt.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

import java.io.Serializable;

@JsonIgnoreProperties(ignoreUnknown = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
@Builder
public class Resp<T> implements Serializable {

    @Schema(description = "状态码", examples = {"200"})
    private String code;

    @Schema(description = "状态信息",  examples = {"success"})
    private String msg;

    @Schema(description = "数据")
    private T data;

    public static <T> Resp<T> success(T data) {
        return Resp.<T>builder()
                .code("200")
                .msg("success")
                .data(data)
                .build();
    }

    public static <T> Resp<T> success(T data, String msg) {
        return Resp.<T>builder()
                .code("200")
                .msg(msg)
                .data(data)
                .build();
    }

    public static <T> Resp<T> success(T data, Throwable throwable) {
        return Resp.<T>builder()
                .code("200")
                .msg(throwable.getLocalizedMessage())
                .data(data)
                .build();
    }

    public static <T> Resp<T> success() {
        return Resp.<T>builder()
                .code("200")
                .msg("success")
                .build();
    }

    public static <T> Resp<T> successMsg(String msg) {
        return Resp.<T>builder()
                .code("200")
                .msg(msg)
                .build();
    }

    public static <T> Resp<T> error(String msg) {
        return Resp.<T>builder()
                .code("-1")
                .msg(msg)
                .build();
    }

    public static <T> Resp<T> error(String msg, Throwable throwable) {
        return Resp.<T>builder()
                .code("-1")
                .msg(msg + "\t" + throwable.getLocalizedMessage())
                .build();
    }

    public static <T> Resp<T> unauthorized(String msg) {
        return Resp.<T>builder()
                .code("401")
                .msg(msg)
                .build();
    }

    public static <T> Resp<T> unauthorized() {
        return Resp.<T>builder()
                .code("401")
                .msg("unauthorized")
                .build();
    }

    public static <T> Resp<T> forbidden(String msg) {
        return Resp.<T>builder()
                .code("403")
                .msg(msg)
                .build();
    }
}
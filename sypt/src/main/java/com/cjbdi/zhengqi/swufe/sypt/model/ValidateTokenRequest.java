package com.cjbdi.zhengqi.swufe.sypt.model;

import lombok.Data;

import java.io.Serializable;

@Data
public class ValidateTokenRequest implements Serializable {

    private String token;

}

package com.cjbdi.zhengqi.swufe.sypt.model;

import com.cjbdi.zhengqi.swufe.sypt.entity.user.SysUser;
import lombok.Data;

import java.io.Serializable;

@Data
public class UserCreationRequest implements Serializable {

    private SysUser sysUser;

    private String password;

}

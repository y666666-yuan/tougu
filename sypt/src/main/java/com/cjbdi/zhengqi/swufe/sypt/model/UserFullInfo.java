package com.cjbdi.zhengqi.swufe.sypt.model;

import com.cjbdi.zhengqi.swufe.sypt.entity.user.SysPermission;
import com.cjbdi.zhengqi.swufe.sypt.entity.user.SysRole;
import com.cjbdi.zhengqi.swufe.sypt.entity.user.SysUser;
import lombok.Data;

import java.io.Serializable;
import java.util.List;

@Data
public class UserFullInfo implements Serializable {

    private SysUser sysUser;

    private List<SysRole> sysRoles;

    private List<SysPermission> sysPermissions;

}

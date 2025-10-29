package com.cjbdi.zhengqi.swufe.sypt.entity.user;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * <p>
 * 系统角色表
 * </p>
 *
 * @author liangboning
 * @since 2025-05-27
 */
@Getter
@Setter
@ToString
@TableName("sys_role")
public class SysRole implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 角色ID
     */
    @TableId(value = "role_id", type = IdType.AUTO)
    private Long roleId;

    /**
     * 角色标识（如 ADMIN、USER）
     */
    private String roleCode;

    /**
     * 角色名称（如 管理员、普通用户）
     */
    private String roleName;

    /**
     * 角色描述
     */
    private String description;

    private LocalDateTime createdAt;

    private LocalDateTime updatedAt;
}

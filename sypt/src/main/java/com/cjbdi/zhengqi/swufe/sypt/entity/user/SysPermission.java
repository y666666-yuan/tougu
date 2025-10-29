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
 * 系统权限表
 * </p>
 *
 * @author liangboning
 * @since 2025-05-27
 */
@Getter
@Setter
@ToString
@TableName("sys_permission")
public class SysPermission implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 自增ID
     */
    @TableId(value = "permission_id", type = IdType.AUTO)
    private Long permissionId;

    /**
     * 权限名称（如 user:read、user:write）
     */
    private String permissionName;

    /**
     * 权限描述
     */
    private String description;

    /**
     * 权限分类
     */
    private String category;

    private LocalDateTime createdAt;

    private LocalDateTime updatedAt;
}

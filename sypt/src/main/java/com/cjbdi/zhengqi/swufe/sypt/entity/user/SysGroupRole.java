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
 * 小组-角色关联表
 * </p>
 *
 * @author liangboning
 * @since 2025-05-27
 */
@Getter
@Setter
@ToString
@TableName("sys_group_role")
public class SysGroupRole implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 自增ID
     */
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    private Long groupId;

    private Long roleId;

    private LocalDateTime grantedAt;

    private Long grantedBy;
}

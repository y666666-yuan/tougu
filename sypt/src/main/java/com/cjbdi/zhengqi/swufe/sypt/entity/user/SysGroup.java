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
 * 系统用户组表
 * </p>
 *
 * @author liangboning
 * @since 2025-05-27
 */
@Getter
@Setter
@ToString
@TableName("sys_group")
public class SysGroup implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 小组ID-自增
     */
    @TableId(value = "group_id", type = IdType.AUTO)
    private Long groupId;

    private String groupName;

    private String description;

    private Long createdBy;

    private LocalDateTime createdAt;
}

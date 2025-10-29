package com.cjbdi.zhengqi.swufe.sypt.entity.user;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.io.Serializable;
import java.time.LocalDate;
import java.time.LocalDateTime;

/**
 * <p>
 * 系统用户表
 * </p>
 *
 * @author liangboning
 * @since 2025-05-27
 */
@Getter
@Setter
@ToString
@TableName("sys_user")
public class SysUser implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 用户ID
     */
    @TableId(value = "user_id", type = IdType.AUTO)
    private Long userId;

    private String username;

    /**
     * 昵称
     */
    private String nickname;

    /**
     * 性别，0=未知，1=男，2=女
     */
    private Byte gender;

    /**
     * 出生日期
     */
    private LocalDate dob;

    /**
     * 联系地址
     */
    private String address;

    private String email;

    private String phone;

    private String avatarUrl;

    private String passwordHash;

    /**
     * 账户状态，1=正常，2=禁用，3=冻结，4=注销
     */
    private Byte status;

    /**
     * 登录失败次数
     */
    private Short failedLoginAttempts;

    /**
     * 最后登录时间
     */
    private LocalDateTime lastLoginTime;

    private LocalDateTime createdAt;

    private LocalDateTime updatedAt;
}

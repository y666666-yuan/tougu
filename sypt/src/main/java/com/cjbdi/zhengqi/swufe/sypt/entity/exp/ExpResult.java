package com.cjbdi.zhengqi.swufe.sypt.entity.exp;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.io.Serializable;

/**
 * <p>
 * 实验结果表
 * </p>
 *
 * @author liangboning
 * @since 2025-07-08
 */
@Getter
@Setter
@ToString
@TableName("exp_result")
public class ExpResult implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 实验id（主键）
     */
    @TableId("exp_id")
    private String expId;

    /**
     * 实验名称
     */
    private String expName;

    /**
     * 算法类型
     */
    private String algorithmType;

    /**
     * 算法名称
     */
    private String algorithmName;

    /**
     * 创建人ID
     */
    private Long createdBy;

    /**
     * 创建时间（自动填充）
     */
    private String createdAt;
}

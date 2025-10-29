

-- 实验结果表
CREATE TABLE if not exists exp_result (
  exp_id varchar(100) unique not null comment '实验id（主键）',
  exp_name VARCHAR(200) NOT NULL comment '实验名称',
  algorithm_type VARCHAR(200) NOT NULL comment '算法类型',
  algorithm_name VARCHAR(200) NOT NULL comment '算法名称',
  created_by bigint unsigned NOT NULL comment '创建人ID',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP comment '创建时间（自动填充）',
  primary key (exp_id),
  FOREIGN KEY (created_by) REFERENCES sys_user(user_id) ON DELETE CASCADE
) engine=innodb auto_increment=0 default charset=utf8 comment='实验结果表';
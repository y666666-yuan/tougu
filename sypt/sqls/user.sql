-- 用户表
CREATE TABLE if not exists sys_user (
  user_id bigint unsigned auto_increment comment '用户ID（主键，自增，从10001开始）',
  username VARCHAR(50) UNIQUE NOT NULL comment '用户名（唯一标识，用于登录，支持字母、数字、下划线组合）',
  nickname VARCHAR(50) comment '昵称（用户自定义展示名称，可重复）',
  gender TINYINT UNSIGNED NOT NULL DEFAULT 0 comment '性别（0=未知，1=男，2=女）',
  dob DATE comment '出生日期',
  address TEXT comment '联系地址',
  email VARCHAR(100) UNIQUE comment '电子邮箱',
  phone VARCHAR(50) UNIQUE comment '手机号码',
  avatar_url VARCHAR(200) comment '头像地址',
  password_hash VARCHAR(300) NOT NULL comment '密码哈希值（使用BCrypt）',
  status TINYINT UNSIGNED NOT NULL DEFAULT 1 comment '账户状态（1=正常，2=禁用[管理员手动冻结]，3=冻结[自动锁定，如多次登录失败]，4=注销[不可逆删除前的软删除状态]）',
  failed_login_attempts smallint unsigned NOT NULL DEFAULT 0 comment '登录失败次数',
  last_login_time TIMESTAMP comment '最后登录时间',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP comment '创建时间',
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP comment '更新时间',
  primary key (user_id)
) engine=innodb auto_increment=10001 default charset=utf8 comment='系统用户表';

-- 小组表
CREATE TABLE if not exists sys_group (
  group_id bigint unsigned auto_increment comment '小组ID-自增（主键，用于标识用户组）',
  group_name VARCHAR(50) UNIQUE NOT NULL comment '小组名称（唯一标识，如“开发组”“测试组”）',
  description TEXT comment '小组描述（功能说明，如“负责项目开发的团队”）',
  created_by bigint unsigned NOT NULL comment '创建人ID（关联sys_user.user_id，记录谁创建了该小组）',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP comment '创建时间（自动填充，记录小组创建时间）',
  primary key (group_id),
  FOREIGN KEY (created_by) REFERENCES sys_user(user_id) ON DELETE CASCADE
) engine=innodb auto_increment=0 default charset=utf8 comment='系统用户组表';

-- 用户-小组关联表
CREATE TABLE if not exists sys_user_group (
  id bigint unsigned auto_increment comment '自增ID（主键，唯一标识关联记录）',
  user_id bigint unsigned NOT NULL comment '用户ID（关联sys_user.user_id，标识所属用户）',
  group_id bigint unsigned NOT NULL comment '小组ID（关联sys_group.group_id，标识所属小组）',
  joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP comment '加入时间（自动填充，记录用户加入小组的时间）',
  PRIMARY KEY (id),
  UNIQUE KEY uniq_user_group (user_id, group_id) comment '唯一约束：确保用户只能加入同一小组一次',
  FOREIGN KEY (user_id) REFERENCES sys_user(user_id) ON DELETE CASCADE,
  FOREIGN KEY (group_id) REFERENCES sys_group(group_id) ON DELETE CASCADE
) engine=innodb auto_increment=0 default charset=utf8 comment='用户-小组关联表';

-- 权限表
CREATE TABLE if not exists sys_permission (
  permission_id bigint unsigned auto_increment comment '自增ID（主键，唯一标识权限）',
  permission_name VARCHAR(50) UNIQUE NOT NULL COMMENT '权限名称（采用“资源:操作”格式，如 user:read、group:create，便于权限管理）',
  description TEXT COMMENT '权限描述（详细说明权限用途，如“读取用户信息的权限”）',
  category VARCHAR(50) COMMENT '权限分类（如“用户管理”“系统设置”，用于权限分组展示）',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间（自动填充，记录权限创建时间）',
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间（自动更新，记录权限最后修改时间）',
  PRIMARY KEY (permission_id)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 COMMENT='系统权限表';

-- 角色表
CREATE TABLE if not exists sys_role (
  role_id bigint unsigned auto_increment comment '角色ID（主键，唯一标识角色）',
  role_code VARCHAR(100) UNIQUE NOT NULL COMMENT '角色标识（唯一编码，如 admin、user，用于程序逻辑判断）',
  role_name VARCHAR(150) UNIQUE NOT NULL COMMENT '角色名称（用户友好名称，如“系统管理员”“普通用户”）',
  description TEXT COMMENT '角色描述（说明角色职责，如“拥有系统最高管理权限”）',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间（自动填充，记录角色创建时间）',
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间（自动更新，记录角色最后修改时间）',
  PRIMARY KEY (role_id)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 COMMENT='系统角色表';

-- 用户-角色关联表
CREATE TABLE if not exists sys_user_role (
  id bigint unsigned auto_increment comment '自增ID（主键，唯一标识关联记录）',
  user_id bigint unsigned NOT NULL comment '用户ID（关联sys_user.user_id，标识目标用户）',
  role_id bigint unsigned NOT NULL comment '角色ID（关联sys_role.role_id，标识分配的角色）',
  assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '分配时间（自动填充，记录角色分配时间）',
  assigned_by bigint unsigned NOT NULL COMMENT '分配人ID（关联sys_user.user_id，记录谁执行了角色分配操作）',
  PRIMARY KEY (id),
  UNIQUE KEY uniq_user_role (user_id, role_id),
  FOREIGN KEY (user_id) REFERENCES sys_user(user_id) ON DELETE CASCADE,
  FOREIGN KEY (role_id) REFERENCES sys_role(role_id) ON DELETE CASCADE,
  FOREIGN KEY (assigned_by) REFERENCES sys_user(user_id) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 COMMENT='用户-角色关联表';

-- 角色-权限关联表
CREATE TABLE if not exists sys_role_permission (
  id bigint unsigned auto_increment comment '自增ID（主键，唯一标识关联记录）',
  role_id bigint unsigned NOT NULL comment '角色ID（关联sys_role.role_id，标识目标角色）',
  permission_id bigint unsigned NOT NULL comment '权限ID（关联sys_permission.permission_id，标识授予的权限）',
  granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '授权时间（自动填充，记录权限授予时间）',
  granted_by bigint unsigned NOT NULL COMMENT '授权人ID（关联sys_user.user_id，记录谁执行了授权操作）',
  PRIMARY KEY (id),
  UNIQUE KEY uniq_role_permission (role_id, permission_id),
  FOREIGN KEY (role_id) REFERENCES sys_role(role_id) ON DELETE CASCADE,
  FOREIGN KEY (permission_id) REFERENCES sys_permission(permission_id) ON DELETE CASCADE,
  FOREIGN KEY (granted_by) REFERENCES sys_user(user_id) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 COMMENT='角色-权限关联表';

-- 小组-角色关联表
CREATE TABLE if not exists sys_group_role (
  id bigint unsigned auto_increment comment '自增ID（主键，唯一标识关联记录）',
  group_id bigint unsigned NOT NULL comment '小组ID（关联sys_group.group_id，标识目标小组）',
  role_id bigint unsigned NOT NULL comment '角色ID（关联sys_role.role_id，标识授予的角色）',
  granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP comment '授权时间（自动填充，记录角色授予小组的时间）',
  granted_by bigint unsigned NOT NULL comment '授权人ID（关联sys_user.user_id，记录谁执行了授权操作）',
  PRIMARY KEY (id),
  UNIQUE KEY uniq_group_role (group_id, role_id), -- 确保小组和权限组合唯一
  FOREIGN KEY (group_id) REFERENCES sys_group(group_id) ON DELETE CASCADE,
  FOREIGN KEY (role_id) REFERENCES sys_role(role_id) ON DELETE CASCADE,
  FOREIGN KEY (granted_by) REFERENCES sys_user(user_id) ON DELETE CASCADE
) engine=innodb auto_increment=0 default charset=utf8 comment='小组-角色关联表';
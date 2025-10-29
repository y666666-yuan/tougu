# 源码教程

教程以windows环境为主。

## 1.环境准备

| 序号 | 软件        | 版本      | 用途          |
|----|-----------|---------|-------------|
| 1  | mysql 数据库 | 5.7     | 数据库存储系统用户数据 |
| 2  | java      | 17      | 后端系统开发语言    |
| 3  | python    | 3.6     | 算法模块开发语言    |
| 4  | node      | 20.15.0 | 前端开发语言      |

## 2.IDE准备

python推荐 pycharm
java 推荐 IDEA
node 推荐 VSCODE

## 3. 项目介绍

| 序号 | 源码包          | 说明                                |
|----|--------------|-----------------------------------|
| 1  | sypt.zip     | java开发，使用springboot框架，负责系统用户登录等功能 |
| 2  | zntgsy.zip   | python开发， 使用flask框架，负责算法模块        |
| 3  | zntg-web.zip | vue框架开发，前端页面                      |

`zntg-web` 页面发起调用，所有前端接口均通过`sypt`进行鉴权，在调用算法模块`zntgsy`


## 4.源码启动

## 4.1 数据库建表
创建数据库表
```sql
create
database xcsypt;
CREATE TABLE xcsypt.`exp_result`
(
    `exp_id`         varchar(100) NOT NULL COMMENT '实验id（主键）',
    `exp_name`       varchar(200) NOT NULL COMMENT '实验名称',
    `algorithm_type` varchar(200)          DEFAULT NULL COMMENT '算法类型',
    `algorithm_name` varchar(200)          DEFAULT NULL COMMENT '算法名称',
    `exp_type_name`  varchar(200)          DEFAULT NULL COMMENT '实验类型名称',
    `exp_type_code`  bigint(20) DEFAULT NULL COMMENT '实验类型编码',
    `created_by`     bigint(20) unsigned NOT NULL COMMENT '创建人ID',
    `created_at`     timestamp    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间（自动填充）',
    PRIMARY KEY (`exp_id`),
    UNIQUE KEY `exp_id` (`exp_id`),
    KEY              `created_by` (`created_by`),
    CONSTRAINT `exp_result_ibfk_1` FOREIGN KEY (`created_by`) REFERENCES `sys_user` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='实验结果表';
CREATE TABLE xcsypt.`sys_group`
(
    `group_id`    bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '小组ID-自增（主键，用于标识用户组）',
    `group_name`  varchar(50) NOT NULL COMMENT '小组名称（唯一标识，如“开发组”“测试组”）',
    `description` text COMMENT '小组描述（功能说明，如“负责项目开发的团队”）',
    `created_by`  bigint(20) unsigned NOT NULL COMMENT '创建人ID（关联sys_user.user_id，记录谁创建了该小组）',
    `created_at`  timestamp   NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间（自动填充，记录小组创建时间）',
    PRIMARY KEY (`group_id`),
    UNIQUE KEY `group_name` (`group_name`),
    KEY           `created_by` (`created_by`),
    CONSTRAINT `sys_group_ibfk_1` FOREIGN KEY (`created_by`) REFERENCES `sys_user` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='系统用户组表';
CREATE TABLE xcsypt.`sys_group_role`
(
    `id`         bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID（主键，唯一标识关联记录）',
    `group_id`   bigint(20) unsigned NOT NULL COMMENT '小组ID（关联sys_group.group_id，标识目标小组）',
    `role_id`    bigint(20) unsigned NOT NULL COMMENT '角色ID（关联sys_role.role_id，标识授予的角色）',
    `granted_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '授权时间（自动填充，记录角色授予小组的时间）',
    `granted_by` bigint(20) unsigned NOT NULL COMMENT '授权人ID（关联sys_user.user_id，记录谁执行了授权操作）',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uniq_group_role` (`group_id`,`role_id`),
    KEY          `role_id` (`role_id`),
    KEY          `granted_by` (`granted_by`),
    CONSTRAINT `sys_group_role_ibfk_1` FOREIGN KEY (`group_id`) REFERENCES `sys_group` (`group_id`) ON DELETE CASCADE,
    CONSTRAINT `sys_group_role_ibfk_2` FOREIGN KEY (`role_id`) REFERENCES `sys_role` (`role_id`) ON DELETE CASCADE,
    CONSTRAINT `sys_group_role_ibfk_3` FOREIGN KEY (`granted_by`) REFERENCES `sys_user` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='小组-角色关联表';
CREATE TABLE xcsypt.`sys_permission`
(
    `permission_id`   bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID（主键，唯一标识权限）',
    `permission_name` varchar(50) NOT NULL COMMENT '权限名称（采用“资源:操作”格式，如 user:read、group:create，便于权限管理）',
    `description`     text COMMENT '权限描述（详细说明权限用途，如“读取用户信息的权限”）',
    `category`        varchar(50)          DEFAULT NULL COMMENT '权限分类（如“用户管理”“系统设置”，用于权限分组展示）',
    `created_at`      timestamp   NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间（自动填充，记录权限创建时间）',
    `updated_at`      timestamp   NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间（自动更新，记录权限最后修改时间）',
    PRIMARY KEY (`permission_id`),
    UNIQUE KEY `permission_name` (`permission_name`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8 COMMENT='系统权限表';
CREATE TABLE xcsypt.`sys_role`
(
    `role_id`     bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '角色ID（主键，唯一标识角色）',
    `role_code`   varchar(100) NOT NULL COMMENT '角色标识（唯一编码，如 admin、user，用于程序逻辑判断）',
    `role_name`   varchar(150) NOT NULL COMMENT '角色名称（用户友好名称，如“系统管理员”“普通用户”）',
    `description` text COMMENT '角色描述（说明角色职责，如“拥有系统最高管理权限”）',
    `created_at`  timestamp    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间（自动填充，记录角色创建时间）',
    `updated_at`  timestamp    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间（自动更新，记录角色最后修改时间）',
    PRIMARY KEY (`role_id`),
    UNIQUE KEY `role_code` (`role_code`),
    UNIQUE KEY `role_name` (`role_name`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8 COMMENT='系统角色表';
CREATE TABLE xcsypt.`sys_role_permission`
(
    `id`            bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID（主键，唯一标识关联记录）',
    `role_id`       bigint(20) unsigned NOT NULL COMMENT '角色ID（关联sys_role.role_id，标识目标角色）',
    `permission_id` bigint(20) unsigned NOT NULL COMMENT '权限ID（关联sys_permission.permission_id，标识授予的权限）',
    `granted_at`    timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '授权时间（自动填充，记录权限授予时间）',
    `granted_by`    bigint(20) unsigned NOT NULL COMMENT '授权人ID（关联sys_user.user_id，记录谁执行了授权操作）',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uniq_role_permission` (`role_id`,`permission_id`),
    KEY             `permission_id` (`permission_id`),
    KEY             `granted_by` (`granted_by`),
    CONSTRAINT `sys_role_permission_ibfk_1` FOREIGN KEY (`role_id`) REFERENCES `sys_role` (`role_id`) ON DELETE CASCADE,
    CONSTRAINT `sys_role_permission_ibfk_2` FOREIGN KEY (`permission_id`) REFERENCES `sys_permission` (`permission_id`) ON DELETE CASCADE,
    CONSTRAINT `sys_role_permission_ibfk_3` FOREIGN KEY (`granted_by`) REFERENCES `sys_user` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8 COMMENT='角色-权限关联表';
CREATE TABLE xcsypt.`sys_user`
(
    `user_id`               bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '用户ID（主键，自增，从10001开始）',
    `username`              varchar(50)  NOT NULL COMMENT '用户名（唯一标识，用于登录，支持字母、数字、下划线组合）',
    `nickname`              varchar(50)           DEFAULT NULL COMMENT '昵称（用户自定义展示名称，可重复）',
    `gender`                tinyint(3) unsigned NOT NULL DEFAULT '0' COMMENT '性别（0=未知，1=男，2=女）',
    `dob`                   date                  DEFAULT NULL COMMENT '出生日期',
    `address`               text COMMENT '联系地址',
    `email`                 varchar(100)          DEFAULT NULL COMMENT '电子邮箱',
    `phone`                 varchar(50)           DEFAULT NULL COMMENT '手机号码',
    `avatar_url`            varchar(200)          DEFAULT NULL COMMENT '头像地址',
    `password_hash`         varchar(300) NOT NULL COMMENT '密码哈希值（使用BCrypt）',
    `status`                tinyint(3) unsigned NOT NULL DEFAULT '1' COMMENT '账户状态（1=正常，2=禁用[管理员手动冻结]，3=冻结[自动锁定，如多次登录失败]，4=注销[不可逆删除前的软删除状态]）',
    `failed_login_attempts` smallint(5) unsigned NOT NULL DEFAULT '0' COMMENT '登录失败次数',
    `last_login_time`       timestamp    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后登录时间',
    `created_at`            timestamp    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at`            timestamp    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`user_id`),
    UNIQUE KEY `username` (`username`),
    UNIQUE KEY `email` (`email`),
    UNIQUE KEY `phone` (`phone`)
) ENGINE=InnoDB AUTO_INCREMENT=10003 DEFAULT CHARSET=utf8 COMMENT='系统用户表';
CREATE TABLE xcsypt.`sys_user_group`
(
    `id`        bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID（主键，唯一标识关联记录）',
    `user_id`   bigint(20) unsigned NOT NULL COMMENT '用户ID（关联sys_user.user_id，标识所属用户）',
    `group_id`  bigint(20) unsigned NOT NULL COMMENT '小组ID（关联sys_group.group_id，标识所属小组）',
    `joined_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '加入时间（自动填充，记录用户加入小组的时间）',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uniq_user_group` (`user_id`,`group_id`) COMMENT '唯一约束：确保用户只能加入同一小组一次',
    KEY         `group_id` (`group_id`),
    CONSTRAINT `sys_user_group_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `sys_user` (`user_id`) ON DELETE CASCADE,
    CONSTRAINT `sys_user_group_ibfk_2` FOREIGN KEY (`group_id`) REFERENCES `sys_group` (`group_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='用户-小组关联表';

CREATE TABLE xcsypt.`sys_user_role`
(
    `id`          bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID（主键，唯一标识关联记录）',
    `user_id`     bigint(20) unsigned NOT NULL COMMENT '用户ID（关联sys_user.user_id，标识目标用户）',
    `role_id`     bigint(20) unsigned NOT NULL COMMENT '角色ID（关联sys_role.role_id，标识分配的角色）',
    `assigned_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '分配时间（自动填充，记录角色分配时间）',
    `assigned_by` bigint(20) unsigned NOT NULL COMMENT '分配人ID（关联sys_user.user_id，记录谁执行了角色分配操作）',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uniq_user_role` (`user_id`,`role_id`),
    KEY           `role_id` (`role_id`),
    KEY           `assigned_by` (`assigned_by`),
    CONSTRAINT `sys_user_role_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `sys_user` (`user_id`) ON DELETE CASCADE,
    CONSTRAINT `sys_user_role_ibfk_2` FOREIGN KEY (`role_id`) REFERENCES `sys_role` (`role_id`) ON DELETE CASCADE,
    CONSTRAINT `sys_user_role_ibfk_3` FOREIGN KEY (`assigned_by`) REFERENCES `sys_user` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8 COMMENT='用户-角色关联表';
INSERT INTO xcsypt.sys_permission
(permission_id, permission_name, description, category, created_at, updated_at)
VALUES (1, 'ADMIN', '超级管理员权限', NULL, '2025-05-27 15:23:50', '2025-05-27 15:23:50');
INSERT INTO xcsypt.sys_role
    (role_id, role_code, role_name, description, created_at, updated_at)
VALUES (1, 'ADMIN', '管理员', '超级管理员角色', '2025-05-27 15:25:36', '2025-05-27 15:25:36');
INSERT INTO xcsypt.sys_role_permission
    (id, role_id, permission_id, granted_at, granted_by)
VALUES (1, 1, 1, '2025-05-27 15:25:36', 10001);
INSERT INTO xcsypt.sys_user
(user_id, username, nickname, gender, dob, address, email, phone, avatar_url, password_hash, status,
 failed_login_attempts, last_login_time, created_at, updated_at)
VALUES (10001, 'admin', NULL, 0, NULL, NULL, NULL, NULL, NULL,
        '$2a$10$znOd87Y8BsxglmRLJBW0PueIBceSiYkzdZM.971ku6xgSkijLGTWK', 1, 0, '2025-05-27 15:25:36',
        '2025-05-27 15:25:36', '2025-05-27 15:25:36');
INSERT INTO xcsypt.sys_user_role
    (id, user_id, role_id, assigned_at, assigned_by)
VALUES (1, 10001, 1, '2025-05-27 15:25:36', 10001);

```

### zntg-web 
解压`zntg-web.zip`包，进入对应的文件夹，参见文件夹中的README.md即可启动项目

## sypt
解压`sypt.zip`包,使用`IDEA`编辑器打开项目，会自动拉取依赖。依赖拉完，修改配置文件`sypt/src/main/resources/application-test.yml`和`sypt/src/main/resources/application-xg.yml`中数据库配置
```yaml
spring:
  datasource:
    url: jdbc:mysql://你的数据库IP:数据库端口/xcsypt?useUnicode=true&zeroDateTimeBehavior=convertToNull&autoReconnect=true&characterEncoding=utf-8
    username: 数据库用户名
    password: 数据库密码
    driver-class-name: com.mysql.cj.jdbc.Driver

forwarding:
  routes:
    - id: algorithm-service
      uri: http://算法模块IP:算法模块端口
      paths:
        - /sypt/algorithm-api/**
      stripPrefix: 2
```

## zntgsy
解压`zntgsy.zip`包，使用`PyCharm`编辑器打开项目，安装依赖（软件应该会自动安装，如果没有安装，使用命令`install -r requirements.txt`），直接启动即可。

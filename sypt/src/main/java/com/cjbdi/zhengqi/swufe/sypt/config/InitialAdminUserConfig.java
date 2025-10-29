package com.cjbdi.zhengqi.swufe.sypt.config;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.cjbdi.zhengqi.swufe.sypt.entity.user.*;
import com.cjbdi.zhengqi.swufe.sypt.prop.config.InitialAdminUserPropConfig;
import com.cjbdi.zhengqi.swufe.sypt.service.user.*;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

import java.util.List;

@Slf4j
@Component
public class InitialAdminUserConfig {

    @Autowired
    private InitialAdminUserPropConfig initialAdminUserPropConfig;

    @Autowired
    private ISysUserService sysUsersService;

    @Autowired
    private ISysPermissionService sysPermissionsService;

    @Autowired
    private ISysRoleService sysRolesService;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private ISysRolePermissionService sysRolePermissionService;

    @Autowired
    private ISysUserPermissionService sysUserPermissionsService;

    @Autowired
    private ISysUserRoleService sysUserRoleService;

    private SysRole initialAdminRole() {
        LambdaQueryWrapper<SysRole> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(SysRole::getRoleCode, "ADMIN");
        List<SysRole> sysRolesList = sysRolesService.list(queryWrapper);
        if (CollUtil.isEmpty(sysRolesList)) {
            SysRole sysRole = new SysRole();
            sysRole.setRoleName("管理员");
            sysRole.setRoleCode("ADMIN");
            sysRole.setDescription("超级管理员角色");
            if (sysRolesService.save(sysRole)) {
                log.info("创建超级管理员角色成功");
            } else {
                log.info("创建超级管理员角色失败");
            }
        }
        sysRolesList = sysRolesService.list(queryWrapper);
        if (CollUtil.isEmpty(sysRolesList)) {
            log.warn("未找到超级管理员角色");
            return null;
        }
        return sysRolesList.get(0);
    }

    private List<SysPermission> initialAdminPermissions() {
        List<String> permissionNames = List.of(
                "admin:user:view",
                "admin:user:create",
                "admin:user:update",
                "admin:user:delete"
        );
        return permissionNames.stream().map(this::initialAdminPermission).toList();
    }

    private SysPermission initialAdminPermission(String permissionName) {
        LambdaQueryWrapper<SysPermission> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(SysPermission::getPermissionName, permissionName);
        List<SysPermission> sysPermissionsList = sysPermissionsService.list(queryWrapper);
        if (CollUtil.isEmpty(sysPermissionsList)) {
            SysPermission sysPermission = new SysPermission();
            sysPermission.setPermissionName(permissionName);
            sysPermission.setDescription(permissionName);
            if (sysPermissionsService.save(sysPermission)) {
                log.info("创建超级管理员权限成功");
            } else {
                log.info("创建超级管理员权限失败");
            }
        }
        sysPermissionsList = sysPermissionsService.list(queryWrapper);
        if (CollUtil.isEmpty(sysPermissionsList)) {
            log.warn("未找到超级管理员权限");
            return null;
        }
        return sysPermissionsList.get(0);
    }

    private SysUser initialAdminUser(SysRole sysRole) {
        if (StrUtil.isBlank(initialAdminUserPropConfig.getUsername()) || StrUtil.isBlank(initialAdminUserPropConfig.getPassword())) {
            log.warn("初始化超级管理员用户失败，配置中用户名或密码为空");
            return null;
        }
        String username = initialAdminUserPropConfig.getUsername();
        String password = initialAdminUserPropConfig.getPassword();
        SysUser storedUser = sysUsersService.selectByUsername(username);
        if (storedUser != null) {
            log.info("超级管理员用户存在");
            return storedUser;
        }
        SysUser sysUser = new SysUser();
        sysUser.setUsername(username);
        String hashPassword = passwordEncoder.encode(password);
        sysUser.setPasswordHash(hashPassword);
        sysUser.setUserId(null);
        sysUser.setStatus((byte) 1);
        if (sysUsersService.save(sysUser)) {
            initialAdminUserRole(sysRole, sysUser);
            log.info("创建超级管理员用户成功");
            return sysUser;
        }
        log.warn("创建超级管理员用户失败");
        return null;
    }

    private SysUserRole initialAdminUserRole(SysRole sysRole, SysUser sysUser) {
        if (sysRole == null || sysUser == null) {
            log.warn("初始化超级管理员用户角色失败，未找到超级管理员角色或用户");
            return null;
        }
        LambdaQueryWrapper<SysUserRole> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(SysUserRole::getUserId, sysUser.getUserId());
        queryWrapper.eq(SysUserRole::getRoleId, sysRole.getRoleId());
        List<SysUserRole> sysUserRolesList = sysUserRoleService.list(queryWrapper);
        if (CollUtil.isEmpty(sysUserRolesList)) {
            SysUserRole sysUserRole = new SysUserRole();
            sysUserRole.setUserId(sysUser.getUserId());
            sysUserRole.setRoleId(sysRole.getRoleId());
            sysUserRole.setAssignedBy(sysUser.getUserId());
            if (sysUserRoleService.save(sysUserRole)) {
                log.info("创建超级管理员用户角色成功");
            } else {
                log.info("创建超级管理员用户角色失败");
            }
        }
        sysUserRolesList = sysUserRoleService.list(queryWrapper);
        if (CollUtil.isEmpty(sysUserRolesList)) {
            log.warn("未找到超级管理员用户角色");
            return null;
        }
        return sysUserRolesList.get(0);
    }

    private List<SysRolePermission> connectRelationAdminRoleToPermission(SysUser adminUser,
                                                                   SysRole sysRole,
                                                                   List<SysPermission> sysPermissions) {
        if (sysRole == null || CollUtil.isEmpty(sysPermissions)) {
            log.warn("初始化超级管理员角色权限失败，未找到超级管理员角色或权限");
            return null;
        }
        LambdaQueryWrapper<SysRolePermission> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(SysRolePermission::getRoleId, sysRole.getRoleId());
        queryWrapper.in(SysRolePermission::getPermissionId, sysPermissions.stream().map(SysPermission::getPermissionId).toList());
        List<SysRolePermission> sysRolePermissionsList = sysRolePermissionService.list(queryWrapper);
        for (SysPermission sysPermission : sysPermissions) {
            if (sysRolePermissionsList.stream().noneMatch(srp -> srp.getPermissionId().equals(sysPermission.getPermissionId()))) {
                SysRolePermission sysRolePermission = new SysRolePermission();
                sysRolePermission.setRoleId(sysRole.getRoleId());
                sysRolePermission.setPermissionId(sysPermission.getPermissionId());
                sysRolePermission.setGrantedBy(adminUser.getUserId());
                if (sysRolePermissionService.save(sysRolePermission)) {
                    log.info("创建超级管理员角色权限成功");
                } else {
                    log.info("创建超级管理员角色权限失败");
                }
            }
        }
        sysRolePermissionsList = sysRolePermissionService.list(queryWrapper);
        if (sysRolePermissionsList.size() != sysPermissions.size()) {
            log.warn("未找到超级管理员角色权限");
            return sysRolePermissionsList;
        }
        return sysRolePermissionsList;
    }

    @PostConstruct
    public void init() {
        String username = initialAdminUserPropConfig.getUsername();
        String password = initialAdminUserPropConfig.getPassword();
        if (StrUtil.isBlank(username) || StrUtil.isBlank(password)) {
            log.warn("初始化超级管理员用户失败，配置中用户名或密码为空");
            return;
        }
        SysUser storedUser = sysUsersService.selectByUsername(username);
        if (storedUser != null) {
            log.info("超级管理员用户存在");
            return;
        }

        List<SysPermission> sysAdminPermissions = initialAdminPermissions();
        SysRole sysAdminRole = initialAdminRole();
        SysUser sysUser = initialAdminUser(sysAdminRole);
        if (sysUser != null) {
            log.info("创建超级管理员用户成功" + JSONUtil.toJsonStr(sysUser));
        }
        connectRelationAdminRoleToPermission(sysUser, sysAdminRole, sysAdminPermissions);
        log.info("创建超级管理员用户结束");
    }

}

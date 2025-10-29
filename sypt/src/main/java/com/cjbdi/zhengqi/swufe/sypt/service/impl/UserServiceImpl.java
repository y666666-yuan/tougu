package com.cjbdi.zhengqi.swufe.sypt.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.cjbdi.zhengqi.swufe.sypt.dto.LoginRequest;
import com.cjbdi.zhengqi.swufe.sypt.entity.User;
import com.cjbdi.zhengqi.swufe.sypt.entity.user.*;
import com.cjbdi.zhengqi.swufe.sypt.service.UserService;
import com.cjbdi.zhengqi.swufe.sypt.service.user.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class UserServiceImpl implements UserService {

    @Autowired
    private ISysUserService sysUsersService;

    @Autowired
    private ISysUserRoleService sysUserRoleService;

    @Autowired
    private ISysUserPermissionService sysUserPermissionService;

    @Autowired
    private ISysPermissionService sysPermissionService;

    @Autowired
    private ISysRoleService sysRoleService;

    @Autowired
    private PasswordEncoder passwordEncoder;

    private List<String> permissionsByUser(SysUser sysUser) {
        List<String> permissions = CollUtil.newArrayList();
        {
            LambdaQueryWrapper<SysUserPermission> queryWrapper = new LambdaQueryWrapper<>();
            queryWrapper.eq(SysUserPermission::getUserId, sysUser.getUserId());
            List<SysUserPermission> sysUserPermissionsList = sysUserPermissionService.list(queryWrapper);
            if (CollUtil.isNotEmpty(sysUserPermissionsList)) {
                List<Long> permissionIds = CollUtil.newArrayList();
                for (SysUserPermission sysUserPermission : sysUserPermissionsList) {
                    permissionIds.add(sysUserPermission.getPermissionId());
                }
                LambdaQueryWrapper<SysPermission> queryWrapper2 = new LambdaQueryWrapper<>();
                queryWrapper2.in(SysPermission::getPermissionId, permissionIds);
                List<SysPermission> sysPermissionsList = sysPermissionService.list(queryWrapper2);
                if (CollUtil.isNotEmpty(sysPermissionsList)) {
                    for (SysPermission sysPermission : sysPermissionsList) {
                        permissions.add(sysPermission.getPermissionName());
                    }
                }
            }
        }
        return permissions;
    }

    private List<String> rolesByUser(SysUser sysUser) {
        List<String> roles = CollUtil.newArrayList();
        LambdaQueryWrapper<SysUserRole> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(SysUserRole::getUserId, sysUser.getUserId());
        List<SysUserRole> sysUserRolesList = sysUserRoleService.list(queryWrapper);
        for (SysUserRole sysUserRole : sysUserRolesList) {
            LambdaQueryWrapper<SysRole> queryWrapper2 = new LambdaQueryWrapper<>();
            queryWrapper2.eq(SysRole::getRoleId, sysUserRole.getRoleId());
            List<SysRole> sysRolesList = sysRoleService.list(queryWrapper2);
            if (CollUtil.isNotEmpty(sysRolesList)) {
                for (SysRole sysRole : sysRolesList) {
                    roles.add(sysRole.getRoleCode());
                }
            }
        }
        return roles;
    }

    @Override
    public Optional<User> authenticate(LoginRequest request) throws Exception {
        String username = request.getUsername();
        String password = request.getPassword();
        if (StrUtil.isBlank(username) || StrUtil.isBlank(password)) {
            throw new Exception("用户名或密码不能为空");
        }
        LambdaQueryWrapper<SysUser> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(SysUser::getUsername, username);
        List<SysUser> sysUsers = sysUsersService.list(queryWrapper);
        if (CollUtil.isEmpty(sysUsers)) {
            throw new Exception("用户不存在");
        }
        SysUser sysUser = sysUsers.get(0);
        String hashPassword = sysUser.getPasswordHash();
        if (!passwordEncoder.matches(password, hashPassword)) {
            throw new Exception("密码错误");
        }
        User user = new User();
        user.setId(sysUser.getUserId());
        user.setUsername(sysUser.getUsername());
        user.setNickname(sysUser.getNickname());
        user.setRoles(rolesByUser(sysUser));
        return Optional.of(user);
    }
}

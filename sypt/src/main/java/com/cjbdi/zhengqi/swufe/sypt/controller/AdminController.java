package com.cjbdi.zhengqi.swufe.sypt.controller;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.BooleanUtil;
import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.cjbdi.zhengqi.swufe.sypt.entity.user.SysRole;
import com.cjbdi.zhengqi.swufe.sypt.entity.user.SysUser;
import com.cjbdi.zhengqi.swufe.sypt.entity.user.SysUserRole;
import com.cjbdi.zhengqi.swufe.sypt.model.*;
import com.cjbdi.zhengqi.swufe.sypt.service.user.*;
import io.swagger.v3.oas.annotations.Hidden;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.List;

@Tag(name = "管理员接口")
@RestController
@Hidden
@RequestMapping("/admin")
public class AdminController {

    @Autowired
    private ISysUserService sysUsersService;

    @Autowired
    private ISysUserRoleService sysUserRoleService;

    @Autowired
    private ISysRoleService sysRoleService;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private ISysRolePermissionService sysRolePermissionService;

    @Autowired
    private ISysPermissionService sysPermissionService;

    @Operation(
            summary = "创建用户接口",
            description = "创建用户接口，返回用户信息及JWT令牌",
            requestBody = @io.swagger.v3.oas.annotations.parameters.RequestBody(
                    description = "用户创建请求体",
                    required = true,
                    content = @io.swagger.v3.oas.annotations.media.Content(
                            mediaType = "application/json",
                            schema = @io.swagger.v3.oas.annotations.media.Schema(
                                    implementation = UserCreationRequest.class
                            ),
                            examples = @io.swagger.v3.oas.annotations.media.ExampleObject(
                                    name = "User Creation Request",
                                    value = "{\"username\": \"test\", \"password\": \"test\"}"
                            )
                    )
            )
    )
    @PreAuthorize("hasAuthority('admin:user:create')")
    @PostMapping("/user/create")
    public Resp<Object> createUser(@RequestBody UserCreationRequest userCreationRequest) {
        if (userCreationRequest.getSysUser() == null) {
            return Resp.error("用户信息不能为空");
        }
        String username = userCreationRequest.getSysUser().getUsername();
        if (StrUtil.isBlank(username)) {
            return Resp.error("用户名不能为空");
        }
        String password = userCreationRequest.getPassword();
        if (StrUtil.isBlank(password)) {
            return Resp.error("密码不能为空");
        }
        SysUser storedUser = sysUsersService.selectByUsername(username);
        if (storedUser != null) {
            return Resp.error("用户名已存在");
        }
        SysUser sysUser = userCreationRequest.getSysUser();
        sysUser.setUsername(username);
        sysUser.setPasswordHash(passwordEncoder.encode(password));
        sysUser.setUserId(null);
        sysUser.setStatus((byte) 1);
        sysUser.setCreatedAt(null);
        sysUser.setUpdatedAt(null);
        sysUser.setFailedLoginAttempts((short)0);
        sysUser.setLastLoginTime(null);

        boolean createResult = sysUsersService.save(sysUser);
        if (BooleanUtil.isFalse(createResult)) {
            return Resp.error("创建用户失败");
        }
        return Resp.successMsg("创建成功");
    }

    @Operation(
            summary = "删除用户接口",
            description = "删除用户接口，返回用户信息及JWT令牌",
            requestBody = @io.swagger.v3.oas.annotations.parameters.RequestBody(
                    description = "用户删除请求体",
                    required = true,
                    content = @io.swagger.v3.oas.annotations.media.Content(
                            mediaType = "application/json",
                            schema = @io.swagger.v3.oas.annotations.media.Schema(
                                    implementation = UserDeletionRequest.class
                            ),
                            examples = @io.swagger.v3.oas.annotations.media.ExampleObject(
                                    name = "User Deletion Request",
                                    value = "{\"username\": \"test\"}"
                            )
                    )
            )
    )
    @PreAuthorize("hasAuthority('admin:user:delete')")
    @PostMapping("/user/delete")
    public Resp<Object> deleteUser(@RequestBody UserDeletionRequest userDeletionRequest) {
        return Resp.successMsg("删除成功");
    }

    @Operation(
            summary = "分页查询用户接口",
            description = "分页查询用户接口，返回用户信息及JWT令牌",
            requestBody = @io.swagger.v3.oas.annotations.parameters.RequestBody(
                    description = "分页查询用户请求体",
                    required = true,
                    content = @io.swagger.v3.oas.annotations.media.Content(
                            mediaType = "application/json",
                            schema = @io.swagger.v3.oas.annotations.media.Schema(
                                    implementation = PagedUsersRequest.class
                            ),
                            examples = @io.swagger.v3.oas.annotations.media.ExampleObject(
                                    name = "Paged Users Request",
                                    value = "{\"pageNum\": 1, \"pageSize\": 10}"
                            )
                    )
            )
    )
    @PreAuthorize("hasAuthority('admin:user:view')")
    @PostMapping("/user/page")
    public Resp<PagedUserFullInfo> pageUsers(@RequestBody PagedUsersRequest pagedUsersRequest) {
        Page<SysUser> page = new Page<>(pagedUsersRequest.getPageNum(), pagedUsersRequest.getPageSize());
        Page<SysUser> sysUsers = sysUsersService.page(page);
        List<UserFullInfo> userFullInfoList = new ArrayList<>();
        sysUsers.getRecords().forEach(sysUser -> {
            sysUser.setPasswordHash(null);
            UserFullInfo userFullInfo = new UserFullInfo();
            userFullInfo.setSysUser(sysUser);
            userFullInfo.setSysRoles(new ArrayList<>());
            {
                LambdaQueryWrapper<SysUserRole> queryWrapper = new LambdaQueryWrapper<>();
                queryWrapper.eq(SysUserRole::getUserId, sysUser.getUserId());
                List<SysUserRole> sysUserRolesList = sysUserRoleService.list(queryWrapper);
                for (SysUserRole sysUserRole : sysUserRolesList) {
                    LambdaQueryWrapper<SysRole> queryWrapper2 = new LambdaQueryWrapper<>();
                    queryWrapper2.eq(SysRole::getRoleId, sysUserRole.getRoleId());
                    List<SysRole> sysRolesList = sysRoleService.list(queryWrapper2);
                    userFullInfo.getSysRoles().addAll(sysRolesList);
                }
            }
            userFullInfoList.add(userFullInfo);
        });
        PagedUserFullInfo pagedUserFullInfo = new PagedUserFullInfo();
        pagedUserFullInfo.setUsers(userFullInfoList);
        pagedUserFullInfo.setTotal(sysUsers.getTotal());
        pagedUserFullInfo.setCurrent(sysUsers.getCurrent());
        pagedUserFullInfo.setSize(sysUsers.getSize());

        return Resp.success(pagedUserFullInfo);
    }

    @Operation(
            summary = "分页查询角色接口",
            description = "分页查询角色接口，返回角色信息及JWT令牌",
            requestBody = @io.swagger.v3.oas.annotations.parameters.RequestBody(
                    description = "分页查询角色请求体",
                    required = true,
                    content = @io.swagger.v3.oas.annotations.media.Content(
                            mediaType = "application/json",
                            schema = @io.swagger.v3.oas.annotations.media.Schema(
                                    implementation = Page.class
                            ),
                            examples = {
                                    @ExampleObject(
                                            name = "分页查询角色请求体",
                                            value = "{\"current\":1,\"size\":10}"
                                    )
                            }
                    )
            )
    )
    @PreAuthorize("hasAuthority('admin:role:view')")
    @PostMapping("/role/page")
    public Resp<Page<SysRole>> pageRoles(@RequestBody Page<SysRole> pagedRolesRequest) {
        Page<SysRole> sysRoles = sysRoleService.page(pagedRolesRequest);
        return Resp.success(sysRoles);
    }

    @Operation(summary = "创建角色")
    @PreAuthorize("hasAuthority('admin:role:create')")
    @PostMapping("/role/create")
    public Resp<SysRole> createRole(@RequestBody SysRole sysRole) {
        String roleCode = sysRole.getRoleCode();
        if (StrUtil.isBlank(roleCode)) {
            return Resp.error("roleCode不能为空");
        }
        if (sysRole.getRoleId() != null) {
            return Resp.error("不能指定roleId");
        }
        if (StrUtil.isBlank(sysRole.getRoleName())) {
            return Resp.error("roleName不能为空");
        }
        LambdaQueryWrapper<SysRole> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(SysRole::getRoleCode, roleCode);
        List<SysRole> sysRolesList = sysRoleService.list(queryWrapper);
        if (CollUtil.isNotEmpty(sysRolesList)) {
            return Resp.error("角色已存在");
        }
        return sysRoleService.save(sysRole) ? Resp.success(sysRole) : Resp.successMsg("创建角色失败");
    }

    @Operation(summary = "删除角色")
    @PreAuthorize("hasAuthority('admin:role:delete')")
    @PostMapping("/role/delete")
    public Resp deleteRole(@RequestBody SysRole sysRole) {
        if (sysRole == null || sysRole.getRoleId() == null) {
            return Resp.success("roleId不能为空");
        }
        try {
            return sysRoleService.removeById(sysRole.getRoleId()) ? Resp.success() : Resp.error("删除角色失败");
        } catch (Exception e) {
            return Resp.success("删除角色失败", e);
        }
    }



}

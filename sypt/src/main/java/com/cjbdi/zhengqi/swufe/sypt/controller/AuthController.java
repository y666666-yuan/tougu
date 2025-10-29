package com.cjbdi.zhengqi.swufe.sypt.controller;

import com.cjbdi.zhengqi.swufe.sypt.dto.LoginRequest;
import com.cjbdi.zhengqi.swufe.sypt.entity.User;
import com.cjbdi.zhengqi.swufe.sypt.model.Resp;
import com.cjbdi.zhengqi.swufe.sypt.model.ValidateTokenRequest;
import com.cjbdi.zhengqi.swufe.sypt.security.JwtService;
import com.cjbdi.zhengqi.swufe.sypt.service.UserService;
import io.swagger.v3.oas.annotations.Hidden;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Optional;

@Tag(name = "认证接口")
@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final UserService userService;

    @Autowired
    private JwtService jwtService;

    public AuthController(UserService userService) {
        this.userService = userService;
    }

    @Operation(
            summary = "用户登录接口",
            description = "用户登录接口，返回用户信息及JWT令牌",
            requestBody = @io.swagger.v3.oas.annotations.parameters.RequestBody(
                    description = "用户登录请求体",
                    required = true,
                    content = @io.swagger.v3.oas.annotations.media.Content(
                            mediaType = "application/json",
                            schema = @io.swagger.v3.oas.annotations.media.Schema(
                                    implementation = LoginRequest.class
                            ),
                            examples = {
                                    @io.swagger.v3.oas.annotations.media.ExampleObject(
                                            name = "管理员",
                                            value = "{\"username\": \"admin\", \"password\": \"admin123\"}"
                                    ),
                                    @io.swagger.v3.oas.annotations.media.ExampleObject(
                                            name = "用户-存在-密码正确",
                                            value = "{\"username\": \"test\", \"password\": \"test1\"}"
                                    ),
                                    @io.swagger.v3.oas.annotations.media.ExampleObject(
                                            name = "用户-存在-密码错误",
                                            value = "{\"username\": \"test\", \"password\": \"test2\"}"
                                    ),
                                    @io.swagger.v3.oas.annotations.media.ExampleObject(
                                            name = "用户-不存在",
                                            value = "{\"username\": \"test2\", \"password\": \"test\"}"
                                    )
                            }
                    )
            )
    )
    @PostMapping("/login")
    public Resp<User> login(@RequestBody LoginRequest request) {
        Optional<User> userOptional = null;
        try {
            userOptional = userService.authenticate(request);
        } catch (Exception e) {
            return Resp.error(e.getLocalizedMessage());
        }
        if (userOptional.isPresent()) {
            User user = userOptional.get();
            String token = jwtService.generateToken(user);
            user.setToken(token);
            return Resp.success(user);
        } else {
            return Resp.unauthorized("Invalid credentials");
        }
    }

    @Operation(
            summary = "验证令牌接口",
            description = "验证令牌接口，返回令牌是否合法"
    )
    @PostMapping("/validate-token")
    public Resp<Object> validateToken(@RequestBody ValidateTokenRequest validateTokenRequest) {
        try {
            if (!jwtService.validateToken(validateTokenRequest.getToken())) {
                return Resp.success(null);
            }
            return Resp.success(true);
        } catch (Exception e) {
            return Resp.success(false, e);
        }
    }

    @Hidden
    @Operation(
            summary = "用户登出接口",
            description = "用户登出接口，返回登出信息"
    )
    @PostMapping("/logout")
    public Resp<Object> logout() {
        return Resp.successMsg("Logged out successfully");
    }
}
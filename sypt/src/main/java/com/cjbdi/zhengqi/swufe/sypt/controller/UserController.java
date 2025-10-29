package com.cjbdi.zhengqi.swufe.sypt.controller;

import com.cjbdi.zhengqi.swufe.sypt.entity.User;
import com.cjbdi.zhengqi.swufe.sypt.model.Resp;
import com.cjbdi.zhengqi.swufe.sypt.security.JwtService;
import io.swagger.v3.oas.annotations.Hidden;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Tag(name = "用户接口")
@RestController
@Hidden
@RequestMapping("/api/user")
public class UserController {

    @Autowired
    private JwtService jwtService;

    @GetMapping("/info")
    public Resp<User> userInfo(HttpServletRequest httpServletRequest) {
        User user = jwtService.getUserFromRequest(httpServletRequest);
        if (user == null) {
            return Resp.unauthorized();
        }
        return Resp.success(user);
    }

}

package com.cjbdi.zhengqi.swufe.sypt.controller;

import com.cjbdi.zhengqi.swufe.sypt.model.Resp;
import io.swagger.v3.oas.annotations.Hidden;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Tag(name = "示例接口")
@RestController
@Hidden
@RequestMapping("/demo")
public class DemoController {

    @Operation(
            summary = "需要登录才能访问的接口",
            description = "需要登录才能访问的接口",
            security = @SecurityRequirement(
                    name = "Bearer",
                    scopes = "Bearer"
            )
    )
    @GetMapping("/need-login")
    public Resp<Object> demo() {
        return Resp.successMsg("success");
    }

    @Operation(
            summary = "不需要登录就能访问的接口",
            description = "不需要登录就能访问的接口"
    )
    @GetMapping("/not-need-login")
    public Resp<Object> demoNoLogin() {
        return Resp.successMsg("success");
    }

}

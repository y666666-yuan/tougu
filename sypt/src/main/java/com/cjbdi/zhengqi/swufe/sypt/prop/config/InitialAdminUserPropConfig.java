package com.cjbdi.zhengqi.swufe.sypt.prop.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "admin.initial")
@Data
public class InitialAdminUserPropConfig {

    private String username;
    private String password;

}

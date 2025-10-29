package com.cjbdi.zhengqi.swufe.sypt.prop.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "auth")
@Data
public class AuthConfig {

    // 令牌密钥
    private String jwtSecretKey = "RYGomzMvwTWtn7C8uQ6anBugXRqoVRAKPDGaGiK+b3E=";

    // 令牌过期时间
    private Long jwtExpiration = 24 * 60 * 60 * 1000L;

}

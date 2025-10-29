package com.cjbdi.zhengqi.swufe.sypt.prop.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
@ConfigurationProperties(prefix = "auth.ignores")
@Data
public class AuthIgnoreUrlsConfig {

    private List<String> urls;

}

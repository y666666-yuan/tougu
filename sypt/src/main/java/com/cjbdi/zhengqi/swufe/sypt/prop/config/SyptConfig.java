package com.cjbdi.zhengqi.swufe.sypt.prop.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.io.Serializable;

@Configuration
@ConfigurationProperties(prefix = "sypt")
@Data
public class SyptConfig implements Serializable {

    private String token;

    private String baseUrl;

}

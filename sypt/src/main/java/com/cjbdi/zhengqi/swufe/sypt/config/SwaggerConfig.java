package com.cjbdi.zhengqi.swufe.sypt.config;

import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.security.SecurityRequirement;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.Collections;

@Configuration
public class SwaggerConfig {

    @Bean
    public OpenAPI openAPI() {
        return new OpenAPI()
                .info(new Info().title("西南财经大学-智能投顾创新实验平台"))
                .components(
                        new Components()
                                .addSecuritySchemes(
                                        "Bearer",
                                        new io.swagger.v3.oas.models.security.SecurityScheme()
                                                .type(io.swagger.v3.oas.models.security.SecurityScheme.Type.HTTP)
                                                .scheme("bearer")
                                                .bearerFormat("JWT")
                                                .in(io.swagger.v3.oas.models.security.SecurityScheme.In.HEADER)
                                                .name("Authorization")
                                )
                )
                .security(Collections.singletonList(
                        new SecurityRequirement().addList("Bearer")
                ));
    }

}

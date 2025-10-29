package com.cjbdi.zhengqi.swufe.sypt.config;

import cn.hutool.core.collection.CollUtil;
import com.cjbdi.zhengqi.swufe.sypt.filter.GatewayFilter;
import com.cjbdi.zhengqi.swufe.sypt.model.Resp;
import com.cjbdi.zhengqi.swufe.sypt.prop.config.AuthIgnoreUrlsConfig;
import com.cjbdi.zhengqi.swufe.sypt.security.SecurityFilter;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.security.web.util.matcher.AntPathRequestMatcher;

import java.util.List;

@Configuration
@EnableWebSecurity
@EnableMethodSecurity(prePostEnabled = true)
public class SecurityConfig {

    @Autowired
    private SecurityFilter securityFilter;

    @Autowired
    private GatewayFilter gatewayFilter;

    @Autowired
    private AuthIgnoreUrlsConfig authIgnoreUrlsConfig;

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        List<String> ignoreUrls = authIgnoreUrlsConfig.getUrls();
        ignoreUrls.add("/swagger-ui/**");
        ignoreUrls.add("/v3/api-docs/**");
        ignoreUrls = CollUtil.distinct(ignoreUrls);
        List<AntPathRequestMatcher> matchers = ignoreUrls.stream()
                .map(AntPathRequestMatcher::new)
                .toList();
        http
                .csrf(AbstractHttpConfigurer::disable)
                .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers(
                                matchers.toArray(new AntPathRequestMatcher[0])
                        ).permitAll()
                        .anyRequest().authenticated()
                )
                .exceptionHandling(ex -> ex
                        .authenticationEntryPoint((request, response, authException) -> {
                            response.setContentType(MediaType.APPLICATION_JSON_VALUE);
                            response.setStatus(401);
                            ObjectMapper mapper = new ObjectMapper();
                            Resp<Object> body = Resp.unauthorized(request.getServletPath());
                            mapper.writeValue(response.getOutputStream(), body);
                        })
                        .accessDeniedHandler((request, response, accessDeniedException) -> {
                            response.setContentType(MediaType.APPLICATION_JSON_VALUE);
                            response.setStatus(403);
                            final ObjectMapper mapper = new ObjectMapper();
                            mapper.writeValue(response.getOutputStream(), Resp.forbidden("权限不足" + request.getServletPath()));
                        })
                )
                .addFilterBefore(securityFilter, UsernamePasswordAuthenticationFilter.class)
                .addFilterAfter(gatewayFilter, SecurityFilter.class);

        return http.build();
    }
}
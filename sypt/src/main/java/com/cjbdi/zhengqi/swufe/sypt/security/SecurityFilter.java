package com.cjbdi.zhengqi.swufe.sypt.security;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.json.JSONUtil;
import com.cjbdi.zhengqi.swufe.sypt.entity.User;
import com.cjbdi.zhengqi.swufe.sypt.model.Resp;
import com.cjbdi.zhengqi.swufe.sypt.prop.config.AuthIgnoreUrlsConfig;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.util.matcher.AntPathRequestMatcher;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

@Slf4j
@Component
public class SecurityFilter extends OncePerRequestFilter {

    @Autowired
    private JwtService jwtService;

    @Autowired
    private AuthIgnoreUrlsConfig authIgnoreUrlsConfig;

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {
        Long userId = null;
        try {
            List<String> ignoreUrls = new ArrayList<>(authIgnoreUrlsConfig.getUrls());
            ignoreUrls.add("/swagger-ui/**");
            ignoreUrls.add("/v3/api-docs/**");
            ignoreUrls = CollUtil.distinct(ignoreUrls);
            List<AntPathRequestMatcher> matchers = ignoreUrls.stream()
                    .map(AntPathRequestMatcher::new)
                    .toList();
            if (matchers.stream().anyMatch(matcher -> matcher.matches(request))) {
                filterChain.doFilter(request, response);
                return;
            }

            String jwt = jwtService.extractJwtFromRequest(request);

            if (jwt != null) {
                if (jwtService.validateToken(jwt)) {
                    User user = jwtService.getUserFromToken(jwt);
                    if (user != null) {
                        userId = user.getId();
                        Authentication authentication = createAuthentication(user);
                        SecurityContextHolder.getContext().setAuthentication(authentication);
                    } else {
                        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                        response.getWriter().write(JSONUtil.toJsonStr(Resp.unauthorized()));
                        return;
                    }
                } else {
                    response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                    response.getWriter().write(JSONUtil.toJsonStr(Resp.unauthorized()));
                    return;
                }
            } else {
                response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                response.getWriter().write(JSONUtil.toJsonStr(Resp.unauthorized()));
                return;
            }
        } catch (Exception e) {
            logger.error("处理 JWT 认证时发生错误", e);
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.getWriter().write(JSONUtil.toJsonStr(Resp.unauthorized()));
            return;
        }

        ParameterRequestWrapper wrapper = new ParameterRequestWrapper(request);
        if (userId != null) {
            wrapper.addParameter("user_id", String.valueOf(userId));
        } else {
            wrapper.addParameter("user_id", "");
        }
        filterChain.doFilter(wrapper, response);
    }

    private Authentication createAuthentication(User user) {
        Collection<? extends GrantedAuthority> authorities = extractAuthorities(user);

        // 创建带完整用户细节的认证对象
        return new UsernamePasswordAuthenticationToken(
                user,
                null,
                authorities
        );
    }

    private Collection<? extends GrantedAuthority> extractAuthorities(User user) {
        List<String> roles = user.getRoles();
        if (roles == null) {
            roles = List.of();
        }

        List<GrantedAuthority> authorities = new ArrayList<>(roles.stream()
                .map(role -> new SimpleGrantedAuthority("ROLE_" + role))
                .toList());

        List<String> permissions = user.getPermissions();
        if (CollUtil.isEmpty(permissions)) {
            permissions = new ArrayList<>();
            permissions.add("ROLE_USER");
        }
        if (permissions != null) {
            authorities.addAll(permissions.stream()
                    .map(SimpleGrantedAuthority::new)
                    .toList());
        }


        return authorities;
    }

}
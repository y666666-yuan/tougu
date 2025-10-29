package com.cjbdi.zhengqi.swufe.sypt.security;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONUtil;
import com.cjbdi.zhengqi.swufe.sypt.entity.User;
import com.cjbdi.zhengqi.swufe.sypt.prop.config.AuthConfig;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jws;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.Date;

@Service
public class JwtService {

    private static final String AUTHORIZATION_HEADER = "Authorization";
    private static final String BEARER_PREFIX = "Bearer ";

    @Autowired
    private AuthConfig authConfig;

    @Autowired
    private HttpServletRequest httpServletRequest;

    public User getUserFromToken(String token) {
        if (StrUtil.isBlank(token)) {
            return null;
        }
        try {
            Jws<Claims> claimsJws = Jwts.parser().verifyWith(Keys.hmacShaKeyFor(authConfig.getJwtSecretKey().getBytes()))
                    .build().parseSignedClaims(token);
            String subject = claimsJws.getPayload().getSubject();
            return JSONUtil.toBean(subject, User.class);
        } catch (Exception e) {
            return null;
        }
    }

    public String extractJwtFromRequest(HttpServletRequest request) {
        String header = request.getHeader(AUTHORIZATION_HEADER);
        if (StrUtil.isNotBlank(header) && header.startsWith(BEARER_PREFIX)) {
            return header.substring(BEARER_PREFIX.length());
        } else if (StrUtil.isNotBlank(header)) {
            return header;
        }

        String tokenParam = request.getParameter("token");
        if (StringUtils.hasText(tokenParam)) {
            return tokenParam;
        }

        return null;
    }

    public User getUserFromRequest(HttpServletRequest request) {
        String token = extractJwtFromRequest(request);
        return getUserFromToken(token);
    }

    public User getUser() {
        return getUserFromRequest(httpServletRequest);
    }

    public String generateToken(User user) {
        return Jwts.builder().claims().and()
                .subject(JSONUtil.toJsonStr(user))
                .issuedAt(new Date(System.currentTimeMillis()))
                .expiration(new Date(System.currentTimeMillis() + authConfig.getJwtExpiration()))
                .signWith(Keys.hmacShaKeyFor(authConfig.getJwtSecretKey().getBytes()))
                .issuer("cjbdi")
                .compact();
    }

    public boolean validateToken(String token) throws Exception {
        try {
            if (StrUtil.isBlank(token)) {
                return false;
            }
            String prefix = "Bearer ";
            if (token.startsWith(prefix)) {
                token = token.substring(prefix.length());
            }
            Jws<Claims> claimsJws = Jwts.parser().verifyWith(Keys.hmacShaKeyFor(authConfig.getJwtSecretKey().getBytes()))
                    .build().parseSignedClaims(token);
            Claims claims = claimsJws.getPayload();
            if (claims.getExpiration().before(new Date())) {
                throw new Exception("登录已过期");
            }
            if (claims.getIssuedAt().after(new Date())) {
                throw new Exception("token不在有效期内");
            }
            if (!claims.getIssuer().equals("cjbdi")) {
                throw new Exception("token校验不通过");
            }
            return true;
        } catch (Exception e) {
            throw new Exception("token校验不通过,"+e.getLocalizedMessage());
        }
    }

}

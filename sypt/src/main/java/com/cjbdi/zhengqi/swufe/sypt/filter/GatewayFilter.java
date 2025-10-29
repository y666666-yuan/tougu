package com.cjbdi.zhengqi.swufe.sypt.filter;

import cn.hutool.core.util.StrUtil;
import com.cjbdi.zhengqi.swufe.sypt.prop.config.ForwardingConfig;
import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.util.AntPathMatcher;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Enumeration;

@Slf4j
@Component
public class GatewayFilter implements Filter {

    @Autowired
    private ForwardingConfig forwardingConfig;

    private final AntPathMatcher pathMatcher = new AntPathMatcher();
    // 设置连接超时时间为50秒
    private static final int CONNECTION_TIMEOUT = 50000;
    // 设置读取超时时间为100秒
    private static final int READ_TIMEOUT = 100000;

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {
        HttpServletRequest httpRequest = (HttpServletRequest) request;
        HttpServletResponse httpResponse = (HttpServletResponse) response;

        String requestUri = httpRequest.getRequestURI();
        URL targetUrl = null;

        try {
            targetUrl = getForwardUrl(requestUri);
        } catch (Exception e) {
            log.error("获取转发URL失败: {}", requestUri, e);
            httpResponse.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "网关错误");
            return;
        }

        if (targetUrl != null) {
            forwardRequest(httpRequest, httpResponse, targetUrl);
        } else {
            chain.doFilter(request, response);
        }
    }

    private URL getForwardUrl(String requestUri) throws Exception {
        for (ForwardingConfig.Route route : forwardingConfig.getRoutes()) {
            for (String path : route.getPaths()) {
                if (pathMatcher.match(path, requestUri)) {
                    String targetUrl = route.getUri();
                    Integer stripPrefix = route.getStripPrefix();
                    String targetRequestUri = requestUri;

                    if (stripPrefix > 0 && targetRequestUri.length() > 1) {
                        String tmpPrefix = "";
                        if (targetRequestUri.startsWith("/")) {
                            targetRequestUri = targetRequestUri.substring(1);
                            tmpPrefix = "/";
                        }

                        String[] segments = targetRequestUri.split("/");
                        if (segments.length > stripPrefix) {
                            StringBuilder newUri = new StringBuilder(tmpPrefix);
                            for (int i = stripPrefix; i < segments.length; i++) {
                                newUri.append(segments[i]).append("/");
                            }
                            if (newUri.length() > 1 && newUri.charAt(newUri.length() - 1) == '/') {
                                newUri.deleteCharAt(newUri.length() - 1);
                            }
                            targetRequestUri = newUri.toString();
                        } else {
                            targetRequestUri = tmpPrefix;
                        }
                    }

                    return new URL(targetUrl + targetRequestUri);
                }
            }
        }
        return null;
    }

    private void forwardRequest(HttpServletRequest request,
                                HttpServletResponse response,
                                URL url) {
        HttpURLConnection connection = null;
        try {
            String queryString = request.getQueryString();
            URL fullUrl = StrUtil.isBlank(queryString) ? url : new URL(url.toString() + "?" + queryString);

            connection = (HttpURLConnection) fullUrl.openConnection();
            connection.setRequestMethod(request.getMethod());
            connection.setConnectTimeout(CONNECTION_TIMEOUT);
            connection.setReadTimeout(READ_TIMEOUT);
            connection.setInstanceFollowRedirects(false);

            // 复制请求头
            copyRequestHeaders(request, connection);

            // 处理请求体
            if (isBodyRequestMethod(request.getMethod())) {
                connection.setDoOutput(true);
                copyRequestBody(request, connection);
            }

            // 获取响应
            int responseCode = connection.getResponseCode();
            response.setStatus(responseCode);

            // 复制响应头
            copyResponseHeaders(connection, response);

            // 复制响应体
            copyResponseBody(connection, response, responseCode);

        } catch (Exception e) {
            log.error("转发请求失败: {}", url, e);
            try {
                response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "转发请求失败");
            } catch (IOException ioException) {
                log.error("设置错误响应失败", ioException);
            }
        } finally {
            // 确保释放连接资源
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private void copyRequestHeaders(HttpServletRequest request, HttpURLConnection connection) {
        Enumeration<String> headerNames = request.getHeaderNames();
        while (headerNames.hasMoreElements()) {
            String headerName = headerNames.nextElement();
            Enumeration<String> headerValues = request.getHeaders(headerName);
            while (headerValues.hasMoreElements()) {
                String headerValue = headerValues.nextElement();
                connection.addRequestProperty(headerName, headerValue);
            }
        }
    }

    private boolean isBodyRequestMethod(String method) {
        return "POST".equalsIgnoreCase(method) ||
                "PUT".equalsIgnoreCase(method) ||
                "PATCH".equalsIgnoreCase(method);
    }

    private void copyRequestBody(HttpServletRequest request, HttpURLConnection connection) throws IOException {
        try (InputStream is = request.getInputStream();
             OutputStream os = connection.getOutputStream()) {

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
        }
    }

    private void copyResponseHeaders(HttpURLConnection connection, HttpServletResponse response) {
        for (int i = 0; ; i++) {
            String headerName = connection.getHeaderFieldKey(i);
            String headerValue = connection.getHeaderField(i);

            if (headerName == null && headerValue == null) {
                break;
            }

            if (headerName != null) {
                response.addHeader(headerName, headerValue);
            }
        }
    }

    private void copyResponseBody(HttpURLConnection connection, HttpServletResponse response, int responseCode) throws IOException {
        try (InputStream is = (responseCode >= 400) ? connection.getErrorStream() : connection.getInputStream();
             OutputStream os = response.getOutputStream()) {

            if (is != null) {
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
            }
        }
    }
}
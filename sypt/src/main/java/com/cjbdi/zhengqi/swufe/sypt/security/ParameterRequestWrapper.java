package com.cjbdi.zhengqi.swufe.sypt.security;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletRequestWrapper;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class ParameterRequestWrapper extends HttpServletRequestWrapper {
    private final Map<String, String> params;
    public ParameterRequestWrapper(HttpServletRequest request) {
        super(request);
        this.params = new HashMap<>();
    }

    public void addParameter(String name, String value) {
        params.put(name, value);
    }

    @Override
    public String getParameter(String name) {
        if (params.containsKey(name)) {
            return params.get(name);
        }
        return super.getParameter(name);
    }

    @Override
    public Map<String, String[]> getParameterMap() {
        Map<String, String[]> superParameterMap = super.getParameterMap();
        Map<String, String[]> parameterMap;
        if (CollUtil.isEmpty(superParameterMap)) {
            parameterMap = new HashMap<>();
        } else {
            parameterMap = new HashMap<>(superParameterMap);
        }
        if (CollUtil.isNotEmpty(params)) {
            params.forEach((key, value) -> parameterMap.put(key, new String[]{value}));
        }
        return parameterMap;
    }

    @Override
    public Enumeration<String> getParameterNames() {
        Enumeration<String> superParameterNames = super.getParameterNames();
        Set<String> parameterNames = new HashSet<>();
        while (superParameterNames.hasMoreElements()) {
            parameterNames.add(superParameterNames.nextElement());
        }
        parameterNames.addAll(params.keySet());
        return Collections.enumeration(parameterNames);
    }

    @Override
    public String[] getParameterValues(String name) {
        if (params.containsKey(name)) {
            return new String[]{params.get(name)};
        }
        return super.getParameterValues(name);
    }

    @Override
    public String getQueryString() {
        String originalQueryString = super.getQueryString();

        if (params.isEmpty()) {
            return originalQueryString;
        }

        StringBuilder customQueryString = new StringBuilder();
        for (Map.Entry<String, String> entry : params.entrySet()) {
            if (!customQueryString.isEmpty()) {
                customQueryString.append("&");
            }
            try {
                customQueryString.append(URLEncoder.encode(entry.getKey(), StandardCharsets.UTF_8))
                        .append("=")
                        .append(URLEncoder.encode(entry.getValue(), StandardCharsets.UTF_8));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        if (StrUtil.isNotBlank(originalQueryString)) {
            return originalQueryString + "&" + customQueryString;
        }

        return customQueryString.toString();
    }
}

package com.cjbdi.zhengqi.swufe.sypt.entity;

import lombok.Data;

import java.util.List;

@Data
public class User {
    private Long id;
    private String username;
    private String nickname;
    private String email;
    private String token;
    private List<String> roles;
    private List<String> permissions;
}
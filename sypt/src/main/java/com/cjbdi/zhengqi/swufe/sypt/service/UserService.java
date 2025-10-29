package com.cjbdi.zhengqi.swufe.sypt.service;

import com.cjbdi.zhengqi.swufe.sypt.dto.LoginRequest;
import com.cjbdi.zhengqi.swufe.sypt.entity.User;

import java.util.Optional;

public interface UserService {

    Optional<User> authenticate(LoginRequest request) throws Exception;

}

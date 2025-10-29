package com.cjbdi.zhengqi.swufe.sypt.gen;

import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.util.Base64;

public class GenKey {

    public static void main(String[] args) throws Exception {
        KeyGenerator  keyGenerator = KeyGenerator.getInstance("HmacSHA256");
        SecretKey  secretKey = keyGenerator.generateKey();
        String secretKeyStr = Base64.getEncoder().encodeToString(secretKey.getEncoded());
        System.out.println(secretKeyStr);
        System.out.println(secretKeyStr.length());
    }

}

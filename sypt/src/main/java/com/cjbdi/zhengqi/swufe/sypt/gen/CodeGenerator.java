package com.cjbdi.zhengqi.swufe.sypt.gen;

import com.baomidou.mybatisplus.generator.FastAutoGenerator;
import com.baomidou.mybatisplus.generator.engine.FreemarkerTemplateEngine;

import java.nio.file.Paths;

public class CodeGenerator {

//    public static void main(String[] args) {
//        String url = "jdbc:mysql://192.168.15.5:3306/xcdpzs?useUnicode=true&zeroDateTimeBehavior=convertToNull&autoReconnect=true&characterEncoding=utf-8";
//        String username = "root";
//        String password = "cjbdi";
//        FastAutoGenerator.create(url, username, password)
//                .globalConfig(builder -> builder
//                        .author("liangboning")
//                        .outputDir(Paths.get(System.getProperty("user.dir")) + "/src/main/java")
//                        .commentDate("yyyy-MM-dd")
//                )
//                .packageConfig(builder -> builder
//                        .parent("com.cjbdi.zhengqi.dp")
//                        .entity("bond.entity")
//                        .mapper("bond.mapper")
//                        .service("bond.service")
//                        .serviceImpl("bond.service.impl")
//                        .xml("bond.mapper.xml")
//                )
//                .strategyConfig(builder -> builder
//                        .addInclude("issuer_rating_info")
//                        .entityBuilder()
//                        .enableLombok()
//                )
//                .templateEngine(new FreemarkerTemplateEngine())
//                .execute();
//    }

    public static void main(String[] args) {
        String url = "jdbc:mysql://192.168.15.5:3306/xcsypt?useUnicode=true&zeroDateTimeBehavior=convertToNull&autoReconnect=true&characterEncoding=utf-8";
        String username = "root";
        String password = "cjbdi";
        String subFolder = "exp";
        FastAutoGenerator.create(url, username, password)
                .globalConfig(builder -> builder
                        .author("liangboning")
                        .outputDir(Paths.get(System.getProperty("user.dir")) + "/src/main/java")
                        .commentDate("yyyy-MM-dd")
                )
                .packageConfig(builder -> builder
                        .parent("com.cjbdi.zhengqi.swufe.sypt")
                        .entity("entity." + subFolder)
                        .mapper("mapper."+subFolder)
                        .service("service."+subFolder)
                        .serviceImpl("service."+subFolder+".impl")
                        .xml("mapper."+subFolder + ".xml")
                )
                .strategyConfig(builder -> builder
                        .addInclude("exp_result")
                        .entityBuilder()
                        .enableLombok()
                )
                .templateEngine(new FreemarkerTemplateEngine())
                .execute();
    }

}

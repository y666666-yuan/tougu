package com.cjbdi.zhengqi.swufe.sypt.model;

import lombok.Data;

import java.io.Serializable;
import java.util.List;

@Data
public class PagedUserFullInfo implements Serializable {

    private List<UserFullInfo> users;

    private Long total;

    private Long current;

    private Integer pageSize;

    private Integer pages;

    private Long size;

}

package com.cjbdi.zhengqi.swufe.sypt.model;

import lombok.Data;

import java.io.Serializable;

@Data
public class PagedUsersRequest implements Serializable {

    private Integer pageNum;

    private Integer pageSize;
}

package com.cjbdi.zhengqi.swufe.sypt.controller;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.OrderItem;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.cjbdi.zhengqi.swufe.sypt.entity.User;
import com.cjbdi.zhengqi.swufe.sypt.entity.exp.ExpResult;
import com.cjbdi.zhengqi.swufe.sypt.model.Resp;
import com.cjbdi.zhengqi.swufe.sypt.security.JwtService;
import com.cjbdi.zhengqi.swufe.sypt.service.exp.IExpResultService;
import io.swagger.v3.oas.annotations.Hidden;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@Tag(name = "实验数据接口")
@RestController
@RequestMapping("/api/exp-data")
public class ExperimentalDataController {

    @Autowired
    private IExpResultService expResultService;

    @Autowired
    private JwtService jwtService;

    /**
     * @param expResult
     * {
     *   "expId": "test1",
     *   "expName": "testName",
     *   "algorithmType" "xxx",
     *   "algorithmName": "testAlgo"
     * }
     *
     * @return Resp<ExpResult>
     */
    @Operation(
            summary = "保存实验结果",
            description = """
                    # 保存实验结果参数说明
                    | 参数名 | 参数说明 | 示例值 |
                    | --- | --- | --- |
                    | expId | 实验id | "1" |
                    | expName | 实验的名称 | "实验1" |
                    | algorithmType | 算法类型 | "ehgnn" |
                    | algorithmName | 算法名称 | "异构图神经网络" |
                    """,
            requestBody = @io.swagger.v3.oas.annotations.parameters.RequestBody(
                    description = "实验结果",
                    required = true,
                    content = @io.swagger.v3.oas.annotations.media.Content(
                            mediaType = "application/json",
                            schema = @io.swagger.v3.oas.annotations.media.Schema(
                                    implementation = ExpResult.class
                            ),
                            examples = {
                                    @io.swagger.v3.oas.annotations.media.ExampleObject(
                                            name = "参数示例",
                                            value = """
                                                    {
                                                      "expId": "1",
                                                      "expName": "实验1",
                                                      "algorithmType": "ehgnn",
                                                      "algorithmName": "异构图神经网络"
                                                    }
                                                    """
                                    )
                            }
                    )
            )
    )
    @PostMapping("/save-exp-result")
    public Resp<ExpResult> saveExpResult(@RequestBody ExpResult expResult) {
        Long userId = getUserId();
        if (userId == null) {
            return Resp.unauthorized();
        }
        expResult.setCreatedBy(userId);
        expResult.setCreatedAt(null);
        LambdaQueryWrapper<ExpResult> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(ExpResult::getExpId, expResult.getExpId());
        if (expResultService.count(queryWrapper) > 0) {
            return Resp.error("实验结果(expId=" + expResult.getExpId() + ")已存在");
        }
        if (expResultService.save(expResult)) {
            LambdaQueryWrapper<ExpResult> savedQueryWrapper = new LambdaQueryWrapper<>();
            savedQueryWrapper.eq(ExpResult::getExpId, expResult.getExpId());
            savedQueryWrapper.eq(ExpResult::getCreatedBy, userId);
            ExpResult savedExpResult = expResultService.getOne(savedQueryWrapper);
            if (savedExpResult != null) {
                return Resp.success(savedExpResult);
            }
            return Resp.error("保存实验结果失败(1)");
        }
        return Resp.error("保存实验结果失败(2)");
    }

    /**
     *
     * @param page
     * {
     *   "total": 0,
     *   "size": 0,
     *   "current": 0,
     *   "orders": [
     *     {
     *       "column": "string",
     *       "asc": true
     *     }
     *   ]
     * }
     * @return Resp<Page<ExpResult>>
     */
    @Operation(
            summary = "分页查询实验结果",
            description = """
                    # 实验结果字段说明
                    | 字段 | 说明 | 示例值 |
                    | --- | --- | --- |
                    | expId | 实验id | "1" |
                    | expName | 实验的名称 | "实验1" |
                    | algorithmName | 算法名称 | "异构图神经网络" |
                    | createdBy | 创建人的Id | 10001 |
                    | createdAt | 创建时间 | "2025-07-08 09:52:13" |
                    """,
            requestBody = @io.swagger.v3.oas.annotations.parameters.RequestBody(
                    description = "分页参数",
                    required = true,
                    content = @io.swagger.v3.oas.annotations.media.Content(
                            mediaType = "application/json",
                            schema = @io.swagger.v3.oas.annotations.media.Schema(
                                    implementation = Page.class
                            ),
                            examples = {
                                    @io.swagger.v3.oas.annotations.media.ExampleObject(
                                            name = "参数示例1",
                                            value = """
                                                    {
                                                      "size": 10,
                                                      "current": 0
                                                    }
                                                    """
                                    ),
                                    @io.swagger.v3.oas.annotations.media.ExampleObject(
                                            name = "参数示例2",
                                            value = """
                                                    {
                                                      "size": 10,
                                                      "current": 0,
                                                      "orders": [
                                                        {
                                                          "column": "createdAt",
                                                          "asc": false
                                                        }
                                                      ]
                                                    }
                                                    """
                                    )
                            }
                    )
            )
    )
    @PostMapping("/page-exp-result")
    public Resp<Page<ExpResult>> pageExpResult(@RequestBody Page<ExpResult> page) {
        Long userId = getUserId();
        if (userId == null) {
            return Resp.unauthorized();
        }
        if (page.getSize() <= 0) {
            page.setSize(10);
        }
        if (CollUtil.isNotEmpty(page.orders())) {
            page.orders().forEach(orderItem -> {
                if (StrUtil.isNotBlank(orderItem.getColumn())) {
                    orderItem.setColumn(StrUtil.toUnderlineCase(orderItem.getColumn()));
                }
            });
        } else {
            page.addOrder(OrderItem.desc("created_at"));
        }
        LambdaQueryWrapper<ExpResult> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(ExpResult::getCreatedBy, userId);
        return Resp.success(expResultService.page(page, queryWrapper));
    }

    @Hidden
    @Operation(summary = "更新实验结果")
    @PostMapping("/update-exp-result")
    public Resp<ExpResult> updateExpResult(@RequestBody ExpResult expResult) {
        Long userId = getUserId();
        if (userId == null) {
            return Resp.unauthorized();
        }
        expResult.setCreatedBy(userId);
        LambdaQueryWrapper<ExpResult> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(ExpResult::getExpId, expResult.getExpId());
        queryWrapper.eq(ExpResult::getCreatedBy, userId);
        if (expResultService.list(queryWrapper).isEmpty()) {
            return Resp.error("实验结果 " + expResult.getExpId() + " 不存在");
        }
        if (expResultService.updateById(expResult)) {
            return Resp.success(expResultService.getOne(queryWrapper));
        }
        return Resp.error("更新实验结果失败");
    }

    @Hidden
    @Operation(summary = "删除实验结果")
    @GetMapping("/delete-exp-result")
    public Resp<String> deleteExpResult(@RequestParam String expId) {
        Long userId = getUserId();
        if (userId == null) {
            return Resp.unauthorized();
        }
        LambdaQueryWrapper<ExpResult> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(ExpResult::getExpId, expId);
        queryWrapper.eq(ExpResult::getCreatedBy, userId);
        if (expResultService.list(queryWrapper).isEmpty()) {
            return Resp.error("实验结果 " + expId + " 不存在");
        }
        return expResultService.remove(queryWrapper) ? Resp.successMsg("删除实验结果成功") : Resp.error("删除实验结果失败");
    }

    @Hidden
    @Operation(summary = "清空实验结果")
    @GetMapping("/clear-exp-result")
    public Resp<String> clearExpResult() {
        Long userId = getUserId();
        if (userId == null) {
            return Resp.unauthorized();
        }
        LambdaQueryWrapper<ExpResult> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(ExpResult::getCreatedBy, userId);
        long beforeDeleteCount = expResultService.count(queryWrapper);
        boolean deleteResult = expResultService.remove(queryWrapper);
        long afterDeleteCount = expResultService.count(queryWrapper);
        if (afterDeleteCount > 0 || !deleteResult) {
            return Resp.error("清空实验结果失败");
        }
        return Resp.successMsg(beforeDeleteCount + "条实验结果清空成功");
    }

    private Long getUserId() {
        User user = jwtService.getUser();
        return user == null ? null : user.getId();
    }

}

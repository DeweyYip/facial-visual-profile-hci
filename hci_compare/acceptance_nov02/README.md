# 11/2：20 个结构化语言验收输入

## 运行

需要已更新的 v5 页面和 v5.1 补丁。此包只增加验收工具，不改页面、生成规则或权重。
先在运行页面的终端按 Control+C，避免批量测试和页面同时加载语言模型。
在 FacialVisualProfileHCI 根目录依次执行：

```bash
unzip -o ~/Downloads/fvp_hci_nov02_acceptance.zip -d .
.venv/bin/python hci_compare/check_acceptance.py
.venv/bin/python hci_compare/run_acceptance.py --local
```

第一条检查使用受控服务，不加载 Qwen。第二条正式运行 20 个固定输入，
其中 15 个调用真实 Qwen3-1.7B，5 个应由 gate 跳过，不调用语言模型。
复用 llm_trial/local 中已下载的模型；不需要网络、不重新下载、不需要视觉模型权重。
预计需要数分钟，生成较长的案例可能更慢。每个案例完成后都会保存进度。
结束或中断时终止本工具启动的语言服务。

如只想先跑模板规则检查：

```bash
.venv/bin/python hci_compare/run_acceptance.py
```

这条命令的 local_status 为 not_run，不能当作 Qwen 验收结果。

## 结果及判读

结果保存在 hci_compare/acceptance_nov02/results/<UTC时间>/：

- report.json：冻结文件哈希、每个案例的预期/实际属性、显示文字、来源、状态和回退原因。
- review.md：逐条对照允许属性和显示文字，记录遗漏、无依据内容与可读性。

可以打开结果目录：

```bash
open hci_compare/acceptance_nov02/results/
```

完成后上传最近一次运行的 report.json；不是只看脚本是否运行结束。
system_contract_pass 表示所有输入的模板、拒绝生成或回退行为符合约定。
accepted_local_generations 单独统计真实生成并通过检查的数量。
template_fallbacks 不计入生成成功。发生回退而系统正常恢复时退出码为 2，
并保留详细报告；系统约定不通过时退出码为 1。请查看原因，不要修改测试期望来消除失败。

本工具检查当前有限表达规则下的证据忠实性，不评价照片里的人是否真的具有该属性。
人工可读性记录初始为 pending，不会自动填成人工审核通过。

## 输入范围及冻结方式

这 20 个输入是 v5.1 规则确定后新建的合成结构化验收集，
不是此前已经预留的 held-out split，也不是 20 张真实照片。
旧对话所说的“20 个保留输入”此前没有实际建立，因此不能追认它们早已保留。
本次正确记录为 post-freeze acceptance：先冻结实现，再编写新的输入和独立期望。
它们与原 60 个开发 payload 不相同，但不代表统计上独立采样或独立研究者验证。

cases.json 的预期属性列表独立填写，不由 describe() 自动生成。
freeze.json 固定 cases.json 及当前 app.py、language.py、local_wording.py、
attribute_names.json 和 language_input.schema.json 的 SHA256。
运行前核对这些文件，有差异就停止。修改规则后需要建立新的协议版本，不能继续沿用本报告的冻结声明。
案例中的图像 SHA256 是合成占位标识，并不对应真实图片像素。

| 输入 | 来源 | 检查内容 |
| --- | --- | --- |
| 01–07 | R18/R50/M3 轮换 | 表情、配饰、眉毛、面部结构、胡须和头发组合 |
| 08–09 | R50/M3 | 描述门槛以下与恰好达到 0.05 margin |
| 10–12 | R18/R50/M3 | 发色冲突、发质冲突和秃发抑制 |
| 13 | R18 | 最多六项属性 |
| 14 | R50 | 其他模型有正预测时仍只使用当前来源 |
| 15 | M3 | 有效检测、强模糊标记与提示保留 |
| 16 | R18 | 正预测全部靠近阈值，不生成 |
| 17–19 | R50/M3/R18 | 无脸、多脸、检测错误时不生成 |
| 20 | M3 | 来源模型不可用时不生成 |

不重跑人脸检测、图像预处理或 R18/R50/M3 推理；这些模块此前已有运行检查。
完成本次验收也不能宣称完成真实照片准确率、用户研究或任意自由文本的语义评估。

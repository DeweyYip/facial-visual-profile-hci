# v5.1：本地生成字段约束

用户真实页面返回 `invalid_sentence_fields`，说明生成已返回，但句子对象缺少必需字段或包含未约定字段。
页面已正确显示模板回退；不能把这次回退当作本地生成成功。

此前生成端只要求合法 JSON；本补丁同时给 llama.cpp 发送 JSON Schema：
外层必须有 sentences，每条必须有 text 和 evidence_id，不允许额外字段，
数量等于当前合资格属性数。text 与 evidence_id 分别限制为当前允许值。
生成后继续检查每个短语与其证据编号的对应关系、完整覆盖和重复引用。
不符合约定仍然回退模板。

安装：停止应用，在 FacialVisualProfileHCI 根目录执行：

```bash
unzip -o ~/Downloads/fvp_hci_v5_1_patch.zip -d .
.venv/bin/python hci_compare/check_local_wording.py
.venv/bin/python hci_compare/app.py
```

然后重新上传照片、Compare models，再尝试本地生成。
不改动视觉模型、阈值、已有权重、下载资源或应用依赖。
本补丁通过 9 项受控测试，包括实际本机假 HTTP 服务接收 schema、回退与程序清理。
这些测试不加载 Qwen；真实生成是否通过，以用户 Mac 更新后的结果为准。

# Facial Visual Profile · UI v6.2

浅米色界面，按用户提供的布局草图重排。左侧上传与原图预览，右侧为扰动图与关键点图。
下方是扰动选择、强度滑杆、比较按钮、三模型短句描述和属性浏览器。

## 安装

在运行旧页面的终端按 Control+C。在 FacialVisualProfileHCI 根目录执行：

```bash
mkdir -p backups
cp -R hci_compare backups/hci_compare_before_ui_v6_2
unzip -o ~/Downloads/fvp_hci_ui_v6_2_patch.zip -d .
.venv/bin/python hci_compare/check_language.py
.venv/bin/python hci_compare/check_errors.py
.venv/bin/python hci_compare/check_ui.py
.venv/bin/python hci_compare/app.py
```

每一步通过后再继续。复用现有 .venv 和 weights，无需下载模型或安装新的依赖。
页面仍只在本机启动。此更新不主动创建公共分享链接。

## 页面使用

1. 在左侧上传一张静态图片；左侧按原始宽高比显示预览，不附加黑色留边。
2. 选择 Image variation 与强度。0 表示原始输入。
3. 点击 Compare models；右侧显示最终模型输入和关键点。
4. 阅读三个来源各自的短句描述。
5. 在 Explore the attributes 中按 Expression、Hair、Face、Facial hair、Accessories 筛选。
   Predicted 只显示至少一个模型预测为正的属性；Differences 只显示可用模型判断分歧的属性；
   All attributes 可以浏览完整 24 项。
6. 展开 Look closer at one attribute，选择属性查看三个模型的分数与阈值刻度。
7. 阈值、描述证据和组件排查信息收在最后两个折叠区。

要更换图片，点击 Reload image，返回上传区并选择新图片。
上传成功后隐藏文件名、文件大小和下载栏。按钮会同时清除旧图片及其全部结果。
换图或修改扰动会清除旧结果，重新点击 Compare models 即可。
比较和重新加载期间，输入及筛选控件暂时锁定。

## 本次改动的边界

v6.2：顶部移除 Portrait workspace，FVP 居中。详情说明分成三个简短区块；表格使用易读属性名称、简短表头及三位小数。保留 Evidence ID 和完整结构化证据，原始数值与分类判断不变。

v6.1：补齐深色系统主题下表格的浅色背景与文字样式；新增 Reload image。
仅左侧展示预览去除 letterbox，右侧两张结果图仍显示真实 224×224 画布，
保留模型实际接收的黑色填充与遮挡。未修改输入像素、扰动或检测流程。

- 移除 Status 板块、本地 Qwen 按钮、structured language input 和 local generation checks 的可见入口。
- 页面不再导入或启动 Qwen；原有 llm_trial、权重和报告不删除、不改写。
- 用已有过滤后的属性短语组织自然短句，来源、门槛规则和最多六项描述的限制不变。
- 完整分类预测仍使用各模型原有阈值；浏览器中的 Predicted 与描述过滤不同，后者会省略接近阈值或冲突项目。
- 原始上传仍使用 gr.File，保留文件字节和原始文件名，不用 gr.Image 输入重新编码。
- backend.py、language.py、vendor、execution_binding、检测器来源与 requirements.txt 保持原文件字节。
- 无自动裁剪、对齐或 EXIF 旋转；五种扰动和最终画布上的 MTCNN 检测规则不变。

styles.css 可单独调整外观；ui_presenter.py 仅负责展示；app.py 负责布局和事件。
页面使用系统字体，不引入远程字体、图片或 JavaScript 资源。

11/2 的验收报告属于当时冻结的 v5.1 快照。此次 app.py 外观更新是新版本，
不改写旧 freeze.json，也不把旧报告冒充为新页面的验收。
新界面由 check_language.py、check_errors.py、check_ui.py 和浏览器交互检查验证。

## 验证范围

60 个合成语言开发案例、12 项受控错误/页面恢复检查、5 项展示检查通过。
浏览器检查使用受控模型输出，覆盖桌面和 390px 手机宽度、上传、比较、属性过滤、
调整强度和更换图片后的清空行为。这不是新一轮真实权重准确率测试。
更新后，实际照片和真实模型的最终页面表现以用户 Mac 上运行为准。

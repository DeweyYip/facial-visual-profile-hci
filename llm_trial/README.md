# 10/31：Intel Mac 本地語言模型試用

本工具與 `hci_compare/app.py` 分開執行，不修改現有介面、模型權重或 Python 套件。
這一步比較 Qwen3-0.6B 和 Qwen3-1.7B；11/1 才依結果決定是否接入。
Reload models 收入進階／疑難排解區的介面調整留待下一次頁面更新。

## 執行

將壓縮包解壓縮到 **FacialVisualProfileHCI 根目錄**，得到並排的 `hci_compare/`、
`weights/`、`llm_trial/`。需要已跑通的 10/30 版本及 Python 3.10。
先在現有 Gradio 終端按 Control+C，讓測速期間的背景負載較一致。

在專案根目錄執行：

```bash
.venv/bin/python llm_trial/check_trial.py
.venv/bin/python llm_trial/setup.py
.venv/bin/python llm_trial/run_trial.py
```

setup 使用官方 llama.cpp 的固定 b11712 Intel macOS CPU 二進位包，
固定官方發布頁所列的 SHA256，下載後核對，並保存來源與本地檔案雜湊。
模型使用 Qwen 官方的固定 revision **Q8_0** 檔案，核對固定 SHA256：

| 模型 | 模型下載大小 |
| --- | --- |
| Qwen3-0.6B | 約 639 MB |
| Qwen3-1.7B | 約 1,834 MB |

合計約 2.47 GB，另有執行器及解壓空間。這不是所需 RAM 的估計。
初次 setup 需要連線 GitHub 和 Hugging Face；試用本身只連線本機回環位址。
不需要 Google Drive 中的 R18/R50/M3 權重，也不需要 API key。
不使用 MLX、GPU 或 llama-cpp-python，不更改已固定的 torch 環境。
下載失敗可稍後重跑 setup；已下載並驗證的模型會重用。
中斷的下載會重新下載該檔。下載/執行失敗會保留錯誤，不會宣稱模型測試通過。

想先試小模型可用：

```bash
.venv/bin/python llm_trial/setup.py --models 0.6b
.venv/bin/python llm_trial/run_trial.py --models 0.6b
```

之後照預設命令即可補齊 1.7B。兩個模型逐一載入，每個完成後終止其程序。
預設 4 CPU threads、2048 context、384 輸出 token 上限、temperature=0、seed=31，
關閉 thinking，使用 JSON 模式，丟棄一次額外 warm-up，請求不重用 prompt cache。
不是冷磁碟或跨機器基準測試，也不以單次測量宣稱統計優勢。

## 結果

每次結果放在 `llm_trial/results/<UTC時間>/`：

- `summary.csv`：啟動至健康檢查就緒秒數、生成請求數、格式/證據檢查數、失敗數、跳過數、回應時間中位數。
- `0.6b.json` / `1.7b.json`：原始輸出、完成原因、每次耗時、服務提供的 token/速度資料、模板參照、來源雜湊與錯誤。
- `review.csv`：人工對照 allowed facts 與原文，填 faithful_yes_no、natural_1_to_5、notes。
- `environment.json`：OS、架構、Python、執行參數與執行器來源。
- `*_server.log`：載入失敗、CPU 指令集或動態函式庫錯誤時供排查。

RSS 每 250ms 讀取一次，是**語言模型服務程序的觀測峰值**；不是完整 HCI 的 RAM、
絕對峰值或模型大小。初始化時間可能包含執行器內建 warm-up。
回應耗時不含啟動與丟棄的 warm-up，也不含被 gate 跳過的案例。
HTTP 請求失敗仍記錄，完整耗時只統計成功取得回應的請求。

## 檢查範圍

預設取既有 60 個**合成結構化開發案例**中的 12 個，涵蓋單屬性、門檻、
無臉、模型不可用、頭髮過濾、模型分歧及影像擾動標記。案例不是照片。
不符合描述 gate 或無合資格屬性時不呼叫 LLM，沿用模板資訊。
不混合三個視覺模型的證據。語言模型不直接看影像。

自動檢查：合法 JSON、限定欄位、每個已允許 evidence_id 恰好出現一次、無跨模型 id、
生成完整結束。這些檢查**不能證明句子的語意忠實**，例如用正確 id 寫錯屬性仍可能通過。
因此 `structure_pass_manual_review_pending` 只表示結構與引用通過，人工語意審核仍待完成。
任何原始生成內容都不自動接入目前頁面。

這不等於 11/2 的獨立 20 個保留輸入評估，也不構成模型準確率測試。
如需跑全部開發案例可加 `--all-dev`，仍然屬於開發試用。
如需測目前頁面匯出的結構化 language JSON：

```bash
.venv/bin/python llm_trial/run_trial.py --payload /absolute/path/input.json --source m3
```

這只代表該次匯出輸入；不會重跑人臉偵測或視覺模型，也不自動變成保留測試。

## 本包驗證及限制

`check_trial.py` 使用受控的假 HTTP 服務，驗證 gate 不呼叫生成、錯誤引用拒絕、
格式錯誤記錄、報告輸出、程序清理及下載校驗；它不載入 Qwen。
本包尚未在你的 Intel Mac 完成真實 Qwen 載入與生成；執行後以結果檔為準。
已下載官方 Intel 執行器壓縮包核對 SHA256、實際解壓縮，並確認執行器架構為 x86_64；
這些靜態檢查不等於在 Mac 上執行通過。
若 Mac 回報無法啟動，貼出該模型的 server.log 與終端錯誤再處理。

參考：[Qwen llama.cpp 文件](https://qwen.readthedocs.io/en/v3.0/run_locally/llama.cpp.html)、
[llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/b11712/tools/server)、
[Qwen3-0.6B GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)、
[Qwen3-1.7B GGUF](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF)。

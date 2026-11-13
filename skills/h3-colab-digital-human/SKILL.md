---
name: h3-colab-digital-human
description: 用一张人物照＋一句中文需求，自动在用户已开好的 Colab（或本机）ComfyUI 上生成 MiniMax H3 会说话的数字人视频。当用户说「做数字人」「让照片里的人说话」「H3 生成视频」「帮我用 Colab 生成」并提供 trycloudflare 网址或 ComfyUI 网址时使用。需要能运行本机 PowerShell（或 Python）的桌面版 AI。
---

# H3 数字人：一张照片＋一句话 → 会说话的视频

## 用户要先做的事（AI 做不到，请提醒）
1. 从 <https://github.com/dseditor/minimax-h3-colab> 打开 `H3_Basic_ComfyUI.ipynb`（或 `H3_Digital_Human.ipynb`），运行阶段选 **L4**。
2. 依序运行 **①、②、③-b、④**，把 ③-b 印出来的 `https://….trycloudflare.com` 网址交给你。
   （③ 的 Colab 内置代理只有用户本人的浏览器打得开，AI 用不了，一定要用 ③-b。）
3. 给你：**人物照的文件路径**、**一句中文需求**（说什么、什么语气、几秒；没给秒数就用 10 秒）。

网址等于 GPU 钥匙：只在这次对话使用，不要写进任何公开的文件。

## 你要做的事

### 1. 先测连接
```powershell
powershell -ExecutionPolicy Bypass -File "<skill>/scripts/h3_colab.ps1" -Url <网址> -Check
```
`[FAIL]` 就停下来，请用户确认 Colab 的 ② 与 ③-b 还在运行（运行阶段没有中断）、网址没贴错。
（没有 PowerShell 的环境改用 `python <skill>/scripts/h3_colab.py --url <网址> --check`，参数相同。）

### 2. 看照片，写提示词
- **打开照片亲眼看过再写。** 这个模型参考力偏弱：提示词描述的人物、服装、场景只要跟照片对不上，模型就不看参考图，直接照文本生成。
- 用 Ref2VA 格式，**完整规则在 MiniMax 官方指南 `ref-en.txt`，写之前先读**。照片就是 `<Picture 1>`：
  - 已安装官方 `h3-prompt-writing` skill 的话，直接读它的 `references/ref-en.txt`。
  - 没有的话下载（固定版本）存到本 skill 的 `references/`：
    `https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/d21241f0a4b3acbb34c97dae47fa417b7065e438/.claude/skills/h3-prompt-writing/references/ref-en.txt`
  - `<Subject 1>` ＝照片里的人（发型、发饰、服装、配件照实写），`<Subject 2>` ＝照片里的场景。
  - 照片里没有的东西不要给 Subject 标签，也尽量不要加道具：道具写越多，模型越容易变出奇怪的东西。
- 数字人的固定写法：
  - 单一镜头、中近景、构图贴近照片，镜头几乎不动（顶多很慢地推近）。
  - 台词放进 `<d>[Chinese] …</d>`，**保留用户给的中文原文**，说话者标 `(S1)`。
  - **台词长度：每秒约 4～5 个中文本**（10 秒约 40～45 字）。太长句尾会糊，太短可以补一个自然的手势或表情。
  - 写明「她是画面中唯一的人、唯一的声音；画面干净，没有任何文本或字幕」。
  - 全文 300～500 个英文本。
- 写好存成 UTF-8 的 `prompt.txt`，先把台词那一句贴给用户确认，对了再送出。

### 3. 生成并下载
```powershell
powershell -ExecutionPolicy Bypass -File "<skill>/scripts/h3_colab.ps1" -Url <网址> -Image <照片> -PromptFile prompt.txt -Seconds 10 -Out <输出.mp4>
```
- 秒数 5～15，缺省分辨率 0.4MP，seed 缺省随机（`-Seed` 可固定）。
- L4 生成 10 秒大约 4 分钟；脚本每 10 秒回报一次进度，跑完印出 `[DONE]` 与文件位置。
- 失败时脚本会印出 ComfyUI 的错误内容，照实转告用户，不要自行猜测原因。

### 4. 交付
告诉用户视频路径，并请他看三件事：脸和衣服像不像照片、嘴型对不对、有没有多出字幕或第二个人。
不像照片时，先回头对照片检查提示词的人物描述，不要急着换 seed 重抽。

## 文件
- `assets/workflow_api.json`：ComfyUI API 格式工作流（PRISM 融合 int8 模型、6 步、全原生节点），脚本只会改提示词、图片、秒数、分辨率、seed 五个值。
- `references/`：放 MiniMax 官方提示词指南（不随本 repo 散布，来源与版本见 `references/H3_NOTICE.md`）。
- `scripts/h3_colab.ps1`：Windows PowerShell 5.1／7 都能跑，上传和下载用 Windows 内置的 `curl.exe`。
- `scripts/h3_colab.py`：同功能的 Python 版（只用标准函数库）。

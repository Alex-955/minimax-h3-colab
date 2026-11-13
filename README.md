# MiniMax H3 × ComfyUI on Google Colab

在 Google Colab 的 L4 或 A100 上运行 **MiniMax H3 视频**和 **MiniMax-H3-Image 生图**。Toybox 网页可生成 4K 图片，并把结果发送到视频首帧。

| 笔记本 | 用途 | 打开 |
|---|---|---|
| **H3_Basic_ComfyUI** | 基础学习：装好 ComfyUI＋H3 模型组，**不带工作流**，自己拉节点 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dseditor/minimax-h3-colab/blob/main/H3_Basic_ComfyUI.ipynb) |
| **H3_Digital_Human** | 数字人：一张人物照＋一段提示词 → 会说话的视频；也可以交给桌面版 AI 用 skill 生成 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dseditor/minimax-h3-colab/blob/main/H3_Digital_Human.ipynb) |
| **minimax_h3_image_colab** | Toybox 网页、原有视频与 P-H3-Image、独立 H3-Image 4K 生图 | [私有仓库运行说明](#在-colab-运行私有仓库版本) |
| **H3_Toybox** | 与增强版入口使用同一份启动逻辑 | [H3_Toybox.ipynb](H3_Toybox.ipynb) |

## 在 Colab 运行私有仓库版本

1. 将本项目推送到自己的 GitHub 私有仓库。在 Colab 选择「文件 → 打开笔记本 → GitHub」，勾选「包含私有仓库」并授权，打开 `minimax_h3_image_colab.ipynb`。也可把本地的同名笔记本上传到 Colab。
2. 笔记本默认从 `Alex-955/minimax-h3-colab` 的 `main` 分支读取源码。使用其他仓库时，可在第 ① 格填写 `REPO_URL` 和 `REPO_BRANCH`，或在 Colab「密钥」里设置 `H3_TOYBOX_REPO` 和 `H3_TOYBOX_BRANCH`。
3. 给私有仓库创建一个仅对该仓库开放 **Contents: Read-only** 的 GitHub fine-grained token。在 Colab「密钥」添加 `GITHUB_TOKEN` 并允许笔记本访问。打开 GitHub 笔记本的授权与运行时拉取源码的凭据各有用途。[GitHub 令牌说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
4. 选 L4 24 GB 或 A100 40/80 GB，再点「运行时 → 全部运行」。首次使用 Drive 时按 Colab 提示授权，打开第 ③ 格给出的 Toybox 链接。

第 ② 格的 `INSTALL_H3_IMAGE` 默认开启。它从 [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main) 下载 INT8 扩散权重和 FP16 图片 VAE（约 **22.15 GB**），放到 `ComfyUI/models/diffusion_models/` 和 `ComfyUI/models/vae/`。视频与 H3-Image 共用 `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`。已有完整文件会跳过，勾选 `USE_DRIVE_CACHE` 可沿用 `MyDrive/H3_models`。默认全选约需 83.20 GB 权重空间，下载单元会按实际文件检查剩余磁盘。

在 Toybox 打开 **H3-Image（4K 生图）**，可选 1K、2K、4K 或自定义尺寸。4K 正方形是 4096×4096，3072×4096 可直接填入自定义宽高。参考图支持最多 16 张。结果卡片上的「发送到图生视频」会把原图加入素材库并填到首帧，确认动作提示词后再点击生成视频。L4 上的 4K 和多图耗用取决于可用显存与主机内存。

---

## 需要准备

- **Google AI Pro 订阅**（Colab Pro 的运算单元）。L4 约每小时 1.5 个运算单元。
- 运行阶段选 **L4 GPU**。A100 也能跑但很耗点数；**T4 跑不动**（int8 模型需要 compute capability 8.0 以上）。
- （建议）**Hugging Face 只读令牌**：左侧栏「🔑 密钥」添加 `HF_TOKEN`，并打开「笔记本访问权」。没有也能下载，只是人多时可能被限速。

## L4 实测时间

| 步骤 | 时间 |
|---|---|
| ① 检查环境＋安装 ComfyUI | 约 45 秒 |
| ② 启动 ComfyUI | 约 30 秒 |
| ④ 下载模型（约 37 GB） | 约 1 分钟 |
| 生成 10 秒视频（0.4 百万像素） | 约 4 分钟 |

---

## 操作步骤

### 1. 打开笔记本

点上方表格的 **Open in Colab**。也可以在 Colab 首页按「打开笔记本 → GitHub」，搜索 `dseditor/minimax-h3-colab`。

![Colab 首页](docs/images/01_colab_home.png)
![从 GitHub 打开](docs/images/02_open_from_github.png)

打开后先到「运行阶段 → 变更运行阶段类型」选 **L4 GPU**。

### 2. 由上往下运行 ①②③

- **①** 先检查 GPU 与驱动，不符合就直接停下、不会安装任何东西；通过才安装 ComfyUI（CUDA 13 版 PyTorch、节点管理器、[ListHelper](https://github.com/dseditor/ComfyUI-ListHelper) 节点组）。
- **②** 在背景启动 ComfyUI。
- **③** 用 Colab 内置代理打打开界面：下方出现的链接显示成 `localhost` 也没关系，**直接点它**。

![① ② ③ 运行结果](docs/images/03_cells_1_to_3.png)

> ③ 需要浏览器登录**跟 Colab 同一个 Google 账号**。账号不同、或要把接口交给 AI 操作时，改用 **③-b**（cloudflare 网址）。
> ⚠️ ③-b 的网址没有密码，拿到的人都能用你的 GPU，不要贴到公开的地方。

### 3. ④ 下载模型

ComfyUI 确认正常之后才下载，避免装坏了还白抓 37 GB。

![下载模型](docs/images/04_download_models.png)

`USE_DRIVE_CACHE` 默认不勾：直接从 Hugging Face 下载最快。只有撞到 HF 流量上限时才勾选，改从「我的Google Drive/H3_models」拷贝（第一次会顺便存一份）。

![从Google Drive拷贝](docs/images/05_drive_cache.png)

### 4a. 基础版：开始玩 ComfyUI

到这里就装好了。在界面里拉节点、或把自己的工作流 JSON 拖进画面。节点管理器已打开，可以自由安装自定义节点。

### 4b. 数字人版：⑤ 上传照片 → ⑥ 贴提示词 → ⑦ 生成

数字人笔记本的 ② 会把工作流放进 ComfyUI 左侧的工作流清单，想看节点可以直接点开。

![工作流清单](docs/images/06_comfyui_workflow.png)

- **⑤** 上传一张人物照。
- **⑥** 运行后下方出现大输入框，把整段提示词直接贴进去（不用管引号、换行），并设置秒数（5～15）、分辨率、seed。重跑这格不会洗掉已贴的内容。
- **⑦** 生成，完成后在笔记本里播放并自动下载。

![提示词输入框](docs/images/07_prompt_box.png)

提示词要照 MiniMax 官方的 Ref2VA 格式写（英文本文、台词保留原文）。可以请 AI 用官方 [`h3-prompt-writing`](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/.claude/skills/h3-prompt-writing) skill 产生。

---

## 交给桌面版 AI：`h3-colab-digital-human` skill

`skills/h3-colab-digital-human/` 让能运行本机 PowerShell（或 Python）的桌面版 AI（Claude Desktop、ChatGPT Desktop 等）直接帮你生成：

1. 你在 Colab 跑 **①、②、③-b、④**（两本笔记本都可以），把 ③-b 的网址交给 AI。
2. 给 AI 一张人物照的路径和一句中文需求（说什么、什么语气、几秒）。
3. AI 看照片写提示词 → 通过 ComfyUI API 上传、生成、下载视频到你的电脑。

安装：把 `skills/h3-colab-digital-human` 整个文件夹放进 AI 的 skills 目录。官方提示词指南不随本 repo 散布，skill 会引导 AI 从 MiniMax 官方 repo 下载（见 `references/H3_NOTICE.md`）。

## H3 Toybox：网页前端版

`minimax_h3_image_colab.ipynb` 的 ①②③ 负责安装、下载权重和打开 Toybox 网页。ComfyUI 在后台运行，网页操作见 [Toybox 使用说明](toybox/README.md)。

- 视频权重约 51 GB，含基础模型、LoRA、语义桥、潜空间放大器与 9B 提示词改写模型。取消 `INCLUDE_REWRITER` 可省约 7 GB，取消 `INSTALL_IMAGE` 可省约 10 GB 的旧 P-H3-Image 权重。 [Toybox 附加权重](https://huggingface.co/dseditor/minimax-h3-studio-assets)
- 节点组：T8 audio（语义桥）、SelfLift（两段采样）、H3 Motion Context（延伸视频）。
- 官方提示词指南不随本 repo 散布，② 会从 MiniMax 官方 repo 下载到 `toybox/guides/`。
- 本机版在 Windows 用 ComfyUI 内嵌 Python 直接跑 `toybox/server.py`；`config.json` 指定 ComfyUI 网址与 input 文件夹。

---

## 模型

| 文件 | 来源 | 大小 |
|---|---|---|
| `Minimax-h3_Singularity_x_10Eros-Max_PRISM_int8` | [OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM](https://huggingface.co/OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM) | 18.9 GB |
| `qwen3vl_32b_minimax_h3_nvfp4_awq` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | 14.6 GB |
| `minimax_h3_video_vae_int8_convrot` | Comfy-Org/MiniMax-H3 | 2.6 GB |
| `minimax_h3_audio_vae_fp32` | Comfy-Org/MiniMax-H3 | 0.6 GB |

主模型是 int8 融合模型，加速已融合，6 步出片，不需要另插件 LoRA。各模型授权依照其来源。

## 踩过的坑（笔记本里已处理）

| 症状 | 原因 | 处理 |
|---|---|---|
| cloudflare 网址打开界面显示「访问遭拒」403 | 浏览器 Origin 与转进来的 Host 对不上 | 启动加 `--enable-cors-header` |
| cloudflare 网址卡在加载画面 | 通道输出接 PIPE 没人读，塞满就停 | 输出改写进文件 |
| Colab 代理下点开工作流清单出现 404 | 代理把网址里的 `%2F` 解开，路径变两段 | 预装一个补路由的小节点（`tools/userdata_slash_fix.py`） |
| 驱动版本太旧 | int8 ConvRot 需要 CUDA 13（驱动 580 以上） | ① 先检查，不符就停下 |

## 给维护者

四本笔记本由 `tools/build_notebooks.py` 生成。修改脚本后运行以下命令重新生成。

```bash
python tools/build_notebooks.py
```

- `workflows/H3_digital_human.json`：ComfyUI 界面格式工作流
- `workflows/H3_digital_human_api.json`：API 格式（⑦ 与 skill 用）
- `workflows/minimax_h3_image_api.json`：H3-Image API 工作流，6 步、CFG 1.0。Toybox 根据请求填入提示词、宽高、种子和 `images.image_1` 等参考图输入。

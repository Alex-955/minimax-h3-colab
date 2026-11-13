# -*- coding: utf-8 -*-
"""生成基础版、数字人版和两份 Toybox Colab 入口。请修改此脚本后重新生成笔记本。"""
import json, io, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
api = json.load(io.open('workflows/H3_digital_human_api.json', encoding='utf-8'))
fix_src = io.open('tools/userdata_slash_fix.py', encoding='utf-8').read()
ui_wf = io.open('workflows/H3_digital_human.json', encoding='utf-8').read()
image_wf = json.load(io.open('workflows/minimax_h3_image_api.json', encoding='utf-8'))
image_files = [f'{folder}/{image_wf[node]["inputs"][field]}' for folder, node, field in (
    ('diffusion_models', 'iu', 'unet_name'), ('vae', 'iv', 'vae_name'),
)]
shared_image_encoder = 'text_encoders/' + image_wf['il']['inputs']['clip_name']
sample_prompt = api['7']['inputs']['prompt']
api['7']['inputs']['prompt'] = ''


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s}


def code(s):
    return {'cell_type': 'code', 'metadata': {'cellView': 'form'}, 'execution_count': None, 'outputs': [], 'source': s}


TQ = '"' * 3
cells = [
    md("""# MiniMax H3 数字人｜Colab 版（一张照片＋一段提示词 → 会说话的视频）

由上往下依序运行。**运行阶段请选 L4 GPU**（A100 也可以但很耗点数；T4 跑不动）。

| 步骤 | 做什么 | 时间（主人 L4 实测） |
|---|---|---|
| ① 检查＋安装 | GPU／驱动不符就停下，不安装；通过才装 ComfyUI＋节点 | 约 1 分钟 |
| ② 启动 | 背景启动 ComfyUI，工作流放进清单 | 约 1 分钟 |
| ③ 打开界面 | Colab 内置代理（推荐）；③-b 是 cloudflare 备用 | 数秒 |
| ④ 下载模型 | 约 37 GB（ComfyUI 确定正常才下载） | 约 1 分钟 |
| ⑤ 上传照片 | 人物照一张 | — |
| ⑥ 贴提示词 | 用 H3 提示词 skill 产生后整段粘贴 | — |
| ⑦ 生成 | 10 秒视频 | 约 4 分钟 |

只想用界面操作节点：跑 ①②③④ 就好。只想用表单生成视频：③ 可以跳过。

**Hugging Face 令牌（建议）**：左侧栏「🔑 密钥」添加 `HF_TOKEN`（请用**只读**令牌），并打开「笔记本访问权」。

**交给桌面版 AI 生成**：跑 ①②③-b④，把 ③-b 的网址交给装了 `h3-colab-digital-human` skill 的 AI（skill 在 repo 的 `skills/`）。

模型：`Minimax-h3_Singularity_x_10Eros-Max_PRISM`（int8，加速已融合，6 步）＋官方 Qwen3-VL 文本编码器与 VAE。授权依照各来源模型。

最新版：<https://github.com/dseditor/minimax-h3-colab>"""),
    code("#@title ① 检查环境＋安装 ComfyUI（GPU 或驱动不符就停下，不会安装）\nFIX_SRC = " + repr(fix_src) + """
import torch, shutil, subprocess, sys, os
from pathlib import Path
if not torch.cuda.is_available():
    raise RuntimeError('没有 GPU：请到「运行阶段 → 变更运行阶段类型」选 L4 GPU，再重跑这格')
cap = torch.cuda.get_device_capability(0)
drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
print('GPU:', torch.cuda.get_device_name(0), '| capability', cap, '| VRAM', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')
print('驱动:', drv, '| 目前 PyTorch CUDA:', torch.version.cuda, '| 可用磁盘', round(shutil.disk_usage('/content').free / 2**30, 1), 'GiB')
if cap < (8, 0):
    raise RuntimeError(f'这张卡（capability {cap}）跑不动 int8 模型，没有安装任何东西。请换 L4（或 A100）再重跑。')
if int(drv.split('.')[0]) < 580:
    raise RuntimeError(f'驱动 {drv} 不支持 CUDA 13（需要 580 以上），没有安装任何东西。请换一台运行阶段（L4）再试。')
print('✅ 环境检查通过，开始安装')
# HF 令牌先放进环境变量：之后启动的 ComfyUI（含 ListHelper 的模型下载节点）都会沿用
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN'); print('已读到 HF_TOKEN')
except Exception:
    print('没有设置 HF_TOKEN，之后改用匿名下载（人多时可能被限速）')
COMFY = Path('/content/ComfyUI')
if not COMFY.exists():
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/Comfy-Org/ComfyUI.git', str(COMFY)], check=True)
if not (torch.version.cuda or '').startswith('13'):
    print('安装 CUDA 13 版 PyTorch（int8 ConvRot 权重需要）…')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch', 'torchvision', 'torchaudio',
                    '--index-url', 'https://download.pytorch.org/whl/cu130'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(COMFY / 'requirements.txt'), 'huggingface_hub', 'hf_xet'], check=True)
# 预装 ListHelper 节点组（含 ModelDownloader：在界面里从 Hugging Face 抓模型）
LH = COMFY / 'custom_nodes' / 'ComfyUI-ListHelper'
if not LH.exists():
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/dseditor/ComfyUI-ListHelper.git', str(LH)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(LH / 'requirements.txt')], check=True)
# Colab 内置代理（③）会把网址的 %2F 解开，工作流清单点开会 404 → 装一个补路由的小节点
FIX = COMFY / 'custom_nodes' / 'colab_userdata_slash_fix'
FIX.mkdir(parents=True, exist_ok=True)
(FIX / '__init__.py').write_text(FIX_SRC, encoding='utf-8')
# 节点管理器（ComfyUI-Manager 已并入官方，用 pip 套件＋启动旗标 --enable-manager）
if (COMFY / 'manager_requirements.txt').exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(COMFY / 'manager_requirements.txt')], check=True)
out = subprocess.run([sys.executable, '-c', 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'],
                     capture_output=True, text=True)
print('ComfyUI 会用的 PyTorch：', out.stdout.strip(), out.stderr[-300:])
"""),
    code("#@title ② 启动 ComfyUI（背景运行）＋把 H3 数字人工作流放进清单\nUI_WORKFLOW = " + repr(ui_wf) + """
import subprocess, time, urllib.request, sys
# --enable-cors-header：经 cloudflare 通道打开界面时，浏览器的 Origin（trycloudflare 网址）
# 跟转进来的 Host（127.0.0.1）对不上，ComfyUI 默认会回 403「访问遭拒」；开这个旗标才放行。
if 'server' in globals() and server.poll() is None:
    server.terminate(); server.wait(); print('已关闭旧的 ComfyUI，重新启动')
# --enable-compress-response-body：接口要抓一大包节点清单和前端文件，压缩后走通道快很多。
# 节点管理器安全等级：normal- ＝可从清单装节点，也可用 Git 网址安装（本机模式下）。
# 这台 Colab 用完即丢；但 HF_TOKEN 在环境变量里，自定义节点读得到 → HF 令牌请用「只读」的。
cfg = COMFY / 'user' / '__manager' / 'config.ini'
cfg.parent.mkdir(parents=True, exist_ok=True)
if not cfg.exists():
    cfg.write_text('[default]\\nsecurity_level = normal-\\n', encoding='utf-8')
# 工作流放进清单（③、③-b 都看得到）；文件名用英文，避免网址编码在代理上出问题
wf_dir = COMFY / 'user' / 'default' / 'workflows'
wf_dir.mkdir(parents=True, exist_ok=True)
(wf_dir / 'H3_digital_human.json').write_text(UI_WORKFLOW, encoding='utf-8')
for old_name in ('H3数字人.json', 'H3數字人.json'):
    (wf_dir / old_name).unlink(missing_ok=True)
Path('/content/H3_digital_human.json').write_text(UI_WORKFLOW, encoding='utf-8')
log = open('/content/comfyui.log', 'w')
server = subprocess.Popen([sys.executable, 'main.py', '--listen', '127.0.0.1', '--port', '8188', '--enable-cors-header', '--enable-manager', '--enable-compress-response-body'],
                          cwd=str(COMFY), stdout=log, stderr=subprocess.STDOUT)
for i in range(180):
    try:
        urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=2)
        print('ComfyUI 已启动'); break
    except Exception:
        time.sleep(2)
else:
    print(open('/content/comfyui.log').read()[-3000:])
    raise RuntimeError('ComfyUI 没有启动成功，看上面的纪录')
"""),
    code("""#@title ③ 打开 ComfyUI 界面（推荐：Colab 内置代理，快、稳、网址不外流）
# 运行后下面会出现一个链接；显示成 localhost 也没关系，直接点它就会开新标签页。
# ⚠️ 打开标签页的浏览器必须登录「跟这本 Colab 同一个 Google 账号」；账号不同就改用下一格 ③-b。
from google.colab import output
output.serve_kernel_port_as_window(8188)
# 如果左侧工作流清单打不开（404），按下面下载 JSON，直接拖进 ComfyUI 画面就好
from google.colab import files
files.download('/content/H3_digital_human.json')
"""),
    code("""#@title ③-b（备用）用 cloudflare 网址打开界面：浏览器登录的 Google 账号和 Colab 不同时用这格
# 会产生一个 trycloudflare 网址，点开就是 ComfyUI；也是交给桌面版 AI 操作时要给的网址。
# ⚠️ 网址没有密码，拿到的人都能用你的 GPU —— 不要贴到公开的地方。
import subprocess, re, time, json, urllib.request
if not Path('/content/cloudflared').exists():
    subprocess.run(['wget', '-q', '-O', '/content/cloudflared',
                    'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'], check=True)
    subprocess.run(['chmod', '+x', '/content/cloudflared'], check=True)
# 通道的输出写进文件，不要接 PIPE：PIPE 没人读会塞满，通道就卡住，网页停在加载画面
if 'tun' in globals() and tun.poll() is None:
    tun.terminate(); tun.wait()
open('/content/cloudflared.log', 'w').close()
tun = subprocess.Popen(['/content/cloudflared', 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:8188'],
                       stdout=open('/content/cloudflared.log', 'a'), stderr=subprocess.STDOUT)
url = None
for i in range(60):
    time.sleep(1)
    m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', open('/content/cloudflared.log').read())
    if m:
        url = m.group(0); break
print('👉 ComfyUI 界面：', url or '（没拿到网址，看 /content/cloudflared.log）')
"""),
    code("""#@title ④ 下载模型（约 37 GB）
# 默认直接从 Hugging Face 下载（最快）。HF 令牌已在 ① 读入。
# 只有在 Hugging Face 撞到流量上限时，才勾选 USE_DRIVE_CACHE：
#   勾选后会从「我的Google Drive/H3_models」拷贝（比 HF 慢），Google Drive没有的文件才去 HF 抓并顺便存一份。
USE_DRIVE_CACHE = False  #@param {type:"boolean"}
from huggingface_hub import hf_hub_download
import shutil, os
CACHE = None
if USE_DRIVE_CACHE:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE = Path('/content/drive/MyDrive/H3_models'); CACHE.mkdir(parents=True, exist_ok=True)
files = [
    ('OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM', 'Minimax-h3_Singularity_x_10Eros-Max_PRISM_int8.safetensors', 'diffusion_models'),
    ('Comfy-Org/MiniMax-H3', 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'text_encoders'),
    ('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_video_vae_int8_convrot.safetensors', 'vae'),
    ('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_audio_vae_fp32.safetensors', 'vae'),
]
for repo, name, folder in files:
    target = COMFY / 'models' / folder / Path(name).name
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = CACHE / Path(name).name if CACHE else None
    if not target.is_file():
        if cached and cached.is_file():
            print('从Google Drive拷贝', cached.name); shutil.copy(str(cached), str(target))
        else:
            got = Path(hf_hub_download(repo_id=repo, filename=name, local_dir='/content/hf'))
            shutil.move(str(got), str(target))
    if cached and not cached.is_file():
        print('存一份到Google Drive', cached.name); shutil.copy(str(target), str(cached))
    print(folder, target.name, f'{target.stat().st_size / 2**30:.1f} GiB')
"""),
    code("""#@title ⑤ 上传人物照（一张）
from google.colab import files
from IPython.display import Image as _Img, display
up = files.upload()
IMAGE_NAME = list(up)[0]
(COMFY / 'input').mkdir(exist_ok=True)
(COMFY / 'input' / IMAGE_NAME).write_bytes(up[IMAGE_NAME])
display(_Img(data=up[IMAGE_NAME], width=320))
"""),
    code("#@title ⑥ 粘贴提示词＋设置（运行后下方会出现输入框）\n"
         "# 运行这格后，下面会出现一个大的输入框：把 skill 产生的整段提示词直接贴进去就好，\n"
         "# 不用管引号、换行。贴完不用再运行这格，直接运行 ⑦。\n"
         "import ipywidgets as W\n"
         "from IPython.display import display\n"
         "SAMPLE_PROMPT = " + repr(sample_prompt) + "\n"
         """_keep = prompt_box.value if 'prompt_box' in globals() else SAMPLE_PROMPT   # 重跑这格不会洗掉已贴的内容
prompt_box = W.Textarea(value=_keep, placeholder='把整段提示词贴在这里',
                        layout=W.Layout(width='100%', height='320px'))
dur_box = W.IntSlider(value=10, min=5, max=15, step=1, description='秒数')
mp_box = W.FloatText(value=0.4, description='百万像素')
seed_box = W.IntText(value=-1, description='seed（-1 随机）')
clear_btn = W.Button(description='清空输入框')
clear_btn.on_click(lambda _: setattr(prompt_box, 'value', ''))
count = W.HTML()
def _count(change=None):
    n = len(prompt_box.value.split())
    count.value = f'<b>{n}</b> 个英文本（官方建议 350～500）'
prompt_box.observe(_count, names='value'); _count()
display(W.VBox([W.HBox([clear_btn, count]), prompt_box, dur_box, mp_box, seed_box]))
"""),
    code("#@title ⑦ 生成视频\n"
         "import json, random, time, urllib.request\n"
         "from IPython.display import Video, display\n"
         "WORKFLOW = json.loads(" + repr(json.dumps(api, ensure_ascii=False)) + ")\n"
         """p = json.loads(json.dumps(WORKFLOW))
PROMPT = prompt_box.value.strip()
assert PROMPT, '⑥ 的输入框是空的，请先粘贴提示词'
print('这次送出的提示词开头：', PROMPT[:160].replace(chr(10), ' '), '…（共', len(PROMPT.split()), '字）')
DURATION_SECONDS, MEGAPIXELS, SEED = dur_box.value, mp_box.value, seed_box.value
p['7']['inputs']['prompt'] = PROMPT
p['5']['inputs']['image'] = IMAGE_NAME
p['56']['inputs']['value'] = int(DURATION_SECONDS)
p['61']['inputs']['megapixels'] = float(MEGAPIXELS)
p['54']['inputs']['seed'] = SEED if SEED >= 0 else random.randint(0, 2**50)
req = urllib.request.Request('http://127.0.0.1:8188/prompt', data=json.dumps({'prompt': p}).encode(),
                             headers={'Content-Type': 'application/json'})
try:
    pid = json.loads(urllib.request.urlopen(req).read())['prompt_id']
except urllib.error.HTTPError as e:
    raise RuntimeError(e.read().decode()[:2000])
t0 = time.time(); print('已送出，生成中…（seed', p['54']['inputs']['seed'], '）')
while True:
    time.sleep(10)
    h = json.loads(urllib.request.urlopen('http://127.0.0.1:8188/history/' + pid).read())
    if pid in h:
        st = h[pid]['status']
        if st.get('status_str') == 'error':
            print(json.dumps(st, ensure_ascii=False)[:3000]); raise RuntimeError('生成失败')
        break
    print(f'  {int(time.time() - t0)} 秒', end='\\r')
vids = [v for o in h[pid]['outputs'].values() for v in o.get('images', []) + o.get('videos', []) if v['filename'].endswith('.mp4')]
out = COMFY / 'output' / vids[0]['subfolder'] / vids[0]['filename']
print(f'完成，{(time.time() - t0) / 60:.1f} 分钟：', out)
display(Video(str(out), embed=True, width=640))
from google.colab import files; files.download(str(out))
"""),
    code("""#@title 🔍 出问题时跑这格：看 GPU 与 ComfyUI 纪录
import subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
log = open('/content/comfyui.log', encoding='utf-8', errors='ignore').read()
keys = ('error', 'Error', 'int8', 'quant', 'fallback', 'out of memory', 'loaded', 'Prompt executed')
print('\\n'.join(l for l in log.splitlines() if any(k in l for k in keys))[-4000:])
"""),
]


def without(cell, *snippets):
    """从共用格子拿掉数字人专用的片段；片段找不到就报错，避免改了上面却静默没拿掉。"""
    src = cell['source']
    for s in snippets:
        assert s in src, '找不到要拿掉的片段：' + s[:60]
        src = src.replace(s, '')
    return dict(cell, source=src)


HEAD_BASIC = md("""# MiniMax H3｜ComfyUI 基础版（Colab）

在 Colab 上装好 ComfyUI 与 MiniMax H3 模型组，**不带任何工作流**——打开界面后自己拉节点、或拖入自己的工作流学习。
由上往下依序运行。**运行阶段请选 L4 GPU**（A100 也可以但很耗点数；T4 跑不动）。

| 步骤 | 做什么 | 时间（L4 实测） |
|---|---|---|
| ① 检查＋安装 | GPU／驱动不符就停下，不安装；通过才装 ComfyUI＋节点管理器＋ListHelper | 约 1 分钟 |
| ② 启动 | 背景启动 ComfyUI | 约 1 分钟 |
| ③ 打开界面 | Colab 内置代理（推荐）；③-b 是 cloudflare 备用 | 数秒 |
| ④ 下载模型 | 约 37 GB（ComfyUI 确定正常才下载） | 约 1 分钟 |

想直接做「会说话的数字人」：改开同一个 repo 的 `H3_Digital_Human.ipynb`。

**Hugging Face 令牌（建议）**：左侧栏「🔑 密钥」添加 `HF_TOKEN`（请用**只读**令牌），并打开「笔记本访问权」。

模型：`Minimax-h3_Singularity_x_10Eros-Max_PRISM`（int8，加速已融合，6 步）＋官方 Qwen3-VL 文本编码器与 VAE。授权依照各来源模型。

最新版：<https://github.com/dseditor/minimax-h3-colab>""")

C2_BASIC = without(cells[2],
    '＋把 H3 数字人工作流放进清单\nUI_WORKFLOW = ' + repr(ui_wf),
    """# 工作流放进清单（③、③-b 都看得到）；文件名用英文，避免网址编码在代理上出问题
wf_dir = COMFY / 'user' / 'default' / 'workflows'
wf_dir.mkdir(parents=True, exist_ok=True)
(wf_dir / 'H3_digital_human.json').write_text(UI_WORKFLOW, encoding='utf-8')
for old_name in ('H3数字人.json', 'H3數字人.json'):
    (wf_dir / old_name).unlink(missing_ok=True)
Path('/content/H3_digital_human.json').write_text(UI_WORKFLOW, encoding='utf-8')
""")
C3_BASIC = without(cells[3], """# 如果左侧工作流清单打不开（404），按下面下载 JSON，直接拖进 ComfyUI 画面就好
from google.colab import files
files.download('/content/H3_digital_human.json')
""")

# ---------------------------------------------------------------- H3 Toybox（网页前端）
# ComfyUI 只当背景引擎、不开它的接口；浏览器只开 toybox/ 这个前端（Colab 内置代理）。
TOYBOX = [
    md("""# H3 Toybox｜Colab 网页版

把想象丢进玩具箱，摇一摇就变成视频 🧸 ComfyUI 只在背景当引擎，**不用开 ComfyUI 界面**，所有操作都在 H3 Toybox 网页里。

运行环境选 **L4 24 GB** 或 **A100 40/80 GB**。首次配置源码地址和密钥后，选择「运行时 → 全部运行」，再打开第 ③ 格的 Toybox 链接。

| 步骤 | 做什么 |
|---|---|
| ① 检查并安装 | 更新 ComfyUI 主分支、安装依赖与视频节点，从指定仓库拉取 Toybox |
| ② 下载模型 | 视频约 50 GB，P-H3-Image 另加约 10 GB，H3-Image 另加约 22.15 GB。下载前按所选模型检查磁盘 |
| ③ 启动＋开网页 | 背景启动 ComfyUI 与 H3 Toybox，用 Colab 内置代理开网页；作品默认存到Google Drive「H3Toybox」文件夹 |
| ③-b | 备用 Cloudflare 链接，默认关闭，可按需手动开启 |

**网页里能做的事**
- 🖼 **图片**（选用）：文生图，或拿最多 8 张图做编辑（换表情、换衣服、换风格、把人放进场景）。用 [P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image-Comfy) 挂在同一颗 H3 模型上，只多约 10 GB
- **H3-Image（4K 生图）**：1K、2K、4K、自定义宽高，支持最多 16 张参考图。生成后可把原图发到视频首帧
- 🤖 **短片 Agent**：一句话规划成多段短片（要先在 ⚙ 设置填 Gemini 密钥）
- 🧸 **参考生成**：图片 9 张／视频 3 支／声音 3 段当参考，提示词里用 `@图片1` 标记
- 🎞 **首尾帧**：只放首帧、只放尾帧、或两张都放
- ✏️ **纯文本**：只靠一段描述
- 🧵 **延伸视频**：参考视频勾「延伸这支视频」，新片段会从它的最后一格接下去，并自动接回原片
- ✨ **提示词改写**：把中文描述扩写成 H3 规格的英文提示词（改写结果默认收起来）。默认用 H3 官方的 9B 改写模型；也可以在 ⚙ 设置改用 **Gemini** 或 **OpenAI 兼容端点**
- ⚡ **快速模式**（默认开）：8 步出片；关掉是 12 步
- 📁 **素材库文件夹**：把图片、视频、声音拖进文件夹整理
- 每支视频都会显示目前做到哪一步、还要多久

**源码仓库**：默认使用 `Alex-955/minimax-h3-colab` 的 `main` 分支。使用其他仓库时，可在第 ① 格填写地址和分支，或在 Colab「密钥」添加 `H3_TOYBOX_REPO` 和 `H3_TOYBOX_BRANCH`。私有仓库需要 `GITHUB_TOKEN`（对该仓库的 Contents 只读权限）。打开相应密钥的「笔记本访问权」。

**Hugging Face 令牌（可选）**：在同一处添加只读 `HF_TOKEN`。

模型：`Minimax-h3_Singularity_x_10Eros-Max_PRISM`（int8）＋官方 Qwen3-VL 文本编码器与 VAE；LoRA、语义桥、潜空间放大器与改写模型镜像在 [dseditor/minimax-h3-studio-assets](https://huggingface.co/dseditor/minimax-h3-studio-assets)（各档来源见该页）。授权依照各来源模型。

L4 的 4K 和多图任务取决于可用显存与主机内存。可先用 1K 或 2K 确认运行环境。"""),
    code("""#@title ① 检查环境＋安装 ComfyUI、节点与 H3 Toybox（GPU 或驱动不符就停下，不会安装）
REPO_URL = ''  #@param {type:"string"}
REPO_BRANCH = ''  #@param {type:"string"}
import base64, os, re, shutil, subprocess, sys, torch
from pathlib import Path
from google.colab import userdata

def colab_secret(name):
    try:
        return userdata.get(name) or ''
    except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
        return ''

repo_url = (REPO_URL.strip() or colab_secret('H3_TOYBOX_REPO').strip()
            or 'https://github.com/Alex-955/minimax-h3-colab').rstrip('/')
repo_branch = (REPO_BRANCH.strip() or colab_secret('H3_TOYBOX_BRANCH') or 'main').strip()
if not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo_url):
    raise ValueError('请填写改造版 GitHub 仓库地址，或设置 Colab 密钥 H3_TOYBOX_REPO。地址中请勿放令牌。')
repo_url = repo_url.removesuffix('.git') + '.git'
if not torch.cuda.is_available():
    raise RuntimeError('没有 GPU：请到「运行阶段 → 变更运行阶段类型」选 L4 GPU，再重跑这格')
cap = torch.cuda.get_device_capability(0)
drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
print('GPU:', torch.cuda.get_device_name(0), '| capability', cap, '| VRAM', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')
print('驱动:', drv, '| 可用磁盘', round(shutil.disk_usage('/content').free / 2**30, 1), 'GiB')
if cap < (8, 0):
    raise RuntimeError(f'这张卡（capability {cap}）跑不动 int8 模型，没有安装任何东西。请换 L4（或 A100）再重跑。')
if int(drv.split('.')[0]) < 580:
    raise RuntimeError(f'驱动 {drv} 不支持 CUDA 13（需要 580 以上），没有安装任何东西。请换一台运行阶段（L4）再试。')
print('✅ 环境检查通过，开始安装')
hf_token = colab_secret('HF_TOKEN')
if hf_token:
    os.environ['HF_TOKEN'] = hf_token
    print('已读取 HF_TOKEN')
else:
    print('未设置 HF_TOKEN，使用匿名下载')

def clone(url, dest, *, branch=None, env=None):
    dest = Path(dest)
    if not dest.exists():
        cmd = ['git', 'clone', '--depth', '1']
        if branch:
            cmd += ['--branch', branch]
        subprocess.run(cmd + [url, str(dest)], env=env, check=True)
    else:
        git = ['git', '-C', str(dest)]
        remote = subprocess.check_output(git + ['remote', 'get-url', 'origin'], text=True).strip()
        current = subprocess.check_output(git + ['branch', '--show-current'], text=True).strip()
        if remote.removesuffix('.git') != url.removesuffix('.git') or (branch and current != branch):
            raise RuntimeError(f'{dest} 已检出其他仓库或分支，请在新的 Colab 运行时重试')
        subprocess.run(git + ['pull', '--ff-only'], env=env, check=True)

source = Path('/content/minimax-h3-toybox')
git_env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
github_token = colab_secret('GITHUB_TOKEN')
if github_token:
    # 仅传给这次 Git 调用。远程地址、Git 配置文件和后续进程中都不写入令牌。
    basic = base64.b64encode(f'x-access-token:{github_token}'.encode()).decode()
    git_env.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
                   GIT_CONFIG_VALUE_0=f'AUTHORIZATION: basic {basic}')
try:
    clone(repo_url, source, branch=repo_branch, env=git_env)
except subprocess.CalledProcessError as e:
    raise RuntimeError('拉取 Toybox 失败，请检查仓库地址、分支和私有仓库 GITHUB_TOKEN 的读取权限') from e
finally:
    del git_env, github_token
    if 'basic' in globals():
        del basic
APP = source / 'toybox'
if not (APP / 'server.py').is_file() or not (source / 'workflows/minimax_h3_image_api.json').is_file():
    raise RuntimeError('该分支缺少改造版 Toybox 或 H3-Image 工作流，请检查是否已推送代码')

COMFY = Path('/content/ComfyUI')
clone('https://github.com/Comfy-Org/ComfyUI.git', COMFY, branch='master')
torch_cuda = subprocess.check_output([sys.executable, '-c', 'import torch; print(torch.version.cuda)'], text=True).strip()
if not torch_cuda.startswith('13'):
    print('安装 CUDA 13 版 PyTorch（int8 ConvRot 权重需要）…')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'torch', 'torchvision', 'torchaudio',
                    '--index-url', 'https://download.pytorch.org/whl/cu130'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(COMFY / 'requirements.txt'), 'huggingface_hub', 'hf_xet'], check=True)
# 工作流用到的三个节点组（都不会动到 PyTorch）
NODES = {
    'minimax-h3-audio-t8': 'https://github.com/T8mars/comfyui-minimax-h3-audio-T8',        # 语义桥 Semantic Bridge
    'comfyui-SelfLift': 'https://github.com/facok/comfyui-SelfLift',                      # 低分辨率→潜空间放大→高分辨率
    'ComfyUI-H3-Motion-Context': 'https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context',  # 延伸视频（气锁）
}
for name, url in NODES.items():
    clone(url, COMFY / 'custom_nodes' / name)
# 导演 Skill 只是文本档：放在 custom_nodes 外面，不加载它的节点
clone('https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8', '/content/t8-enhancer')
print('ffmpeg:', shutil.which('ffmpeg') or '没有（延伸视频会失败）')
subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available() and torch.version.cuda.startswith("13"), (torch.__version__, torch.version.cuda); print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)'],
               check=True)
print('✅ 安装完成')
"""),
    code("""#@title ② 下载视频和生图权重
# 默认直接从 Hugging Face 下载（最快）。HF 令牌已在 ① 读入。
# 只有在 Hugging Face 撞到流量上限时，才勾选 USE_DRIVE_CACHE：
#   勾选后会从「我的Google Drive/H3_models」拷贝（比 HF 慢），Google Drive没有的文件才去 HF 抓并顺便存一份。
USE_DRIVE_CACHE = False  #@param {type:"boolean"}
#@markdown 不勾就少下载 7 GB：网页的本地改写会关掉，可以到 ⚙ 设置改用 Gemini 或 OpenAI 端点改写。
INCLUDE_REWRITER = True  #@param {type:"boolean"}
#@markdown 🖼 加装生图（多约 10 GB）：网页的「图片」分页可以文生图、拿最多 8 张图做编辑。挂在同一颗 H3 模型上。
INSTALL_IMAGE = True  #@param {type:"boolean"}
#@markdown H3-Image 独立 4K 生图权重，约 22.15 GB，复用视频的 Qwen3-VL 编码器。
INSTALL_H3_IMAGE = True  #@param {type:"boolean"}
from huggingface_hub import HfApi, hf_hub_download
import shutil, time
from collections import defaultdict
CACHE = None
if USE_DRIVE_CACHE:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE = Path('/content/drive/MyDrive/H3_models'); CACHE.mkdir(parents=True, exist_ok=True)
ASSETS = 'dseditor/minimax-h3-studio-assets'
H3_IMAGE_REPO = 'Comfy-Org/MiniMax-H3-Image'
H3_IMAGE_FILES = """ + repr(tuple(image_files)) + """
SHARED_IMAGE_ENCODER = """ + repr(shared_image_encoder) + """
files = [  # (repo, 文件, 放到 models/ 底下的哪里)
    ('OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM', 'Minimax-h3_Singularity_x_10Eros-Max_PRISM_int8.safetensors', 'diffusion_models'),
    ('Comfy-Org/MiniMax-H3', SHARED_IMAGE_ENCODER, 'text_encoders'),
    ('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_video_vae_int8_convrot.safetensors', 'vae'),
    ('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_audio_vae_fp32.safetensors', 'vae'),
    # pika flex Ref2VA（2026-11-04 取代 dmad）：视频与 32×32 TTS 共用这一颗
    ('Comfy-Org/MiniMax-H3', 'loras/pika_h3_flex_ref2va_v1_bf16.safetensors', 'loras'),
] + [(ASSETS, f, str(Path(f).parent)) for f in [
    'loras/minimax_h3_turbo_v4_step600_ema_DasiwaREF2VAHybridV1_0_curveproj1025_compat_v001.safetensors',
    'loras/MysticXXX_MMH3-V2.safetensors',
    'loras/H3_Combat_V2.safetensors',
    'loras/Motion_Repair.safetensors',
    'loras/H3_SPEED_slider.safetensors',
    'semantic_bridge/t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors',
    'latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16_V2.safetensors',
] + (['text_encoders/minimax_h3_rewriter_ref2va_qwen3.5_9b_w4a8.safetensors'] if INCLUDE_REWRITER else [])] + ([
    ('PrunaAI/P-H3-Image-Comfy', 'loras/p-h3-image_lora_bf16.safetensors', 'loras'),
    ('PrunaAI/P-H3-Image-Comfy', 'text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors', 'text_encoders'),
    ('PrunaAI/P-H3-Image-Comfy', 'vae/p-h3-image_vae_int8_convrot.safetensors', 'vae'),
] if INSTALL_IMAGE else []) + [
    (H3_IMAGE_REPO, f, str(Path(f).parent)) for f in H3_IMAGE_FILES if INSTALL_H3_IMAGE
]

# 按清单的实际文件大小检查磁盘和缓存。只有完整文件才会进入模型目录。
requested = defaultdict(list)
for repo, name, _ in files:
    requested[repo].append(name)
api = HfApi()
sizes = {(repo, info.path): info.size for repo, names in requested.items()
         for info in api.get_paths_info(repo_id=repo, paths=names)}
missing = [(repo, name) for repo, name, _ in files if not sizes.get((repo, name))]
if missing:
    raise RuntimeError(f'Hugging Face 找不到以下权重：{missing}')

def complete(path, size):
    return path is not None and path.is_file() and path.stat().st_size == size

remaining = sum(sizes[repo, name] for repo, name, folder in files
                if not complete(COMFY / 'models' / folder / Path(name).name, sizes[repo, name]))
free = shutil.disk_usage('/content').free
if remaining and free < remaining + 3 * 2**30:
    raise RuntimeError(f'仍需下载或复制 {remaining / 2**30:.1f} GiB 权重，可用 {free / 2**30:.1f} GiB。'
                       '请释放空间，或取消 INSTALL_IMAGE / INCLUDE_REWRITER 后重试。')

def copy_model(src, dest):
    partial = dest.with_suffix(dest.suffix + '.part')
    try:
        shutil.copyfile(src, partial)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)

t0 = time.time()
for repo, name, folder in files:
    target = COMFY / 'models' / folder / Path(name).name
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = CACHE / Path(name).name if CACHE else None
    size = sizes[repo, name]
    if not complete(target, size):
        target.unlink(missing_ok=True)
        if complete(cached, size):
            print('从 Google Drive 复制', cached.name)
            copy_model(cached, target)
        else:
            got = Path(hf_hub_download(repo_id=repo, filename=name, local_dir='/content/hf'))
            if not complete(got, size):
                raise RuntimeError(f'权重大小与 Hugging Face 清单不符：{name}')
            got.replace(target)
    if cached and not complete(cached, size):
        print('存一份到 Google Drive', cached.name)
        copy_model(target, cached)
    print(f'{folder:<28} {target.name[:60]:<60} {target.stat().st_size / 2**30:5.1f} GiB')
# 官方提示词写作指南（改写模型照它写）
GUIDES = APP / 'guides'; GUIDES.mkdir(exist_ok=True)
for g in ('VIDEO_PROMPT_WRITING_GUIDE_ref_en.md', 'VIDEO_PROMPT_WRITING_GUIDE_base_en.md'):
    got = Path(hf_hub_download(repo_id='MiniMaxAI/MiniMax-H3', filename='docs/' + g, local_dir='/content/hf'))
    shutil.copy(str(got), str(GUIDES / g))
print(f'✅ 下载完成，{(time.time() - t0) / 60:.1f} 分钟')
"""),
    code("""#@title ③ 启动 ComfyUI 和 H3 Toybox，打开网页
# 运行后下面会出现一个链接；显示成 localhost 也没关系，直接点它就会开新标签页。
# ⚠️ 打开标签页的浏览器必须登录「跟这本 Colab 同一个 Google 账号」；账号不同就改用下一格 ③-b。
#@markdown **作品存到Google Drive**：素材库、文件夹、生成的视频、生成纪录都放在「我的Google Drive/H3Toybox」，
#@markdown 断线或不小心关掉运行阶段，重开后再跑一次 ①②③ 东西都还在。第一次会跳出授权窗口，按同意即可。
SAVE_TO_DRIVE = True  #@param {type:"boolean"}
#@markdown API 密钥也存到Google Drive（下次不用重贴；但Google Drive里会有一份密钥档）
SAVE_API_KEYS = False  #@param {type:"boolean"}
import subprocess, time, urllib.request, json, sys, os, shutil
def wait_http(url, name, log, tries=180):
    for i in range(tries):
        try:
            urllib.request.urlopen(url, timeout=2); print(name, '已启动'); return
        except Exception:
            time.sleep(2)
    print(open(log, encoding='utf-8', errors='ignore').read()[-3000:])
    raise RuntimeError(name + ' 没有启动成功，看上面的纪录')

for proc in ('server', 'toybox'):
    if proc in globals() and globals()[proc].poll() is None:
        globals()[proc].terminate(); globals()[proc].wait()
# 作品放哪里：Google Drive（授权成功）或这台运行阶段的暂存空间
DATA = Path('/content/h3toybox_data')
if SAVE_TO_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        DATA = Path('/content/drive/MyDrive/H3Toybox')
    except Exception as e:
        print('⚠️ 没有挂上Google Drive（', e, '）：这次的作品只存在暂存空间，关掉运行阶段就会消失')
DATA.mkdir(parents=True, exist_ok=True)
# ComfyUI 的输出文件夹直接设在 DATA/output。不能用链接把 output/H3Toybox 接过去：
# ComfyUI 存盘与 /view 都会解开链接检查真实位置，指到 output 外面就报
# 「Saving image outside the output folder is not allowed」。
output_root = DATA / 'output'
videos, out_dir = output_root / 'H3Toybox', COMFY / 'output' / 'H3Toybox'
videos.mkdir(parents=True, exist_ok=True)
# 旧版留下的东西搬过来，以前生成的视频才看得到：DATA/videos、旧链接指向的文件夹、机器上的 output/H3Toybox
old_dirs = [DATA / 'videos']
if out_dir.is_symlink():
    old_dirs.append(out_dir.resolve())
    out_dir.unlink()
elif out_dir.is_dir():
    old_dirs.append(out_dir)
for old in old_dirs:
    if old.is_dir() and old.resolve() != videos.resolve():
        for f in old.iterdir():
            if f.is_file() and not (videos / f.name).exists():
                shutil.move(str(f), str(videos / f.name))
# ComfyUI 只当引擎：只听 127.0.0.1，不对外打开界面
server = subprocess.Popen([sys.executable, 'main.py', '--listen', '127.0.0.1', '--port', '8188',
                           '--output-directory', str(output_root)],
                          cwd=str(COMFY), stdout=open('/content/comfyui.log', 'w'), stderr=subprocess.STDOUT)
wait_http('http://127.0.0.1:8188/system_stats', 'ComfyUI', '/content/comfyui.log')
h3_image_ready = all((COMFY / 'models' / f).is_file() for f in (*H3_IMAGE_FILES, SHARED_IMAGE_ENCODER))
if h3_image_ready:
    with urllib.request.urlopen('http://127.0.0.1:8188/object_info/MiniMaxH3ImageConditioning', timeout=10) as response:
        if 'MiniMaxH3ImageConditioning' not in json.load(response):
            raise RuntimeError('ComfyUI 缺少 MiniMaxH3ImageConditioning，请检查第 ① 格更新结果和 comfyui.log')
print('🎬 视频写到：', videos)
# H3 Toybox 的 Colab 设置（保留用户在网页设置页改过的内容，只覆写这台机器的路径）
(COMFY / 'input').mkdir(exist_ok=True)
cfg_path = DATA / 'config.json'
cfg = json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.exists() else {}
cfg.update({
    'comfy_url': 'http://127.0.0.1:8188',
    'comfy_input_dir': str(COMFY / 'input'),
    'skills_dir': '/content/t8-enhancer/directional_skills',
    'free_after_rewrite': True,   # Colab 内存比较小：改写完就把 9B 模型卸载
})
cfg.pop('guides', None)
# 🖼 图片分页：Colab 用 P-H3-Image（挂在 PRISM 上）；有没有装，看文件是不是真的在
PH3 = ('loras/p-h3-image_lora_bf16.safetensors', 'text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors',
       'vae/p-h3-image_vae_int8_convrot.safetensors')
# 🤖 短片 Agent：Colab 上没有本地大模型，默认改走 Gemini（密钥在网页 ⚙ 设置填）
cfg['agent'] = dict(cfg.get('agent') or {}, provider=(cfg.get('agent') or {}).get('provider', 'gemini'))
cfg['image'] = dict(cfg.get('image') or {}, engine='ph3',
                    installed=all((COMFY / 'models' / f).is_file() for f in PH3))
print('🖼 生图：', '已安装' if cfg['image']['installed'] else '没有安装（② 勾 INSTALL_IMAGE 才有）')
cfg['h3_image'] = {'installed': h3_image_ready}
print('H3-Image：', '权重就绪' if cfg['h3_image']['installed'] else '未安装（在第 ② 格勾选 INSTALL_H3_IMAGE）')
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding='utf-8')
secrets = DATA / 'secrets.json' if SAVE_API_KEYS else Path('/content/h3toybox_secrets.json')
toybox = subprocess.Popen([sys.executable, 'server.py'], cwd=str(APP),
                          env=dict(os.environ, H3TOYBOX_PORT='8190', H3TOYBOX_DATA=str(DATA), H3TOYBOX_SECRETS=str(secrets)),
                          stdout=open('/content/toybox.log', 'w'), stderr=subprocess.STDOUT)
wait_http('http://127.0.0.1:8190/api/config', 'H3 Toybox', '/content/toybox.log', tries=30)
print('📁 作品存放位置：', DATA)
from google.colab import output
output.serve_kernel_port_as_window(8190, anchor_text='🧸 打开 H3 Toybox')
"""),
    code("""#@title ③-b（备用）通过 Cloudflare 打开 H3 Toybox
# ⚠️ 网址没有密码，拿到的人都能用你的 GPU —— 不要贴到公开的地方。
ENABLE_TUNNEL = False  #@param {type:"boolean"}
if not ENABLE_TUNNEL:
    print('Cloudflare 通道未开启。请使用第 ③ 格的 Colab 网页链接。')
else:
""" + "\n".join("    " + line for line in """import subprocess, re, time
if not Path('/content/cloudflared').exists():
    subprocess.run(['wget', '-q', '-O', '/content/cloudflared',
                    'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'], check=True)
    subprocess.run(['chmod', '+x', '/content/cloudflared'], check=True)
# 通道的输出写进文件，不要接 PIPE：PIPE 没人读会塞满，通道就卡住
if 'tun' in globals() and tun.poll() is None:
    tun.terminate(); tun.wait()
open('/content/cloudflared.log', 'w').close()
tun = subprocess.Popen(['/content/cloudflared', 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:8190'],
                       stdout=open('/content/cloudflared.log', 'a'), stderr=subprocess.STDOUT)
url = None
for i in range(60):
    time.sleep(1)
    m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', open('/content/cloudflared.log').read())
    if m:
        url = m.group(0); break
print('👉 H3 Toybox：', url or '（没拿到网址，看 /content/cloudflared.log）')
""".splitlines()) + "\n"),
    code("""#@title 🔍 出问题时跑这格：看 GPU、ComfyUI 与 H3 Toybox 的纪录
import subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
print(subprocess.run(['free', '-g'], capture_output=True, text=True).stdout)
log = open('/content/comfyui.log', encoding='utf-8', errors='ignore').read()
keys = ('error', 'Error', 'Traceback', 'out of memory', 'SelfLift timing] high_resolution total', 'Prompt executed', 'IMPORT FAILED')
print('\\n'.join(l for l in log.splitlines() if any(k in l for k in keys))[-4000:])
print('---- H3 Toybox ----')
print(open('/content/toybox.log', encoding='utf-8', errors='ignore').read()[-2000:])
"""),
]

NOTEBOOKS = {
    'H3_Digital_Human.ipynb': cells,
    'H3_Basic_ComfyUI.ipynb': [HEAD_BASIC, cells[1], C2_BASIC, C3_BASIC, cells[4], cells[5], cells[9]],
    'H3_Toybox.ipynb': TOYBOX,
    'minimax_h3_image_colab.ipynb': TOYBOX,
}
for fname, cs in NOTEBOOKS.items():
    nb = {'cells': cs,
          'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'L4'},
                       'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
          'nbformat': 4, 'nbformat_minor': 0}
    io.open(fname, 'w', encoding='utf-8', newline='\n').write(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
    # 自检：每个 code cell 都要能编译
    for i, c in enumerate(cs):
        if c['cell_type'] == 'code':
            compile(c['source'], f'{fname}:cell{i}', 'exec')
    print('ok', fname, 'cells =', len(cs))

"""2026-11-04: H3 Toybox on Colab gets the 🖼 picture tab through P-H3-Image (PrunaAI): a LoRA on the PRISM model the
notebook already downloads + its own text encoder and image VAE = about 10 GB more (a full Qwen-Image would be
about 20 GB). Optional in cell ② (INSTALL_IMAGE); cell ③ tells the Toybox whether the files are really there."""
from pathlib import Path

p = Path(__file__).resolve().parent / "build_notebooks.py"
s = p.read_text(encoding="utf-8")


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''| ② 下載模型 | 約 50 GB（含 7 GB 的提示詞改寫模型，可以不勾） |''',
    '''| ② 下載模型 | 約 50 GB（含 7 GB 的提示詞改寫模型，可以不勾）；勾「加裝生圖」再多約 10 GB |''')
rep('''- 🧸 **參考生成**：圖片 9 張／影片 3 支／聲音 3 段當參考，提示詞裡用 `@圖片1` 標記''',
    '''- 🖼 **圖片**（選用）：文生圖，或拿最多 8 張圖做編輯（換表情、換衣服、換風格、把人放進場景）。用 [P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image-Comfy) 掛在同一顆 H3 模型上，只多約 10 GB
- 🤖 **短片 Agent**：一句話規劃成多段短片（要先在 ⚙ 設定填 Gemini 金鑰）
- 🧸 **參考生成**：圖片 9 張／影片 3 支／聲音 3 段當參考，提示詞裡用 `@圖片1` 標記''')
rep('''if shutil.disk_usage('/content').free < 60 * 2**30:
    raise RuntimeError('磁碟剩不到 60 GiB，放不下約 50 GB 的模型。請換一台執行階段。')
print('✅ 環境檢查通過，開始安裝')
try:''', '''if shutil.disk_usage('/content').free < 72 * 2**30:
    raise RuntimeError('磁碟剩不到 72 GiB，放不下約 60 GB 的模型。請換一台執行階段。')
print('✅ 環境檢查通過，開始安裝')
try:''')
rep('''INCLUDE_REWRITER = True  #@param {type:"boolean"}
from huggingface_hub import hf_hub_download''', '''INCLUDE_REWRITER = True  #@param {type:"boolean"}
#@markdown 🖼 加裝生圖（多約 10 GB）：網頁的「圖片」分頁可以文生圖、拿最多 8 張圖做編輯。掛在同一顆 H3 模型上。
INSTALL_IMAGE = True  #@param {type:"boolean"}
from huggingface_hub import hf_hub_download''')
rep('''] + (['text_encoders/minimax_h3_rewriter_ref2va_qwen3.5_9b_w4a8.safetensors'] if INCLUDE_REWRITER else [])]''',
    '''] + (['text_encoders/minimax_h3_rewriter_ref2va_qwen3.5_9b_w4a8.safetensors'] if INCLUDE_REWRITER else [])] + ([
    ('PrunaAI/P-H3-Image-Comfy', 'loras/p-h3-image_lora_bf16.safetensors', 'loras'),
    ('PrunaAI/P-H3-Image-Comfy', 'text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors', 'text_encoders'),
    ('PrunaAI/P-H3-Image-Comfy', 'vae/p-h3-image_vae_int8_convrot.safetensors', 'vae'),
] if INSTALL_IMAGE else [])''')
rep('''cfg.pop('guides', None)
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding='utf-8')''',
    '''cfg.pop('guides', None)
# 🖼 圖片分頁：Colab 用 P-H3-Image（掛在 PRISM 上）；有沒有裝，看檔案是不是真的在
PH3 = ('loras/p-h3-image_lora_bf16.safetensors', 'text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors',
       'vae/p-h3-image_vae_int8_convrot.safetensors')
cfg['image'] = dict(cfg.get('image') or {}, engine='ph3',
                    installed=all((COMFY / 'models' / f).is_file() for f in PH3))
print('🖼 生圖：', '已安裝' if cfg['image']['installed'] else '沒有安裝（② 勾 INSTALL_IMAGE 才有）')
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding='utf-8')''')
p.write_text(s, encoding="utf-8")
print("patched")

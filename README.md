# VoCoGround：实时摄像头 + 语音需求 + API 打标项目

本项目实现了一个端到端交互流程：

- 启动时可选输入源：实时摄像头 / 本地上传图片
- 通过按钮开始/停止语音输入用户需求
- 在停止语音输入时，将当前输入图像与文本一起送入模型
- 将语音识别文本 + 图片发送到视觉 API
- 获取目标打标结果并叠加展示在窗口画面上

核心目标是让用户可以一边看实时画面，一边用语音描述要找的目标，然后快速看到打标结果。

---

## 1. 功能概览

### 一体化主流程

主脚本：`realtime_voice_camera_grounding.py`

提供以下交互能力：

- 启动模式：camera（实时摄像头）/ image（上传图片）
- 画面内按钮：开始录音 / 停止录音
- 语音文本确认：先显示语音转文字结果，可手动修改（支持中文输入）后再确认上传
- 停止时推理：
  - camera 模式使用点击停止那一刻的画面
  - image 模式使用启动时指定的图片
- 语音识别：Whisper 将录音转为文本需求
- 视觉定位：Qwen-VL 返回目标框
- 多目标支持：同一需求可返回多个锚框数据
- 实时可视化：将 API 打标结果叠加显示在主窗口

### 其他脚本（模块化能力）

- `camera_capture.py`：独立拍照工具
- `whisper_model.py`：独立语音录制与转写工具
- `qwen_vl_ref_grounding.py`：独立图文指代定位与打框工具

---

## 2. 项目结构

```text
apikey_config.json
camera_capture.py
qwen_vl_ref_grounding.py
realtime_voice_camera_grounding.py
whisper_model.py
photos/
```

---

## 3. 环境要求

- Python 3.9+
- 可用的麦克风设备
- ffmpeg（Whisper 相关依赖）

说明：

- 使用 `camera` 模式需要摄像头
- 使用 `image` 模式不需要摄像头

---

## 4. 安装依赖

```bash
pip install -U opencv-python sounddevice numpy openai-whisper openai Pillow
```

若希望“把照片拖入启动框内”，请额外安装：

```bash
pip install tkinterdnd2
```

说明：

- `opencv-python`：摄像头读取与界面显示
- `sounddevice` + `numpy`：麦克风录音
- `openai-whisper`：语音转文本
- `openai`：调用兼容 OpenAI 的 API（用于 Qwen-VL）
- `Pillow`：图像绘制/打标相关支持

---

## 5. API 配置

默认读取 `apikey_config.json`：

```json
{
  "api_key": "你的 API Key",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "model": "qwen3.5-vl-plus"
}
```

你也可以使用环境变量（默认变量名 `DASHSCOPE_API_KEY`）。

---

## 6. 快速开始

运行一体化脚本：

```bash
python realtime_voice_camera_grounding.py
```

启动时可选输入源：

```bash
# 推荐：直接启动后在弹窗里选 camera/image
python realtime_voice_camera_grounding.py

# 也可命令行固定 camera
python realtime_voice_camera_grounding.py --startup-mode camera

# 仍支持命令行直传图片路径（可选）
python realtime_voice_camera_grounding.py --startup-mode image --input-image photos/demo.jpg
```

可选参数示例：

```bash
python realtime_voice_camera_grounding.py --camera-index 0 --whisper-model turbo --device auto
```

常用参数：

```text
--camera-index      摄像头索引，默认 0
--width             采集宽度，默认 1280
--height            采集高度，默认 720
--sample-rate       音频采样率，默认 16000
--startup-mode      启动输入源：auto/camera/image，默认 auto
--input-image       上传图片路径（image 模式）
--whisper-model     tiny/base/small/medium/large/turbo
--device            auto/cpu/cuda
--source-language   输入语种（如 zh/en），默认 auto
--api-config        API 配置文件路径，默认 apikey_config.json
--model             视觉模型名，默认 qwen3.5-vl-plus
--max-boxes         最多保留的锚框数量，默认 20（0 表示不限）
--result-json       锚框结果输出 JSON，默认 latest_boxes.json（留空不保存）
```

锚框结果 JSON 示例：

```json
{
  "query": "圈出左边穿红衣服的人",
  "count": 3,
  "model": "qwen3.5-vl-plus",
  "boxes": [
    {"bbox": [120, 88, 260, 420], "label": "person_red_left", "score": 0.93},
    {"bbox": [300, 96, 438, 430], "label": "person_red_center", "score": 0.89}
  ]
}
```

---

## 7. 交互流程

### camera 模式（实时摄像头）

1. 启动程序后，看到实时摄像头窗口。
2. 点击“开始录音”。
3. 说出需求，例如：圈出左边穿红衣服的人。
4. 点击“停止录音”。
5. 程序先执行语音转文本，并在屏幕显示 `Editable Text`。
6. 通过 `Input Text` 按钮打开编辑框修改文本。
7. 编辑框支持中文输入。
8. 点击 `Confirm Upload` 确认上传。
9. 程序再执行图片 + 文本定位并返回打标结果。
10. 打标结果叠加显示在实时画面中。
11. 可点击 `Clear All` 一键清空当前输入文本。
12. 可点击 `Change Image` 在运行中切换到新图片（无需重启）。
13. 可点击 `Exit` 按钮直接退出（或按 `q`）。

补充：也可不录音，点击 `Input Text` 输入文本后再点 `Confirm Upload` 上传当前画面；再次语音输入时会追加到现有文本，不会重置。

### image 模式（上传图片 + 语音指代）

1. 启动程序后，在“启动模式”弹窗中选择 `Image`。
2. 将照片直接拖入弹窗中的拖拽框（或点击“浏览”）。
3. 点击“开始”，程序加载图片并显示在窗口中。
4. 点击“开始录音”。
5. 说出需求，例如：圈出图中左侧红色杯子。
6. 点击“停止录音”。
7. 程序先显示语音转文字结果，用户可点击 `Input Text` 修改。
8. 编辑框支持中文输入。
9. 点击 `Confirm Upload` 后，程序才会调用视觉 API。
10. 程序执行：文本 + 图片定位 -> 叠加展示结果。
11. 可点击 `Clear All` 清空输入，重新编辑需求。
12. 可点击 `Change Image` 更换图片并继续交互。
13. 可重复录音多次，点击 `Exit`（或按 `q`）退出程序。

补充：在 image 模式下也支持不录音，直接输入文本后确认上传。

补充：若未安装 `tkinterdnd2`，拖拽框会提示“拖拽未启用”，此时可继续使用“浏览”选择图片。

---

## 8. 调试建议

如果遇到问题，优先检查：

- 麦克风权限是否开放
- 摄像头是否被其他程序占用
- API Key、base_url、model 是否配置正确
- 当前 Python 环境是否安装完整依赖

### 常见环境报错快速修复

1. `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

这是 `torch/openai-whisper` 与 `numpy 2.x` 的兼容问题。请在当前环境执行：

```bash
pip install "numpy<2" --upgrade
```

2. `FileNotFoundError` 出现在 `SSL_CERT_FILE`

这是环境变量 `SSL_CERT_FILE` 指向了不存在的证书文件。可在当前终端先执行：

Windows PowerShell:

```powershell
Remove-Item Env:SSL_CERT_FILE -ErrorAction SilentlyContinue
python realtime_voice_camera_grounding.py
```

脚本内部也已增加自动兜底：若检测到无效证书路径，会自动清理并尝试使用 `certifi` 证书。

可先分别运行模块脚本定位问题：

- `python camera_capture.py`
- `python whisper_model.py`
- `python qwen_vl_ref_grounding.py --image photos/xxx.jpg --query "圈出目标"`

---

## 9. 说明

本项目用于学习与演示“多模态交互闭环”：语音输入需求 + 实时视觉理解 + 结果可视化。
可继续扩展的方向包括：

- 连续跟踪同类目标
- 多目标类别筛选
- 语音唤醒词与自动开始/停止
- UI 主题化与状态面板增强

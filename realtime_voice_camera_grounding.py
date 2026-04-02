import argparse
import base64
import importlib
import json
import os
import re
import tempfile
import threading
import time
import warnings
import wave


warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)


class RealtimeVoiceGroundingApp:
    def __init__(self, args):
        self.args = args

        self.cv2 = self._load_module(
            "cv2",
            "缺少依赖 opencv-python，请先安装：pip install opencv-python",
        )
        self.sd = self._load_module(
            "sounddevice",
            "缺少依赖 sounddevice，请先安装：pip install sounddevice",
        )
        self.np = self._load_module(
            "numpy",
            "缺少依赖 numpy，请先安装：pip install numpy",
        )
        self.whisper = self._load_whisper_module()

        openai_mod = self._load_module(
            "openai",
            "缺少依赖 openai，请先安装：pip install openai",
        )
        client_cls = getattr(openai_mod, "OpenAI", None)
        if client_cls is None:
            raise RuntimeError("当前 openai 包版本不支持 OpenAI 客户端，请升级：pip install -U openai")

        api_conf = self._load_api_config(args.api_config)
        api_key = str(api_conf.get("api_key", "")).strip() or os.getenv(args.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"未找到 API Key，请在 {args.api_config} 中配置 api_key，或设置环境变量 {args.api_key_env}"
            )

        self.base_url = str(api_conf.get("base_url", "")).strip() or args.base_url
        self.model_name = str(api_conf.get("model", "")).strip() or args.model
        self.fallback_models = [
            "qwen3-vl-plus-2025-12-19",
            "qwen3-vl-plus",
            "qwen3-vl-flash-2026-01-22",
            "qwen3-vl-flash",
        ]
        self.active_model_name = self.model_name
        self._prepare_ssl_environment()
        self.client = client_cls(api_key=api_key, base_url=self.base_url)

        self.device = self._resolve_whisper_device(args.device)
        self.whisper_model = self.whisper.load_model(args.whisper_model, device=self.device)
        self._init_unicode_text_renderer()
        self._maybe_prompt_startup_source()

        self.window_name = "Realtime Voice Grounding"
        self.window_width = max(960, int(args.window_width))
        self.window_height = max(640, int(args.window_height))
        self.panel_width = max(280, int(args.panel_width))
        self.panel_height = max(180, int(args.panel_height))
        self.top_bar_height = 90
        self.status_bar_height = 180
        self.content_padding = 16
        self.panel_gap = 20
        self.input_mode = self._resolve_input_mode()
        self.upload_image_path = None
        self.static_image = None
        self.cap = None
        self.audio_stream = None

        self.current_frame = None
        self.overlay_image = None
        self.last_boxes = []
        self.last_query = ""
        self.pending_query = ""
        self.pending_snapshot = None
        self.status_text = "Ready: click Start"
        self.last_error = ""

        self.recording = False
        self.processing = False
        self.audio_chunks = []
        self.audio_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.btn_start = (0, 0, 0, 0)
        self.btn_stop = (0, 0, 0, 0)
        self.btn_confirm = (0, 0, 0, 0)
        self.btn_edit = (0, 0, 0, 0)
        self.btn_clear = (0, 0, 0, 0)
        self.btn_change_image = (0, 0, 0, 0)
        self.btn_exit = (0, 0, 0, 0)
        self.left_panel_rect = (0, 0, 0, 0)
        self.right_panel_rect = (0, 0, 0, 0)
        self._refresh_layout()
        self.should_exit = False

        if self.input_mode == "image":
            self.upload_image_path = self._normalize_cli_path(self.args.input_image)
            if not self.upload_image_path:
                raise RuntimeError("已选择图片模式，但未提供图片路径，请使用 --input-image 传入")
            self.static_image = self._load_input_image(self.upload_image_path)
            self.current_frame = self.static_image.copy()
            self.status_text = "Image mode ready: click Start"

    def _load_module(self, module_name, err_msg):
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(err_msg) from exc

    def _load_whisper_module(self):
        try:
            return importlib.import_module("whisper")
        except ImportError as exc:
            raise RuntimeError("缺少依赖 openai-whisper，请先安装：pip install openai-whisper") from exc
        except Exception as exc:
            msg = str(exc)
            if "compiled using NumPy 1.x" in msg or "_ARRAY_API not found" in msg:
                raise RuntimeError(
                    "检测到 numpy 与 torch/whisper 二进制不兼容。"
                    "请在当前环境执行：pip install \"numpy<2\" --upgrade"
                ) from exc
            raise RuntimeError(f"加载 whisper/torch 失败：{msg}") from exc

    def _resolve_input_mode(self):
        if self.args.startup_mode == "auto":
            return "image" if str(self.args.input_image).strip() else "camera"
        return self.args.startup_mode

    def _maybe_prompt_startup_source(self):
        normalized = self._normalize_cli_path(self.args.input_image)
        if normalized:
            self.args.input_image = normalized
            return

        if self.args.startup_mode == "camera":
            return

        choice = self._show_startup_selector()
        self.args.startup_mode = choice["mode"]
        self.args.input_image = choice["image_path"]

    def _show_startup_selector(self):
        tk = self._load_module(
            "tkinter",
            "当前环境缺少 tkinter，无法弹出启动选择窗口。请安装 tkinter 或使用 --startup-mode camera",
        )
        filedialog = importlib.import_module("tkinter.filedialog")
        messagebox = importlib.import_module("tkinter.messagebox")

        tkdnd_mod = None
        dnd_supported = False
        try:
            tkdnd_mod = importlib.import_module("tkinterdnd2")
            dnd_supported = True
        except ImportError:
            dnd_supported = False

        if dnd_supported:
            root = tkdnd_mod.TkinterDnD.Tk()
        else:
            root = tk.Tk()

        root.title("选择输入源")
        root.geometry("560x330")
        root.resizable(False, False)

        mode_default = "image" if self.args.startup_mode in ["auto", "image"] else "camera"
        mode_var = tk.StringVar(value=mode_default)
        path_var = tk.StringVar(value=self._normalize_cli_path(self.args.input_image))
        result = {"mode": "camera", "image_path": ""}
        canceled = {"value": True}

        title_label = tk.Label(root, text="启动模式", font=("Microsoft YaHei", 13, "bold"))
        title_label.pack(pady=(16, 8))

        mode_frame = tk.Frame(root)
        mode_frame.pack(pady=4)

        tk.Radiobutton(mode_frame, text="Camera 实时摄像头", variable=mode_var, value="camera").pack(side=tk.LEFT, padx=12)
        tk.Radiobutton(mode_frame, text="Image 上传图片", variable=mode_var, value="image").pack(side=tk.LEFT, padx=12)

        path_frame = tk.Frame(root)
        path_frame.pack(fill=tk.X, padx=20, pady=(10, 6))

        tk.Label(path_frame, text="图片路径:").pack(side=tk.LEFT)
        path_entry = tk.Entry(path_frame, textvariable=path_var)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))

        def on_browse():
            file_path = filedialog.askopenfilename(
                title="选择图片",
                filetypes=[
                    ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                    ("All Files", "*.*"),
                ],
            )
            if file_path:
                path_var.set(file_path)
                mode_var.set("image")

        tk.Button(path_frame, text="浏览", width=8, command=on_browse).pack(side=tk.LEFT)

        drop_text = "将图片拖入下方框内"
        if not dnd_supported:
            drop_text = "拖拽未启用（请安装 tkinterdnd2），可点击浏览选择图片"

        drop_box = tk.Label(
            root,
            text=drop_text,
            relief="groove",
            borderwidth=2,
            width=64,
            height=8,
            bg="#f7f7f7",
            fg="#333333",
        )
        drop_box.pack(padx=20, pady=(6, 10), fill=tk.BOTH, expand=True)

        if dnd_supported:
            def on_drop(event):
                paths = self._split_dropped_paths(event.data)
                if paths:
                    path_var.set(paths[0])
                    mode_var.set("image")

            drop_box.drop_target_register(tkdnd_mod.DND_FILES)
            drop_box.dnd_bind("<<Drop>>", on_drop)

        def on_confirm():
            selected_mode = mode_var.get()
            image_path = self._normalize_cli_path(path_var.get())

            if selected_mode == "image":
                if not image_path:
                    messagebox.showerror("缺少图片", "请选择或拖入图片后再开始")
                    return
                if not os.path.isfile(image_path):
                    messagebox.showerror("路径无效", f"图片文件不存在:\n{image_path}")
                    return

            result["mode"] = selected_mode
            result["image_path"] = image_path if selected_mode == "image" else ""
            canceled["value"] = False
            root.destroy()

        def on_cancel():
            root.destroy()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=(0, 16))
        tk.Button(btn_frame, text="开始", width=12, command=on_confirm).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_frame, text="取消", width=12, command=on_cancel).pack(side=tk.LEFT, padx=8)

        root.protocol("WM_DELETE_WINDOW", on_cancel)
        root.mainloop()

        if canceled["value"]:
            if self.args.startup_mode == "image":
                raise RuntimeError("已取消图片选择，程序退出")
            return {"mode": "camera", "image_path": ""}

        return result

    def _split_dropped_paths(self, raw):
        text = str(raw or "").strip()
        if not text:
            return []

        paths = []
        for match in re.finditer(r"\{([^}]*)\}|(\S+)", text):
            item = match.group(1) or match.group(2)
            p = self._normalize_cli_path(item)
            if p:
                paths.append(p)
        return paths

    def _normalize_cli_path(self, path):
        text = str(path or "").strip().strip('"').strip("'")
        if not text:
            return ""
        text = text.replace("\\", os.sep).replace("/", os.sep)
        return os.path.abspath(os.path.expanduser(text))

    def _load_input_image(self, image_path):
        if not os.path.isfile(image_path):
            raise RuntimeError(f"图片文件不存在: {image_path}")

        try:
            raw = self.np.fromfile(image_path, dtype=self.np.uint8)
            image = self.cv2.imdecode(raw, self.cv2.IMREAD_COLOR)
        except Exception as exc:
            raise RuntimeError(f"读取图片失败: {image_path}") from exc

        if image is None:
            raise RuntimeError(f"无法解码图片: {image_path}")

        if self.args.width > 0 and self.args.height > 0:
            image = self.cv2.resize(image, (self.args.width, self.args.height))
        return image

    def _refresh_layout(self):
        self._layout_buttons()
        self._layout_panels()

    def _layout_buttons(self):
        labels = [
            ("btn_start", "Start", 140),
            ("btn_stop", "Stop", 140),
            ("btn_confirm", "Confirm Upload", 200),
            ("btn_edit", "Input Text", 160),
            ("btn_clear", "Clear All", 150),
            ("btn_change_image", "Change Image", 170),
            ("btn_exit", "Exit", 110),
        ]

        margin_x = 18
        gap = 12
        y1 = 20
        y2 = 70

        base_total = sum(item[2] for item in labels)
        available = max(400, self.window_width - margin_x * 2 - gap * (len(labels) - 1))
        scale = min(1.0, available / float(base_total))

        widths = [max(88, int(item[2] * scale)) for item in labels]
        total_width = sum(widths) + gap * (len(labels) - 1)
        start_x = max(margin_x, (self.window_width - total_width) // 2)

        x = start_x
        for idx, (attr_name, _, _) in enumerate(labels):
            w = widths[idx]
            setattr(self, attr_name, (x, y1, x + w, y2))
            x += w + gap

    def _layout_panels(self):
        content_top = self.top_bar_height + self.content_padding
        content_bottom = self.window_height - self.status_bar_height - self.content_padding
        available_h = max(120, content_bottom - content_top)
        panel_h = min(self.panel_height, available_h)

        max_panel_w = max(180, (self.window_width - self.content_padding * 2 - self.panel_gap) // 2)
        panel_w = min(self.panel_width, max_panel_w)

        total_w = panel_w * 2 + self.panel_gap
        start_x = max(self.content_padding, (self.window_width - total_w) // 2)
        y1 = content_top + max(0, (available_h - panel_h) // 2)
        y2 = y1 + panel_h

        self.left_panel_rect = (start_x, y1, start_x + panel_w, y2)
        right_x1 = start_x + panel_w + self.panel_gap
        self.right_panel_rect = (right_x1, y1, right_x1 + panel_w, y2)

    def _resize_keep_aspect(self, image, target_w, target_h):
        if image is None or target_w <= 1 or target_h <= 1:
            return None

        src_h, src_w = image.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return None

        scale = min(target_w / float(src_w), target_h / float(src_h))
        out_w = max(1, int(src_w * scale))
        out_h = max(1, int(src_h * scale))
        return self.cv2.resize(image, (out_w, out_h))

    def _draw_image_panel(self, canvas, rect, title, image):
        x1, y1, x2, y2 = rect
        self.cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 220, 220), 2)
        self._draw_text(canvas, title, (x1 + 10, max(20, y1 - 10)), (240, 240, 240), font_scale=0.72, thickness=2)

        inner_pad = 10
        ix1 = x1 + inner_pad
        iy1 = y1 + inner_pad
        ix2 = x2 - inner_pad
        iy2 = y2 - inner_pad
        self.cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), (40, 40, 40), -1)

        if image is None:
            self._draw_text(canvas, "No Image", (ix1 + 18, iy1 + 34), (180, 180, 180), font_scale=0.7, thickness=2)
            return

        rendered = self._resize_keep_aspect(image, ix2 - ix1, iy2 - iy1)
        if rendered is None:
            self._draw_text(canvas, "Invalid Image", (ix1 + 18, iy1 + 34), (180, 180, 180), font_scale=0.7, thickness=2)
            return

        rh, rw = rendered.shape[:2]
        ox = ix1 + ((ix2 - ix1) - rw) // 2
        oy = iy1 + ((iy2 - iy1) - rh) // 2
        canvas[oy:oy + rh, ox:ox + rw] = rendered

    def _prepare_ssl_environment(self):
        cert_path = os.getenv("SSL_CERT_FILE", "").strip()
        if cert_path and (not os.path.isfile(cert_path)):
            # 某些环境会残留失效证书路径，httpx 初始化时会直接报 FileNotFoundError。
            os.environ.pop("SSL_CERT_FILE", None)

        current = os.getenv("SSL_CERT_FILE", "").strip()
        if current:
            return

        try:
            certifi = importlib.import_module("certifi")
            where = getattr(certifi, "where", None)
            if callable(where):
                certifi_path = where()
                if certifi_path and os.path.isfile(certifi_path):
                    os.environ["SSL_CERT_FILE"] = certifi_path
        except ImportError:
            pass

    def _resolve_whisper_device(self, device_arg):
        if device_arg in ["cpu", "cuda"]:
            return device_arg

        try:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"

    def _init_unicode_text_renderer(self):
        self.pil_image_mod = None
        self.pil_draw_mod = None
        self.pil_font_mod = None
        self.unicode_font_path = None

        try:
            self.pil_image_mod = importlib.import_module("PIL.Image")
            self.pil_draw_mod = importlib.import_module("PIL.ImageDraw")
            self.pil_font_mod = importlib.import_module("PIL.ImageFont")
        except ImportError:
            return

        font_candidates = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]
        for path in font_candidates:
            if os.path.isfile(path):
                self.unicode_font_path = path
                break

    def _contains_non_ascii(self, text):
        return any(ord(ch) > 127 for ch in str(text))

    def _draw_text(self, frame_bgr, text, org, color, font_scale=0.6, thickness=2):
        if not text:
            return

        # OpenCV putText 不支持中文；仅在需要时走 Pillow 绘制。
        if (not self._contains_non_ascii(text)) or self.pil_image_mod is None or self.pil_font_mod is None:
            self.cv2.putText(
                frame_bgr,
                str(text),
                org,
                self.cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                self.cv2.LINE_AA,
            )
            return

        rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
        pil_image = self.pil_image_mod.fromarray(rgb)
        draw = self.pil_draw_mod.Draw(pil_image)

        size = max(14, int(30 * font_scale))
        try:
            if self.unicode_font_path:
                font = self.pil_font_mod.truetype(self.unicode_font_path, size=size)
            else:
                font = self.pil_font_mod.load_default()
        except Exception:
            font = self.pil_font_mod.load_default()

        draw.text(org, str(text), fill=(int(color[2]), int(color[1]), int(color[0])), font=font)
        frame_bgr[:, :] = self.cv2.cvtColor(self.np.array(pil_image), self.cv2.COLOR_RGB2BGR)

    def _load_api_config(self, path):
        if not path or not os.path.isfile(path):
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"API 配置文件不是合法 JSON: {path}") from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"API 配置文件内容必须是对象: {path}")

        return data

    def _extract_first_json(self, text):
        text = text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\\n", "", text)
            text = re.sub(r"\\n```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("模型返回中未找到 JSON")

        return json.loads(match.group(0))

    def _build_prompt(self, query):
        return (
            "你是视觉定位助手。请根据用户描述在图中找目标并返回 JSON。"
            "当图中有多个符合条件的目标时，必须返回所有目标，"
            "每个目标单独一个框，不要只返回一个。"
            "输出必须是纯 JSON，不要 markdown，不要解释。"
            "JSON 格式: "
            '{"boxes":[{"label":"目标名","bbox":[x1,y1,x2,y2],"score":0.0}]}'
            "。bbox 优先返回 0~1000 归一化坐标；若返回像素坐标也可。"
            "如果找不到目标，返回 {\"boxes\":[]}。"
            f"用户描述: {query}"
        )

    def _extract_boxes_payload(self, parsed):
        if isinstance(parsed, dict):
            if isinstance(parsed.get("boxes"), list):
                return parsed.get("boxes", [])

            bbox = parsed.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                return [
                    {
                        "label": parsed.get("label") or "target_1",
                        "bbox": bbox,
                        "score": parsed.get("score"),
                    }
                ]

        if isinstance(parsed, list):
            return parsed

        return []

    def _image_to_data_url(self, frame_bgr):
        ok, encoded = self.cv2.imencode(".jpg", frame_bgr)
        if not ok:
            raise RuntimeError("图片编码失败")
        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _infer_boxes(self, frame_bgr, query):
        image_data_url = self._image_to_data_url(frame_bgr)
        prompt = self._build_prompt(query)

        response = self._call_vl_with_fallback(prompt, image_data_url)

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("模型返回为空")

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
            content = "\n".join(text_parts)

        parsed = self._extract_first_json(str(content))
        return self._normalize_boxes(parsed, frame_bgr.shape[1], frame_bgr.shape[0])

    def _call_vl_with_fallback(self, prompt, image_data_url):
        model_candidates = [self.model_name]
        for item in self.fallback_models:
            if item not in model_candidates:
                model_candidates.append(item)

        last_exc = None
        for model in model_candidates:
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_data_url}},
                            ],
                        }
                    ],
                    temperature=0,
                )
                self.active_model_name = model
                return resp
            except Exception as exc:
                msg = str(exc)
                if "model_not_found" in msg or "does not exist" in msg:
                    last_exc = exc
                    continue
                raise

        if last_exc is not None:
            raise RuntimeError(
                "视觉模型不可用。请将 apikey_config.json 的 model 设置为可访问模型，"
                "例如 qwen3-vl-plus-2025-12-19。"
            ) from last_exc

        raise RuntimeError("视觉模型调用失败")

    def _normalize_boxes(self, parsed, width, height):
        boxes = self._extract_boxes_payload(parsed)
        result = []

        for i, item in enumerate(boxes, start=1):
            if not isinstance(item, dict):
                continue

            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except (TypeError, ValueError):
                continue

            if max(x1, y1, x2, y2) <= 1000.0 and min(x1, y1, x2, y2) >= 0.0:
                x1 = x1 / 1000.0 * width
                x2 = x2 / 1000.0 * width
                y1 = y1 / 1000.0 * height
                y2 = y2 / 1000.0 * height

            left = int(max(0, min(width - 1, min(x1, x2))))
            top = int(max(0, min(height - 1, min(y1, y2))))
            right = int(max(0, min(width - 1, max(x1, x2))))
            bottom = int(max(0, min(height - 1, max(y1, y2))))

            if right <= left or bottom <= top:
                continue

            result.append(
                {
                    "bbox": [left, top, right, bottom],
                    "label": str(item.get("label") or f"target_{i}"),
                    "score": item.get("score"),
                }
            )

        max_boxes = max(0, int(self.args.max_boxes))
        if max_boxes > 0:
            result = result[:max_boxes]

        return result

    def _save_boxes_json(self, query, boxes):
        output_path = str(self.args.result_json or "").strip()
        if not output_path:
            return

        payload = {
            "query": query,
            "count": len(boxes),
            "model": self.active_model_name,
            "boxes": boxes,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _draw_boxes(self, frame_bgr, boxes):
        image = frame_bgr.copy()
        for item in boxes:
            left, top, right, bottom = item["bbox"]
            label = item["label"]
            score = item.get("score")
            caption = label
            if isinstance(score, (int, float)):
                caption = f"{label} ({score:.2f})"

            self.cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            text_size, _ = self.cv2.getTextSize(caption, self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tw, th = text_size
            bg_x1 = left
            bg_y1 = max(0, top - th - 8)
            bg_x2 = left + tw + 10
            bg_y2 = top
            self.cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), -1)
            self._draw_text(image, caption, (left + 5, top - 5), (255, 255, 255), font_scale=0.6, thickness=2)

        return image

    def _save_wav(self, path, audio, sample_rate=16000):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

    def _transcribe_audio(self, audio):
        if audio is None or audio.size == 0:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            self._save_wav(wav_path, audio, sample_rate=self.args.sample_rate)
            source_lang = None if self.args.source_language == "auto" else self.args.source_language
            is_cuda = str(getattr(self.whisper_model, "device", "cpu")).startswith("cuda")
            result = self.whisper_model.transcribe(
                wav_path,
                task="transcribe",
                language=source_lang,
                fp16=is_cuda,
            )
            return str(result.get("text", "")).strip()
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            with self.state_lock:
                self.last_error = f"Audio input status: {status}"

        with self.state_lock:
            should_record = self.recording

        if not should_record:
            return

        chunk = indata.copy()
        with self.audio_lock:
            self.audio_chunks.append(chunk)

    def _start_audio_stream(self):
        self.audio_stream = self.sd.InputStream(
            samplerate=self.args.sample_rate,
            channels=1,
            dtype="int16",
            callback=self._audio_callback,
            blocksize=0,
        )
        self.audio_stream.start()

    def _open_camera(self):
        self.cap = self.cv2.VideoCapture(self.args.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头，索引={self.args.camera_index}")

        self.cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        self.cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)

    def _open_input_source(self):
        if self.input_mode == "camera":
            self._open_camera()
            return

        if self.static_image is None:
            raise RuntimeError("图片模式初始化失败：未加载图片")

    def _on_mouse(self, event, x, y, flags, param):
        if event != self.cv2.EVENT_LBUTTONDOWN:
            return
        try:
            if self._inside(x, y, self.btn_start):
                self._start_recording()
            elif self._inside(x, y, self.btn_stop):
                self._stop_recording_and_process()
            elif self._inside(x, y, self.btn_confirm):
                self._confirm_upload()
            elif self._inside(x, y, self.btn_edit):
                self._edit_pending_text()
            elif self._inside(x, y, self.btn_clear):
                self._clear_all_input()
            elif self._inside(x, y, self.btn_change_image):
                self._change_image()
            elif self._inside(x, y, self.btn_exit):
                self._request_exit()
        except Exception as exc:
            with self.state_lock:
                self.last_error = str(exc)
                self.status_text = "Button action failed"

    def _inside(self, x, y, rect):
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _start_recording(self):
        with self.state_lock:
            if self.processing:
                self.status_text = "Processing, please wait"
                return
            if self.recording:
                self.status_text = "Already recording"
                return

        with self.audio_lock:
            self.audio_chunks = []

        with self.state_lock:
            self.recording = True
            self.status_text = "Recording... click Stop"
            self.last_error = ""
            self.last_query = ""

    def _stop_recording_and_process(self):
        with self.state_lock:
            if self.processing:
                self.status_text = "Processing, please wait"
                return
            if not self.recording:
                self.status_text = "Click Start first"
                return

            self.recording = False
            self.processing = True
            self.status_text = "Transcribing speech..."

            snapshot = None if self.current_frame is None else self.current_frame.copy()

        with self.audio_lock:
            chunks = list(self.audio_chunks)
            self.audio_chunks = []

        if snapshot is None:
            with self.state_lock:
                self.processing = False
                self.status_text = "No frame captured, retry"
            return

        audio = None
        if chunks:
            audio = self.np.concatenate(chunks, axis=0)

        worker = threading.Thread(
            target=self._transcribe_request,
            args=(snapshot, audio),
            daemon=True,
        )
        worker.start()

    def _transcribe_request(self, snapshot, audio):
        try:
            query = self._transcribe_audio(audio)
            if not query:
                with self.state_lock:
                    self.status_text = "No speech recognized, retry"
                return

            with self.state_lock:
                self.pending_snapshot = snapshot
                existing = str(self.pending_query or "").strip()
                if existing:
                    self.pending_query = f"{existing} {query}".strip()
                else:
                    self.pending_query = query
                self.status_text = "Review text, edit if needed, then Confirm Upload"
                self.last_error = ""
        except Exception as exc:
            with self.state_lock:
                self.status_text = "Transcribe failed, see error"
                self.last_error = str(exc)
        finally:
            with self.state_lock:
                self.processing = False

    def _confirm_upload(self):
        with self.state_lock:
            if self.processing:
                self.status_text = "Processing, please wait"
                return
            if self.recording:
                self.status_text = "Stop recording first"
                return
            query = str(self.pending_query).strip()
            if not query:
                self.status_text = "Text is empty, type query first"
                return

            if self.pending_snapshot is not None:
                snapshot = self.pending_snapshot.copy()
            else:
                if self.current_frame is None:
                    self.status_text = "No frame captured, retry"
                    return
                snapshot = self.current_frame.copy()

            self.processing = True
            self.status_text = "Uploading text + image to vision API..."

        worker = threading.Thread(
            target=self._process_confirmed_request,
            args=(snapshot, query),
            daemon=True,
        )
        worker.start()

    def _edit_pending_text(self):
        with self.state_lock:
            if self.processing:
                self.status_text = "Processing, please wait"
                return
            if self.recording:
                self.status_text = "Stop recording first"
                return
            current_text = str(self.pending_query)

        edited = self._show_text_editor_dialog(current_text)
        if edited is None:
            return

        with self.state_lock:
            self.pending_query = edited
            if edited.strip() and self.pending_snapshot is None and self.current_frame is not None:
                self.pending_snapshot = self.current_frame.copy()
            if edited.strip():
                self.status_text = "Text updated, click Confirm Upload"
            else:
                self.status_text = "Text is empty, type query first"

    def _show_text_editor_dialog(self, initial_text):
        tk = self._load_module(
            "tkinter",
            "当前环境缺少 tkinter，无法打开中文编辑框",
        )

        root = tk.Tk()
        root.title("编辑文本（支持中文输入）")
        root.geometry("680x260")
        root.resizable(False, False)

        tk.Label(root, text="请确认或修改语音识别文本：", font=("Microsoft YaHei", 11)).pack(anchor="w", padx=12, pady=(10, 6))

        text_box = tk.Text(root, height=8, wrap="word", font=("Microsoft YaHei", 11))
        text_box.pack(fill="both", expand=True, padx=12)
        text_box.insert("1.0", initial_text or "")
        text_box.focus_set()

        result = {"value": None}

        def on_save():
            result["value"] = text_box.get("1.0", "end-1c")
            root.destroy()

        def on_cancel():
            root.destroy()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="确认", width=10, command=on_save).pack(side="left", padx=8)
        tk.Button(btn_frame, text="取消", width=10, command=on_cancel).pack(side="left", padx=8)

        root.protocol("WM_DELETE_WINDOW", on_cancel)
        root.mainloop()
        return result["value"]

    def _clear_all_input(self):
        with self.state_lock:
            if self.processing:
                self.status_text = "Processing, please wait"
                return
            if self.recording:
                self.status_text = "Stop recording first"
                return

            self.pending_query = ""
            self.pending_snapshot = None
            self.status_text = "All input cleared"

    def _pick_image_path(self):
        tk = self._load_module(
            "tkinter",
            "当前环境缺少 tkinter，无法选择图片",
        )
        filedialog = importlib.import_module("tkinter.filedialog")

        # 显式创建并销毁 root，避免某些环境下文件对话框卡死或无响应。
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        try:
            file_path = filedialog.askopenfilename(
                parent=root,
                title="选择新图片",
                filetypes=[
                    ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                    ("All Files", "*.*"),
                ],
            )
        finally:
            root.destroy()

        return self._normalize_cli_path(file_path)

    def _change_image(self):
        with self.state_lock:
            if self.processing:
                self.status_text = "Processing, please wait"
                return
            if self.recording:
                self.status_text = "Stop recording first"
                return

        try:
            image_path = self._pick_image_path()
            if not image_path:
                with self.state_lock:
                    self.status_text = "Image unchanged"
                return

            image = self._load_input_image(image_path)
        except Exception as exc:
            with self.state_lock:
                self.status_text = "Change image failed"
                self.last_error = str(exc)
            return

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        with self.state_lock:
            self.input_mode = "image"
            self.upload_image_path = image_path
            self.static_image = image
            self.current_frame = image.copy()
            self.overlay_image = None
            if str(self.pending_query).strip():
                self.pending_snapshot = image.copy()
            else:
                self.pending_snapshot = None
            self.last_error = ""
            self.status_text = "Image updated, ready to upload"

    def _request_exit(self):
        with self.state_lock:
            if self.processing:
                self.status_text = "Finishing current task, please wait"
                return
            self.should_exit = True
            self.status_text = "Exiting..."

    def _process_confirmed_request(self, snapshot, query):
        try:
            boxes = self._infer_boxes(snapshot, query)
            marked = self._draw_boxes(snapshot, boxes)
            self._save_boxes_json(query, boxes)

            with self.state_lock:
                self.overlay_image = marked
                self.last_boxes = boxes
                self.last_query = query
                self.pending_query = ""
                self.pending_snapshot = None
                self.status_text = f"Done: {len(boxes)} targets ({self.active_model_name})"
                self.last_error = ""
        except Exception as exc:
            with self.state_lock:
                self.status_text = "Failed, see error"
                self.last_error = str(exc)
        finally:
            with self.state_lock:
                self.processing = False

    def _handle_text_edit_key(self, key):
        # 禁用直接键盘编辑，统一通过按钮弹窗输入文本。
        return "none"

    def _draw_buttons(self, frame):
        with self.state_lock:
            is_recording = self.recording
            is_processing = self.processing
            has_text = bool(str(self.pending_query).strip())
            can_confirm = has_text and (not is_recording) and (not is_processing)
            can_edit = (not is_recording) and (not is_processing)
            can_clear = can_edit and bool(str(self.pending_query).strip())
            can_change_image = (not is_recording) and (not is_processing)
            can_exit = not is_processing

        start_color = (40, 160, 40)
        stop_color = (40, 40, 200)
        confirm_color = (0, 140, 255)
        edit_color = (190, 120, 30)
        clear_color = (70, 70, 180)
        change_image_color = (30, 140, 170)
        exit_color = (30, 30, 220)

        if is_recording or is_processing:
            start_color = (90, 90, 90)
        if (not is_recording) or is_processing:
            stop_color = (90, 90, 90)
        if not can_confirm:
            confirm_color = (90, 90, 90)
        if not can_edit:
            edit_color = (90, 90, 90)
        if not can_clear:
            clear_color = (90, 90, 90)
        if not can_change_image:
            change_image_color = (90, 90, 90)
        if not can_exit:
            exit_color = (90, 90, 90)

        self.cv2.rectangle(frame, (self.btn_start[0], self.btn_start[1]), (self.btn_start[2], self.btn_start[3]), start_color, -1)
        self.cv2.rectangle(frame, (self.btn_stop[0], self.btn_stop[1]), (self.btn_stop[2], self.btn_stop[3]), stop_color, -1)
        self.cv2.rectangle(frame, (self.btn_confirm[0], self.btn_confirm[1]), (self.btn_confirm[2], self.btn_confirm[3]), confirm_color, -1)
        self.cv2.rectangle(frame, (self.btn_edit[0], self.btn_edit[1]), (self.btn_edit[2], self.btn_edit[3]), edit_color, -1)
        self.cv2.rectangle(frame, (self.btn_clear[0], self.btn_clear[1]), (self.btn_clear[2], self.btn_clear[3]), clear_color, -1)
        self.cv2.rectangle(
            frame,
            (self.btn_change_image[0], self.btn_change_image[1]),
            (self.btn_change_image[2], self.btn_change_image[3]),
            change_image_color,
            -1,
        )
        self.cv2.rectangle(frame, (self.btn_exit[0], self.btn_exit[1]), (self.btn_exit[2], self.btn_exit[3]), exit_color, -1)

        labels = [
            (self.btn_start, "Start", 0.72),
            (self.btn_stop, "Stop", 0.72),
            (self.btn_confirm, "Confirm Upload", 0.66),
            (self.btn_edit, "Input Text", 0.66),
            (self.btn_clear, "Clear All", 0.66),
            (self.btn_change_image, "Change Image", 0.62),
            (self.btn_exit, "Exit", 0.68),
        ]
        for rect, label, scale in labels:
            x1, y1, x2, y2 = rect
            text_size, _ = self.cv2.getTextSize(label, self.cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            tw, th = text_size
            tx = x1 + max(8, ((x2 - x1) - tw) // 2)
            ty = y1 + ((y2 - y1) + th) // 2
            self._draw_text(frame, label, (tx, ty), (255, 255, 255), font_scale=scale, thickness=2)

    def _draw_status(self, frame):
        with self.state_lock:
            status = self.status_text
            query = self.last_query
            box_count = len(self.last_boxes)
            pending_query = self.pending_query
            err = self.last_error
            is_recording = self.recording
            overlay = None if self.overlay_image is None else self.overlay_image.copy()

        h, w = frame.shape[:2]
        status_top = h - self.status_bar_height
        self.cv2.rectangle(frame, (10, status_top), (w - 10, h - 10), (0, 0, 0), -1)
        self._draw_text(frame, f"Status: {status}", (20, status_top + 35), (255, 255, 255), font_scale=0.65, thickness=2)

        if query:
            show_query = query if len(query) <= 55 else (query[:52] + "...")
            self._draw_text(frame, f"Last Query: {show_query}", (20, status_top + 70), (255, 255, 0), font_scale=0.62, thickness=2)
            self._draw_text(frame, f"Boxes: {box_count}", (20, status_top + 88), (255, 220, 120), font_scale=0.58, thickness=2)

        if pending_query:
            show_pending = pending_query if len(pending_query) <= 70 else (pending_query[:67] + "...")
            self._draw_text(frame, f"Editable Text: {show_pending}_", (20, status_top + 105), (160, 255, 160), font_scale=0.62, thickness=2)
            self._draw_text(
                frame,
                "Use Input Text button to edit (Chinese supported) | Click Confirm to upload | Voice appends",
                (20, status_top + 135),
                (210, 210, 210),
                font_scale=0.55,
                thickness=1,
            )
        else:
            self._draw_text(
                frame,
                "Use Input Text button, or Start/Stop voice then Confirm Upload",
                (20, status_top + 135),
                (210, 210, 210),
                font_scale=0.55,
                thickness=1,
            )

        if err:
            show_err = err if len(err) <= 65 else (err[:62] + "...")
            self._draw_text(frame, f"Error: {show_err}", (20, h - 20), (80, 80, 255), font_scale=0.55, thickness=2)

        if is_recording:
            self.cv2.circle(frame, (w - 35, 35), 10, (0, 0, 255), -1)
            self.cv2.putText(frame, "REC", (w - 85, 42), self.cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, self.cv2.LINE_AA)

        self._draw_image_panel(frame, self.left_panel_rect, "Uploaded / Live Image", self.current_frame)
        self._draw_image_panel(frame, self.right_panel_rect, "API Result", overlay)

    def run(self):
        self._open_input_source()
        self._start_audio_stream()

        self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
        self.cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        self.cv2.setMouseCallback(self.window_name, self._on_mouse)

        try:
            while True:
                if self.input_mode == "camera":
                    ok, frame = self.cap.read()
                    if not ok:
                        raise RuntimeError("无法从摄像头读取画面")
                else:
                    frame = self.static_image.copy()

                self.current_frame = frame.copy()
                canvas = self.np.zeros((self.window_height, self.window_width, 3), dtype=self.np.uint8)
                canvas[:] = (24, 24, 24)

                self._draw_buttons(canvas)
                self._draw_status(canvas)

                hint = "Press q to quit"
                if self.input_mode == "image":
                    hint = "Image mode | Press q to quit"
                self._draw_text(canvas, hint, (20, self.top_bar_height + 10), (255, 255, 255), font_scale=0.68, thickness=2)
                self.cv2.imshow(self.window_name, canvas)

                key = self.cv2.waitKey(1) & 0xFF
                edit_action = self._handle_text_edit_key(key)
                if edit_action == "handled":
                    continue
                if edit_action == "confirm":
                    self._confirm_upload()
                    continue

                if key == ord("e"):
                    self._edit_pending_text()
                    continue

                if key == ord("q"):
                    break

                with self.state_lock:
                    if self.should_exit:
                        break
        finally:
            if self.audio_stream is not None:
                self.audio_stream.stop()
                self.audio_stream.close()

            if self.cap is not None:
                self.cap.release()

            self.cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="实时摄像头 + 语音需求 + API 打标展示")
    parser.add_argument(
        "--startup-mode",
        default="auto",
        choices=["auto", "camera", "image"],
        help="启动输入源模式：auto/camera/image，默认 auto",
    )
    parser.add_argument(
        "--input-image",
        default="",
        help="上传图片路径（支持将图片拖入终端），用于 image 模式",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument("--width", type=int, default=1280, help="预设采集宽度，默认 1280")
    parser.add_argument("--height", type=int, default=720, help="预设采集高度，默认 720")
    parser.add_argument("--window-width", type=int, default=1400, help="程序窗口宽度，默认 1400")
    parser.add_argument("--window-height", type=int, default=900, help="程序窗口高度，默认 900")
    parser.add_argument("--panel-width", type=int, default=620, help="左右图片面板宽度，默认 620")
    parser.add_argument("--panel-height", type=int, default=430, help="左右图片面板高度，默认 430")
    parser.add_argument("--sample-rate", type=int, default=16000, help="麦克风采样率，默认 16000")

    parser.add_argument(
        "--whisper-model",
        default="turbo",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper 模型大小，默认 turbo",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Whisper 推理设备，默认 auto",
    )
    parser.add_argument(
        "--source-language",
        default="auto",
        help="语音输入语言代码，如 zh/en，默认 auto 自动检测",
    )

    parser.add_argument(
        "--model",
        default="qwen3.5-vl-plus",
        help="视觉模型名，默认 qwen3.5-vl-plus",
    )
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=20,
        help="最多保留的锚框数量，默认 20；设为 0 表示不限",
    )
    parser.add_argument(
        "--result-json",
        default="latest_boxes.json",
        help="锚框结果 JSON 输出路径，默认 latest_boxes.json；留空则不保存",
    )
    parser.add_argument(
        "--api-key-env",
        default="DASHSCOPE_API_KEY",
        help="API Key 的环境变量名，默认 DASHSCOPE_API_KEY",
    )
    parser.add_argument(
        "--api-config",
        default="apikey_config.json",
        help="API 配置文件路径，默认 apikey_config.json",
    )
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI 兼容接口地址",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = RealtimeVoiceGroundingApp(args)
    app.run()


if __name__ == "__main__":
    main()

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

        self.window_name = "Realtime Voice Grounding"
        self.cap = None
        self.audio_stream = None

        self.current_frame = None
        self.overlay_image = None
        self.last_query = ""
        self.status_text = "Ready: click Start"
        self.last_error = ""

        self.recording = False
        self.processing = False
        self.audio_chunks = []
        self.audio_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.btn_start = (20, 20, 180, 70)
        self.btn_stop = (200, 20, 360, 70)

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
            "输出必须是纯 JSON，不要 markdown，不要解释。"
            "JSON 格式: "
            '{"boxes":[{"label":"目标名","bbox":[x1,y1,x2,y2],"score":0.0}]}'
            "。bbox 优先返回 0~1000 归一化坐标；若返回像素坐标也可。"
            "如果找不到目标，返回 {\"boxes\":[]}。"
            f"用户描述: {query}"
        )

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
        boxes = parsed.get("boxes", []) if isinstance(parsed, dict) else []
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

        return result

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

    def _on_mouse(self, event, x, y, flags, param):
        if event != self.cv2.EVENT_LBUTTONDOWN:
            return

        if self._inside(x, y, self.btn_start):
            self._start_recording()
        elif self._inside(x, y, self.btn_stop):
            self._stop_recording_and_process()

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
            self.status_text = "Processing: speech + vision..."

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
            target=self._process_request,
            args=(snapshot, audio),
            daemon=True,
        )
        worker.start()

    def _process_request(self, snapshot, audio):
        try:
            query = self._transcribe_audio(audio)
            if not query:
                with self.state_lock:
                    self.overlay_image = None
                    self.last_query = ""
                    self.status_text = "No speech recognized, retry"
                return

            boxes = self._infer_boxes(snapshot, query)
            marked = self._draw_boxes(snapshot, boxes)

            with self.state_lock:
                self.overlay_image = marked
                self.last_query = query
                self.status_text = f"Done: {len(boxes)} targets ({self.active_model_name})"
                self.last_error = ""
        except Exception as exc:
            with self.state_lock:
                self.status_text = "Failed, see error"
                self.last_error = str(exc)
        finally:
            with self.state_lock:
                self.processing = False

    def _draw_buttons(self, frame):
        with self.state_lock:
            is_recording = self.recording
            is_processing = self.processing

        start_color = (40, 160, 40)
        stop_color = (40, 40, 200)

        if is_recording or is_processing:
            start_color = (90, 90, 90)
        if (not is_recording) or is_processing:
            stop_color = (90, 90, 90)

        self.cv2.rectangle(frame, (self.btn_start[0], self.btn_start[1]), (self.btn_start[2], self.btn_start[3]), start_color, -1)
        self.cv2.rectangle(frame, (self.btn_stop[0], self.btn_stop[1]), (self.btn_stop[2], self.btn_stop[3]), stop_color, -1)

        self._draw_text(frame, "Start", (self.btn_start[0] + 34, self.btn_start[1] + 38), (255, 255, 255), font_scale=0.8, thickness=2)
        self._draw_text(frame, "Stop", (self.btn_stop[0] + 44, self.btn_stop[1] + 38), (255, 255, 255), font_scale=0.8, thickness=2)

    def _draw_status(self, frame):
        with self.state_lock:
            status = self.status_text
            query = self.last_query
            err = self.last_error
            is_recording = self.recording
            overlay = None if self.overlay_image is None else self.overlay_image.copy()

        h, w = frame.shape[:2]
        self.cv2.rectangle(frame, (10, h - 130), (w - 10, h - 10), (0, 0, 0), -1)
        self._draw_text(frame, f"Status: {status}", (20, h - 95), (255, 255, 255), font_scale=0.65, thickness=2)

        if query:
            show_query = query if len(query) <= 55 else (query[:52] + "...")
            self._draw_text(frame, f"Query: {show_query}", (20, h - 60), (255, 255, 0), font_scale=0.65, thickness=2)

        if err:
            show_err = err if len(err) <= 65 else (err[:62] + "...")
            self._draw_text(frame, f"Error: {show_err}", (20, h - 25), (80, 80, 255), font_scale=0.55, thickness=2)

        if is_recording:
            self.cv2.circle(frame, (w - 35, 35), 10, (0, 0, 255), -1)
            self.cv2.putText(frame, "REC", (w - 85, 42), self.cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, self.cv2.LINE_AA)

        if overlay is not None:
            target_w = int(w * 0.38)
            target_h = int(h * 0.38)
            resized = self.cv2.resize(overlay, (target_w, target_h))
            x1 = w - target_w - 15
            y1 = 85
            frame[y1:y1 + target_h, x1:x1 + target_w] = resized
            self.cv2.rectangle(frame, (x1, y1), (x1 + target_w, y1 + target_h), (255, 255, 255), 2)
            self._draw_text(frame, "API Result", (x1 + 10, y1 - 12), (255, 255, 255), font_scale=0.7, thickness=2)

    def run(self):
        self._open_camera()
        self._start_audio_stream()

        self.cv2.namedWindow(self.window_name)
        self.cv2.setMouseCallback(self.window_name, self._on_mouse)

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    raise RuntimeError("无法从摄像头读取画面")

                self.current_frame = frame
                canvas = frame.copy()

                self._draw_buttons(canvas)
                self._draw_status(canvas)

                self._draw_text(canvas, "Press q to quit", (20, 105), (255, 255, 255), font_scale=0.7, thickness=2)
                self.cv2.imshow(self.window_name, canvas)

                key = self.cv2.waitKey(1) & 0xFF
                if key == ord("q"):
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
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument("--width", type=int, default=1280, help="预设采集宽度，默认 1280")
    parser.add_argument("--height", type=int, default=720, help="预设采集高度，默认 720")
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

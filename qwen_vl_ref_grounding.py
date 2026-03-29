import argparse
import base64
import importlib
import json
import os
import re
import sys
from io import BytesIO


def parse_args():
    parser = argparse.ArgumentParser(
        description="调用 qwen3.5-vl-plus 做图文指代，并输出打锚框后的图片"
    )
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--query", required=True, help="指代文本，如：圈出左边穿红衣服的人")
    parser.add_argument(
        "--output",
        default="grounded_output.jpg",
        help="输出图片路径，默认 grounded_output.jpg",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5-vl-plus",
        help="模型名，默认 qwen3.5-vl-plus",
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
    parser.add_argument(
        "--box-color",
        default="red",
        help="锚框颜色，默认 red",
    )
    parser.add_argument(
        "--box-width",
        type=int,
        default=4,
        help="锚框线宽，默认 4",
    )
    return parser.parse_args()


def load_openai_client(api_key, base_url):
    try:
        openai_mod = importlib.import_module("openai")
    except ImportError as exc:
        raise RuntimeError("缺少依赖 openai，请先安装：pip install openai") from exc

    client_cls = getattr(openai_mod, "OpenAI", None)
    if client_cls is None:
        raise RuntimeError("当前 openai 包版本不支持 OpenAI 客户端，请升级：pip install -U openai")

    return client_cls(api_key=api_key, base_url=base_url)


def load_api_config(path):
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


def load_pillow():
    try:
        image_mod = importlib.import_module("PIL.Image")
        draw_mod = importlib.import_module("PIL.ImageDraw")
        font_mod = importlib.import_module("PIL.ImageFont")
    except ImportError as exc:
        raise RuntimeError("缺少依赖 Pillow，请先安装：pip install Pillow") from exc

    return image_mod, draw_mod, font_mod


def image_file_to_data_url(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "application/octet-stream")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def extract_first_json(text):
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


def clamp(v, low, high):
    return max(low, min(high, v))


def normalize_boxes(parsed, width, height):
    boxes = parsed.get("boxes", []) if isinstance(parsed, dict) else []
    normalized = []

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

        # 同时兼容两类坐标：
        # 1) 0~1000 归一化坐标（Qwen 常见输出）
        # 2) 像素坐标
        if max(x1, y1, x2, y2) <= 1000.0 and min(x1, y1, x2, y2) >= 0.0:
            x1 = x1 / 1000.0 * width
            x2 = x2 / 1000.0 * width
            y1 = y1 / 1000.0 * height
            y2 = y2 / 1000.0 * height

        left = int(clamp(min(x1, x2), 0, width - 1))
        top = int(clamp(min(y1, y2), 0, height - 1))
        right = int(clamp(max(x1, x2), 0, width - 1))
        bottom = int(clamp(max(y1, y2), 0, height - 1))

        if right <= left or bottom <= top:
            continue

        label = item.get("label") or f"target_{i}"
        score = item.get("score")
        normalized.append(
            {
                "bbox": [left, top, right, bottom],
                "label": str(label),
                "score": score,
            }
        )

    return normalized


def draw_boxes(image_mod, draw_mod, font_mod, image_path, output_path, boxes, box_color, box_width):
    image = image_mod.open(image_path).convert("RGB")
    draw = draw_mod.Draw(image)
    font = font_mod.load_default()

    for item in boxes:
        left, top, right, bottom = item["bbox"]
        label = item["label"]
        score = item.get("score")

        caption = label
        if isinstance(score, (int, float)):
            caption = f"{label} ({score:.2f})"

        draw.rectangle([left, top, right, bottom], outline=box_color, width=box_width)

        text_bbox = draw.textbbox((left, top), caption, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        text_x = left
        text_y = max(0, top - text_h - 4)
        draw.rectangle(
            [text_x, text_y, text_x + text_w + 6, text_y + text_h + 4],
            fill=box_color,
        )
        draw.text((text_x + 3, text_y + 2), caption, fill="white", font=font)

    image.save(output_path)


def build_prompt(query):
    return (
        "你是视觉定位助手。请根据用户描述在图中找目标并返回 JSON。"
        "输出必须是纯 JSON，不要 markdown，不要解释。"
        "JSON 格式: "
        '{"boxes":[{"label":"目标名","bbox":[x1,y1,x2,y2],"score":0.0}]}'
        "。bbox 优先返回 0~1000 归一化坐标；若返回像素坐标也可。"
        "如果找不到目标，返回 {\"boxes\":[]}。"
        f"用户描述: {query}"
    )


def infer_boxes(client, model, image_data_url, query):
    prompt = build_prompt(query)
    response = client.chat.completions.create(
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

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("模型返回为空")

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        content = "\n".join(text_parts)

    parsed = extract_first_json(str(content))
    return parsed


def main():
    args = parse_args()
    api_conf = load_api_config(args.api_config)

    api_key = str(api_conf.get("api_key", "")).strip() or os.getenv(args.api_key_env, "").strip()
    base_url = str(api_conf.get("base_url", "")).strip() or args.base_url
    model_name = str(api_conf.get("model", "")).strip() or args.model

    if not api_key:
        raise RuntimeError(
            f"未找到 API Key，请在 {args.api_config} 中配置 api_key，"
            f"或设置环境变量 {args.api_key_env}"
        )

    image_mod, draw_mod, font_mod = load_pillow()
    image = image_mod.open(args.image).convert("RGB")
    width, height = image.size

    # 转成 data URL，避免公网可访问链接要求。
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{b64}"

    client = load_openai_client(api_key=api_key, base_url=base_url)
    parsed = infer_boxes(client, model_name, image_data_url, args.query)
    boxes = normalize_boxes(parsed, width, height)

    if not boxes:
        print("未检测到可用锚框，已输出原图。")
        image.save(args.output)
        print(f"输出图片: {args.output}")
        return

    draw_boxes(
        image_mod=image_mod,
        draw_mod=draw_mod,
        font_mod=font_mod,
        image_path=args.image,
        output_path=args.output,
        boxes=boxes,
        box_color=args.box_color,
        box_width=max(1, args.box_width),
    )

    print(f"共绘制 {len(boxes)} 个锚框")
    print(f"输出图片: {args.output}")
    print("锚框详情:")
    for i, item in enumerate(boxes, start=1):
        print(f"{i}. label={item['label']}, bbox={item['bbox']}, score={item.get('score')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
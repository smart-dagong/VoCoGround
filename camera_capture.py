import argparse
import importlib
import os
import time
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="摄像头拍照脚本")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="摄像头索引，默认 0",
    )
    parser.add_argument(
        "--output-dir",
        default="photos",
        help="照片保存目录，默认 photos",
    )
    parser.add_argument(
        "--filename-prefix",
        default="photo",
        help="照片文件名前缀，默认 photo",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="需要拍摄的照片数量，默认 1",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="连续拍照间隔秒数，默认 1.0",
    )
    parser.add_argument(
        "--timer",
        type=float,
        default=0.0,
        help="开始拍照前的倒计时秒数，默认 0",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="预设采集宽度，默认 1280",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="预设采集高度，默认 720",
    )
    return parser.parse_args()


def load_cv2():
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise RuntimeError("缺少依赖 opencv-python，请先安装：pip install opencv-python") from exc


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def build_photo_path(output_dir, filename_prefix, shot_idx):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{filename_prefix}_{timestamp}_{shot_idx:03d}.jpg")


def countdown(seconds):
    if seconds <= 0:
        return
    remaining = int(seconds)
    while remaining > 0:
        print(f"倒计时: {remaining} 秒")
        time.sleep(1)
        remaining -= 1
    fractional = seconds - int(seconds)
    if fractional > 0:
        time.sleep(fractional)


def save_frame(cv2, output_path, frame):
    ok = cv2.imwrite(output_path, frame)
    if not ok:
        raise RuntimeError(f"保存照片失败: {output_path}")


def interactive_capture(cv2, cap, args):
    print("窗口已打开。按空格拍照，按 q 退出。")
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("无法从摄像头读取画面")

        cv2.imshow("Camera Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            saved_count += 1
            output_path = build_photo_path(args.output_dir, args.filename_prefix, saved_count)
            save_frame(cv2, output_path, frame)
            print(f"已保存: {output_path}")

            if saved_count >= args.count:
                break

    if saved_count == 0:
        print("未拍摄任何照片。")


def timed_capture(cv2, cap, args):
    countdown(args.timer)

    for i in range(1, args.count + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("无法从摄像头读取画面")

        output_path = build_photo_path(args.output_dir, args.filename_prefix, i)
        save_frame(cv2, output_path, frame)
        print(f"已保存: {output_path}")

        if i < args.count:
            time.sleep(max(0.0, args.interval))


def main():
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count 必须大于 0")

    cv2 = load_cv2()
    ensure_output_dir(args.output_dir)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头，索引={args.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    try:
        if args.timer > 0 or args.count > 1:
            timed_capture(cv2, cap, args)
        else:
            interactive_capture(cv2, cap, args)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

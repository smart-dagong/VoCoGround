import argparse
import importlib
import os
import queue
import tempfile
import threading
import time
import warnings
import wave

import numpy as np
import sounddevice as sd

warnings.filterwarnings(
	"ignore",
	message="pkg_resources is deprecated as an API.*",
	category=UserWarning,
)

import whisper


def resolve_device(device_arg):
	if device_arg in ["cpu", "cuda"]:
		return device_arg

	try:
		torch = importlib.import_module("torch")
		if torch.cuda.is_available():
			return "cuda"
	except ImportError:
		pass

	return "cpu"


def load_model(model_name="turbo", device="auto"):
	resolved_device = resolve_device(device)
	return whisper.load_model(model_name, device=resolved_device)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Whisper 麦克风录音转文本（支持开始/结束控制、翻译、实时字幕）"
	)
	parser.add_argument(
		"--model",
		default="turbo",
		choices=["tiny", "base", "small", "medium", "large", "turbo"],
		help="Whisper 模型大小，默认 turbo",
	)
	parser.add_argument(
		"--device",
		default="auto",
		choices=["auto", "cpu", "cuda"],
		help="推理设备：auto 自动选择（优先 CUDA），或强制 cpu/cuda",
	)
	parser.add_argument(
		"--source-language",
		default="auto",
		help="输入音频语言代码，如 zh/en/ja，默认 auto 自动检测",
	)
	parser.add_argument(
		"--target-language",
		default="original",
		help="输出语言。original=返回原语言；en=Whisper 直接翻译到英文；其他语言将尝试二次翻译",
	)
	parser.add_argument(
		"--save-audio",
		action="store_true",
		help="是否保存最终录音文件",
	)
	parser.add_argument(
		"--save-path",
		default="recording.wav",
		help="保存录音路径，默认 recording.wav",
	)
	parser.add_argument(
		"--realtime",
		action="store_true",
		help="开启流式实时字幕（分块转写）",
	)
	parser.add_argument(
		"--chunk-seconds",
		type=float,
		default=2.0,
		help="实时字幕分块秒数，默认 2.0",
	)
	parser.add_argument(
		"--realtime-threshold",
		type=float,
		default=0.015,
		help="实时字幕音频阈值(0-1，按RMS能量过滤底噪)，默认 0.015",
	)
	return parser.parse_args()


def save_wav(path, audio, sample_rate=16000):
	with wave.open(path, "wb") as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)
		wf.setframerate(sample_rate)
		wf.writeframes(audio.tobytes())


def maybe_translate_text(text, source_language, target_language):
	target = target_language.lower()
	if target in ["original", "orig", "raw"]:
		return text

	try:
		deep_translator = importlib.import_module("deep_translator")
		GoogleTranslator = getattr(deep_translator, "GoogleTranslator")
	except (ImportError, AttributeError) as exc:
		raise RuntimeError(
			"要翻译到非英文语言，请先安装 deep-translator：pip install deep-translator"
		) from exc

	source = "auto" if source_language in [None, "auto"] else source_language
	translated = GoogleTranslator(source=source, target=target).translate(text)
	return translated


def transcribe_audio_file(model, wav_path, source_language, target_language):
	source_lang = None if source_language == "auto" else source_language
	target = target_language.lower()
	is_cuda = str(getattr(model, "device", "cpu")).startswith("cuda")
	common_kwargs = {"language": source_lang, "fp16": is_cuda}

	if target == "en":
		result = model.transcribe(wav_path, task="translate", **common_kwargs)
		return result["text"].strip()

	result = model.transcribe(wav_path, task="transcribe", **common_kwargs)
	text = result["text"].strip()
	return maybe_translate_text(text, source_lang, target)


def chunk_is_loud_enough(chunk, threshold):
	if threshold <= 0:
		return True

	if chunk.size == 0:
		return False

	samples = chunk.astype(np.float32)
	if np.issubdtype(chunk.dtype, np.integer):
		scale = float(np.iinfo(chunk.dtype).max)
		samples = samples / scale

	rms = float(np.sqrt(np.mean(np.square(samples))))
	return rms >= threshold


def record_until_enter(
	model,
	sample_rate=16000,
	channels=1,
	dtype="int16",
	realtime=False,
	chunk_seconds=2.0,
	realtime_threshold=0.015,
	source_language="auto",
	target_language="original",
):
	print("按回车开始录音...")
	input()
	print("录音中，按回车结束。")

	frames = []
	stop_event = threading.Event()
	chunk_queue = queue.Queue()
	chunk_samples = max(1, int(sample_rate * chunk_seconds))

	def callback(indata, frame_count, time_info, status):
		if status:
			print(f"录音状态: {status}")

		copied = indata.copy()
		frames.append(copied)

		if realtime:
			chunk_queue.put(copied)

		if stop_event.is_set():
			raise sd.CallbackStop()

	stream = sd.InputStream(
		samplerate=sample_rate,
		channels=channels,
		dtype=dtype,
		callback=callback,
		blocksize=chunk_samples if realtime else 0,
	)

	def wait_for_stop():
		input()
		stop_event.set()

	stop_thread = threading.Thread(target=wait_for_stop, daemon=True)

	with stream:
		stop_thread.start()

		if realtime:
			print(f"\n实时字幕(阈值={realtime_threshold:.3f}):")
			while not stop_event.is_set() or not chunk_queue.empty():
				try:
					chunk = chunk_queue.get(timeout=0.2)
				except queue.Empty:
					continue

				if not chunk_is_loud_enough(chunk, realtime_threshold):
					continue

				with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
					chunk_path = f.name

				try:
					save_wav(chunk_path, chunk, sample_rate=sample_rate)
					chunk_text = transcribe_audio_file(
						model,
						chunk_path,
						source_language=source_language,
						target_language=target_language,
					)
					if chunk_text:
						print(chunk_text)
				finally:
					try:
						os.remove(chunk_path)
					except OSError:
						pass
		else:
			stop_thread.join()

	if not frames:
		raise RuntimeError("没有录到音频，请检查麦克风权限或设备设置。")

	audio = np.concatenate(frames, axis=0)
	return audio


def main():
	args = parse_args()
	sample_rate = 16000
	device = resolve_device(args.device)

	print(f"加载模型: {args.model} (device={device})")
	model = load_model(args.model, device=device)

	audio = record_until_enter(
		model=model,
		sample_rate=sample_rate,
		realtime=args.realtime,
		chunk_seconds=args.chunk_seconds,
		realtime_threshold=args.realtime_threshold,
		source_language=args.source_language,
		target_language=args.target_language,
	)
	audio_duration = len(audio) / float(sample_rate)

	with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
		wav_path = f.name

	save_wav(wav_path, audio, sample_rate=sample_rate)

	if args.save_audio:
		save_wav(args.save_path, audio, sample_rate=sample_rate)
		print(f"已保存录音文件: {args.save_path}")

	recognize_start = time.perf_counter()
	final_text = transcribe_audio_file(
		model,
		wav_path,
		source_language=args.source_language,
		target_language=args.target_language,
	)
	recognize_elapsed = time.perf_counter() - recognize_start

	print("\n最终识别结果:")
	print(final_text)
	print(f"原音频时长: {audio_duration:.2f} 秒")
	print(f"识别耗时: {recognize_elapsed:.2f} 秒")

	try:
		os.remove(wav_path)
	except OSError:
		pass


if __name__ == "__main__":
	main()
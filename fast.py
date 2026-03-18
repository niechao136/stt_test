from faster_whisper import WhisperModel
import os
import time
# 解决网络连接问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

if __name__ == "__main__":
    # model_size 可以是路径，也可以是 "tiny"
    # device="cpu", compute_type="int8" 是 CPU 运行的最快组合
    model_path = "model/faster-whisper-tiny"
    audio_path = "samples/sample-1757066057945.mp3"
    print(f"正在从 {model_path} 加载模型...")
    start_load = time.time()
    model = WhisperModel(model_size_or_path=model_path, device="cpu", compute_type="int8")
    load_duration = time.time() - start_load
    print(f"模型加载完成，耗时: {load_duration:.2f} 秒")

    start = time.time()
    # segments 是一个生成器
    segments, info = model.transcribe(audio=audio_path, beam_size=5)

    print(f"检测到语言: {info.language}，概率: {info.language_probability:.2f}")

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

    print(f"耗时: {time.time() - start:.2f}s")
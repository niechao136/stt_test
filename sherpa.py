import sherpa_onnx
import os
import numpy as np
import time
import av


def load_audio_fast(path):
    """使用 PyAV 极速解码 MP3 并重采样至 16kHz"""
    # 1. 打开文件容器
    container = av.open(path)

    # 2. 设置重采样器：强制输出为 16000Hz, 单声道, float 格式
    resampler = av.AudioResampler(
        format='fltp',
        layout='mono',
        rate=16000,
    )

    all_frames = []

    # 3. 直接在流中解码
    for frame in container.decode(audio=0):
        # 重采样并加入列表
        resampled_frames = resampler.resample(frame)
        for f in resampled_frames:
            all_frames.append(f.to_ndarray())

    # 4. 合并所有帧
    if not all_frames:
        return np.array([], dtype=np.float32), 16000

    sample = np.concatenate(all_frames).flatten()
    return sample, 16000

if __name__ == "__main__":
    # 1. 确保路径正确
    # model_path = "model/paraformer-large-offline/model.int8.onnx"
    # tokens_path = "model/paraformer-large-offline/tokens.txt"
    model_path = "model/sensevoice-small/model_q8.onnx"
    tokens_path = "model/sensevoice-small/tokens.txt"
    wav_path = "samples/sample-1757066057945.mp3"

    # 2. 初始化识别器
    print(f"正在从 {model_path} 加载模型...")
    start_load = time.time()
    # recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
    #     paraformer=model_path,
    #     tokens=tokens_path,
    #     num_threads=4
    # )
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model_path,
        tokens=tokens_path,
        num_threads=4
    )
    load_duration = time.time() - start_load
    print(f"模型加载完成，耗时: {load_duration:.2f} 秒")
    # 3. 读取音频文件
    # sherpa_onnx 提供了方便的 read_wave 函数，它返回 (samples, sample_rate)
    if not os.path.exists(wav_path):
        print(f"找不到音频文件: {wav_path}")
    else:
        start = time.time()
        samples, sample_rate = load_audio_fast(wav_path)
        # 4. 创建流并传入波形数据
        stream = recognizer.create_stream()
        # accept_waveform 接收 numpy 数组和对应的采样率
        stream.accept_waveform(sample_rate, samples)

        # 5. 解码并输出
        recognizer.decode_stream(stream)
        print("-" * 30)
        print(f"识别结果: {stream.result.text}")
        print("-" * 30)
        print(f"耗时: {time.time() - start:.2f}s")
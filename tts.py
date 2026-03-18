import sherpa_onnx
import wave
import os
import numpy as np


def generate_tts_sample():
    # --- 1. 配置模型路径 (请确保这些文件已下载到本地) ---
    # 以 vits-melo-tts-zh_en 为例，你需要修改为你的实际路径
    model_dir = "model/vits-melo-tts-zh_en"

    # 1. 明确指定 VITS 模型配置
    model_config = sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=os.path.join(model_dir, "model.onnx"),
            lexicon=os.path.join(model_dir, "lexicon.txt"),
            tokens=os.path.join(model_dir, "tokens.txt"),
            data_dir="",  # 如果没有 espeak-ng-data，留空
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0,
        ),
        num_threads=4,
        debug=False,
        provider="cpu",  # 或者 "cuda" 如果你有 GPU
    )

    # 2. 这里的类名可能是 OfflineTtsConfig 而不是 OfflineTtsGeneralConfig
    # 提示：如果 OfflineTtsGeneralConfig 报错，请直接使用 OfflineTtsConfig
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=model_config,
        rule_fsts="",
        max_num_sentences=1,
    )

    tts = sherpa_onnx.OfflineTts(tts_config)

    # --- 3. 生成音频 ---
    text = "你好，这是使用 Sherpa O N N X 在本地生成的语音测试。忽略目录的操作已经完成。"
    sid = 0  # 说话人 ID (对于多发言人模型，可以修改这个数字)

    print("正在生成音频...")
    audio = tts.generate(text, sid=sid)

    # --- 调试：检查生成的音频大小 ---
    print(f"生成的样本数量: {len(audio.samples)}")
    print(f"采样率: {audio.sample_rate}")

    # 计算预计音频时长（秒）
    duration = len(audio.samples) / audio.sample_rate
    print(f"预计音频时长: {duration:.2f} 秒")

    # 检查是否因为生成了过长的音频导致溢出
    if duration > 300:  # 如果超过5分钟，说明模型输出异常
        print("错误：生成的音频过长，可能模型配置有误。")
    else:
        # --- 安全的转换方式 ---
        output_filename = "test_output.wav"

        # 将 float32 转换为 int16，先限制范围防止溢出
        samples = np.clip(audio.samples, -1, 1)
        samples = (samples * 32767).astype(np.int16)

        with wave.open(output_filename, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(audio.sample_rate)
            f.writeframes(samples.tobytes())

        print(f"成功！音频已保存至: {output_filename}")


if __name__ == "__main__":
    generate_tts_sample()
import whisper
import time
import os

if __name__ == "__main__":
    # 1. 设置模型路径和音频路径
    model_path = "model/tiny.pt"
    audio_path = "samples/sample-1757066057945.mp3"

    # 2. 加载模型（记录加载耗时）
    print(f"正在从 {model_path} 加载模型...")
    start_load = time.time()
    model = whisper.load_model(name=model_path)
    load_duration = time.time() - start_load
    print(f"模型加载完成，耗时: {load_duration:.2f} 秒")

    # 3. 开始识别（记录推理耗时）
    if os.path.exists(audio_path):
        print(f"正在处理音频: {audio_path} ...")

        start_inference = time.time()
        # fp16=False 避免 CPU 运行时的警告
        result = model.transcribe(audio=audio_path, fp16=False)
        end_inference = time.time()

        # 4. 计算速度指标
        inference_duration = end_inference - start_inference

        # 获取音频时长（可选：如果想计算实时率）
        # 如果没有安装 pydub，可以粗略通过 result 中的 end 时间获取
        audio_duration = result['segments'][-1]['end'] if result['segments'] else 0
        rtf = inference_duration / audio_duration if audio_duration > 0 else 0

        # 5. 输出结果
        print("-" * 30)
        print(f"识别文字: {result['text']}")
        print("-" * 30)
        print(f"音频时长: {audio_duration:.2f} 秒")
        print(f"识别耗时: {inference_duration:.2f} 秒")
        print(f"实时率 (RTF): {rtf:.4f} (越小越快)")
        # RTF < 1 表示比说话速度快，RTF > 1 表示比说话速度慢
    else:
        print(f"文件未找到: {audio_path}")

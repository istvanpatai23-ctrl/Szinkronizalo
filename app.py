import gradio as gr
import pysrt
from pydub import AudioSegment
from TTS.api import TTS
import torch
import os
import pandas as pd

def srt_time_to_ms(srt_time):
    return (srt_time.hours * 3600 + srt_time.minutes * 60 + srt_time.seconds) * 1000 + srt_time.milliseconds

def run_process(srt_file, model_name, speaker, speed, cps_limit):
    if not srt_file: return None, None
    
    subs = pysrt.open(srt_file.name)
    analysis = []
    for s in subs:
        dur = (s.end.ordinal - s.start.ordinal) / 1000.0
        cps = len(s.text) / dur if dur > 0 else 0
        analysis.append([s.index, f"{cps:.1f}", "⚠️" if cps > cps_limit else "✅", s.text])
    
    df = pd.DataFrame(analysis, columns=["Sor", "CPS", "Status", "Szöveg"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name).to(device)
    
    audio = AudioSegment.silent(duration=0)
    last_ms = 0
    
    for i, s in enumerate(subs):
        start = srt_time_to_ms(s.start)
        if start > last_ms:
            audio += AudioSegment.silent(duration=(start - last_ms))
            
        tmp = f"t_{i}.wav"
        tts.tts_to_file(text=s.text, speaker=speaker if tts.is_multi_speaker else None, file_path=tmp)
        
        seg = AudioSegment.from_wav(tmp)
        if speed != 1.0:
            seg = seg._spawn(seg.raw_data, overrides={"frame_rate": int(seg.frame_rate * speed)}).set_frame_rate(seg.frame_rate)
            
        audio += seg
        last_ms = start + len(seg)
        os.remove(tmp)
        
    out = "output.wav"
    audio.export(out, format="wav")
    return df, out

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            f = gr.File()
            m = gr.Dropdown(["tts_models/en/ljspeech/glow-tts", "tts_models/multilingual/multi-dataset/xtts_v2"], value="tts_models/en/ljspeech/glow-tts")
            sk = gr.Textbox(label="Speaker", value="ED\u0301gar")
            sp = gr.Slider(0.5, 1.5, 1.0, label="Speed")
            cl = gr.Slider(10, 25, 15, label="CPS Limit")
            b = gr.Button("Mehet")
        with gr.Column():
            a = gr.Audio()
            t = gr.Dataframe()
    b.click(run_process, [f, m, sk, sp, cl], [t, a])

demo.launch(share=True)

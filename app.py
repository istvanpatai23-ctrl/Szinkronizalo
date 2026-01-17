import gradio as gr
import pysrt
import tempfile
from pydub import AudioSegment
from TTS.api import TTS

# Load Khmer-compatible multilingual model (loaded once)
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=False,
    gpu=False
)

def srt_to_audio(srt_file):
    subs = pysrt.open(srt_file)
    final_audio = AudioSegment.silent(duration=0)

    for sub in subs:
        text = sub.text.replace("\n", " ").strip()
        if not text:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            tts.tts_to_file(
                text=text,
                speaker_wav=None,
                language="km",
                file_path=f.name
            )
            audio = AudioSegment.from_wav(f.name)

        start_ms = sub.start.ordinal
        end_ms = sub.end.ordinal
        duration = end_ms - start_ms

        audio = audio[:duration]
        silence = AudioSegment.silent(duration=start_ms - len(final_audio))
        final_audio += silence + audio

    output_path = "output_khmer.wav"
    final_audio.export(output_path, format="wav")
    return output_path

ui = gr.Interface(
    fn=srt_to_audio,
    inputs=gr.File(label="Upload SRT (Khmer)"),
    outputs=gr.Audio(type="filepath"),
    title="Fast Khmer Subtitle → Speech (FREE)",
    description="High-speed Khmer TTS without per-line API calls."
)

ui.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
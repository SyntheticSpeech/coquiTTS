import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
tts.tts_to_file(
    "Ladies and Gentleman, today is an important day. I am going to be a graduate student. Yesterday, I was an undergraduate. Four years ago, I was a high school student.",
    language="en",
    speaker_wav="moon_speech.wav",
    file_path="output.wav"
)
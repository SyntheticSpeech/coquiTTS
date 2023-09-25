import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
tts.tts_to_file(
    "the operating system, the compiler, or the network.",
    language="en",
    speaker_wav="./fine-tune_dataset/wavs/8.wav",
    file_path="output_7.wav"
)
tts.tts_to_file(
    "If you study and learn the concepts in this book, you will be on your way to becoming the rare power programmer who knows how things work and how to fix them, why they break.",
    language="en",
    speaker_wav="./fine-tune_dataset/wavs/8.wav",
    file_path="output_9.wav"
)
tts.tts_to_file(
    "Whenever you learn something new, you can try it out right away and see the result firsthand.",
    language="en",
    speaker_wav="./fine-tune_dataset/wavs/8.wav",
    file_path="output_13.wav"
)
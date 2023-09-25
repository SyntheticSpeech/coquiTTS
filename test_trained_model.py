import torch
from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# tts = TTS(model_path="/content/coquiTTS/output/YourTTS-Hao-September-25-2023_09+33PM-1a40e4d3/best_model.pth",
#           config_path="/content/coquiTTS/output/YourTTS-Hao-September-25-2023_09+33PM-1a40e4d3/config.json",).to(device)
# tts.model_name = "tts_models/multilingual/multi-dataset/your_tts"
model_path="/content/coquiTTS/output/YourTTS-Hao-September-25-2023_09+33PM-1a40e4d3/best_model.pth"
config_path="/content/coquiTTS/output/YourTTS-Hao-September-25-2023_09+33PM-1a40e4d3/config.json"
synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    tts_speakers_file=None,
    tts_languages_file=None,
    vocoder_checkpoint=None,
    vocoder_config=None,
    encoder_checkpoint=None,
    encoder_config=None,
    use_cuda=False,
)
wav = synthesizer.tts(
    text="the operating system, the compiler, or the network.",
    speaker_name=None,
    language_name="en",
    speaker_wav="/content/drive/MyDrive/fine-tune_dataset/wavs/8.wav",
    reference_wav=None,
    style_wav=None,
    style_text=None,
    reference_speaker_name=None,
)
synthesizer.save_wav(wav=wav, path="output_7.wav")

wav = synthesizer.tts(
    text="If you study and learn the concepts in this book, you will be on your way to becoming the rare power programmer who knows how things work and how to fix them, why they break.",
    speaker_name=None,
    language_name="en",
    speaker_wav="/content/drive/MyDrive/fine-tune_dataset/wavs/8.wav",
    reference_wav=None,
    style_wav=None,
    style_text=None,
    reference_speaker_name=None,
)
synthesizer.save_wav(wav=wav, path="output_9.wav")

wav = synthesizer.tts(
    text="Whenever you learn something new, you can try it out right away and see the result firsthand.",
    speaker_name=None,
    language_name="en",
    speaker_wav="/content/drive/MyDrive/fine-tune_dataset/wavs/8.wav",
    reference_wav=None,
    style_wav=None,
    style_text=None,
    reference_speaker_name=None,
)
synthesizer.save_wav(wav=wav, path="output_13.wav")

# tts.tts_to_file(
#     "the operating system, the compiler, or the network.",
#     language="en",
#     speaker_wav="./fine-tune_dataset/wavs/8.wav",
#     file_path="output_7.wav"
# )
# tts.tts_to_file(
#     "If you study and learn the concepts in this book, you will be on your way to becoming the rare power programmer who knows how things work and how to fix them, why they break.",
#     language="en",
#     speaker_wav="./fine-tune_dataset/wavs/8.wav",
#     file_path="output_9.wav"
# )
# tts.tts_to_file(
#     "Whenever you learn something new, you can try it out right away and see the result firsthand.",
#     language="en",
#     speaker_wav="./fine-tune_dataset/wavs/8.wav",
#     file_path="output_13.wav"
# )
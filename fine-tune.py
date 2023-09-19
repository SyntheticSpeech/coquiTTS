import os
import subprocess

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.utils.downloaders import download_libri_tts


def m4a_to_wav():
    source_folder = './fine-tune_dataset/m4as'
    dest_folder = './fine-tune_dataset/wavs'
    # Ensure the output folder exists, or create it if it doesn't
    os.makedirs(dest_folder, exist_ok=True)
    # Loop through all files in the input folder
    for filename in os.listdir(source_folder):
        if filename.endswith('.m4a'):
            input_file = os.path.join(source_folder, filename)
            output_file = os.path.join(dest_folder, os.path.splitext(filename)[0] + '.wav')
            # Use subprocess to run FFmpeg to convert the file
            cmd = ['ffmpeg', '-i', input_file, '-acodec', 'pcm_s16le', '-ar', '24000', output_file]
            try:
                subprocess.run(cmd, check=True)
                print(f"Converted {input_file} to {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {input_file}: {e}")

# Download the model
# tts --language_idx en --model_name tts_models/multilingual/multi-dataset/your_tts --text "Ola."
RESTORE_PATH = os.path.normpath("/Users/apple/Library/Application Support/tts/tts_models--multilingual--multi-dataset--your_tts/model_file.pth")
# Name of the run for the Trainer
RUN_NAME = "YourTTS-Hao"
# Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
OUT_PATH = "./output"
# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False
# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 8
# Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this you might need to redownload the dataset !!)
# Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
SAMPLE_RATE = 24000
# Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
MAX_AUDIO_LEN_IN_SECONDS = float("inf")
DATASET_PATH = "./fine-tune_dataset"

# init dataset configs
ds_config = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="fine-tune-hao",
    meta_file_train="train.txt",
    meta_file_val="",
    path=DATASET_PATH,
    language="en"
)
DATASETS_CONFIG_LIST = [ds_config]
### Extract speaker embeddings
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
)
SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"
D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training
# Check if the embeddings weren't already computed, if not compute it
embeddings_file = os.path.join(ds_config.path, "speakers.pth")
if not os.path.isfile(embeddings_file):
    print(f">>> Computing the speaker embeddings for the {ds_config.dataset_name} dataset")
    compute_embeddings(
        SPEAKER_ENCODER_CHECKPOINT_PATH,
        SPEAKER_ENCODER_CONFIG_PATH,
        embeddings_file,
        old_speakers_file=None,
        config_dataset_path=None,
        formatter_name=ds_config.formatter,
        dataset_name=ds_config.dataset_name,
        dataset_path=ds_config.path,
        meta_file_train=ds_config.meta_file_train,
        meta_file_val=ds_config.meta_file_val,
        disable_cuda=False,
        no_eval=True,
    )
D_VECTOR_FILES.append(embeddings_file)


# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    num_layers_text_encoder=10,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
    use_speaker_encoder_as_loss=False,
    # Useful parameters to enable multilingual training
    # use_language_embedding=True,
    # embedded_language_dim=4,
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="YourTTS",
    run_description="""
            - YourTTS trained using custom dataset of Hao's voice
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=8,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=4,
    print_step=1,
    plot_step=100,
    log_model_step=1,
    save_step=1000,
    save_n_checkpoints=2,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=False,
    use_phonemes=False,
    phonemizer="espeak",
    phoneme_language="en",
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="multilingual_cleaners",
    phoneme_cache_path=None,
    precompute_num_workers=1,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    # weighted_sampler_attrs={"language": 1.0, "speaker_name": 1.0},
    weighted_sampler_attrs={"language": 1.0},
    weighted_sampler_multipliers={
        # "speaker_name": {
            # you can force the batching scheme to give a higher weight to a certain speaker and then this speaker will appears more frequently on the batch.
            # It will speedup the speaker adaptation process. Considering the CML train dataset and "new_speaker" as the speaker name of the speaker that you want to adapt.
            # The line above will make the balancer consider the "new_speaker" as 106 speakers so 1/4 of the number of speakers present on CML dataset.
            # 'new_speaker': 106, # (CML tot. train speaker)/4 = (424/4) = 106
        # }
    },
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the YourTTS paper
    speaker_encoder_loss_alpha=9.0,
)

# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=False,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size
)

# Init the model
model = Vits.init_from_config(config)

# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()

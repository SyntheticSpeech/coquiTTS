# YourTTS
| Dataset      | Epoch | Result | Note     | 
| :---        |    :----:   |    :----:   |          ---: |
| Hao's self collected voices (19 Samples)      | 20       | fail, voice not clearly formed       | n/a   |
| Hao's self collected voices (19 Samples)      | 160       | fail, voice not clearly formed      | n/a   |

# VITS
We used a different repo for this:
https://github.com/Plachtaa/VITS-fast-fine-tuning

The project has a pretrained model on VCTK and some Animes. It can be used for fine-tuning Chinese, Japanese and English.

| Dataset      | Epoch | Result | Note     | 
| :---        |    :----:   |    :----:   |          ---: |
| Hao's self collected voices (19 Samples)      | 20       | Tune close, pronounciation fail       | n/a   |
| Hao's self collected voices (41 Samples)      | 25       | No improvement       | n/a   |
| Anne Hathaway, 12 mins audio      | 20       | Tune close, pronounciation better, not perfect       | n/a   |
| Hao's self collected voices (15 mins audio, Chinese)      | 50       | Tune close, still strange pronouns, but better than EN       | n/a   |
| LibriSpeech(Speaker 3572, 103 samples)      | 50       | Pretty steady      | n/a   |
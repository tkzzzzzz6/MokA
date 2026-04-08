# Prepared Datasets Layout

Unified dataset root used by this project:

```text
prepared_datasets/
├── AudioCaps/
│   ├── train.json
│   ├── val.json
│   └── data/
│       └── *.wav
├── video-llava/
│   ├── train_json/
│   │   ├── llava_image_.json
│   │   └── valid_valley_.json
│   └── ... media files referenced by json ...
├── AVE_data/
│   ├── train_samples_ave.json
│   ├── test_samples_ave.json
│   ├── audio_data/
│   │   └── *.mp3
│   ├── AVE/
│   │   └── *.mp4
│   └── converted_label/
│       └── *.txt
└── MUSIC_AVQA_data/
	├── train_samples_with_reasoning_avqa.json
	├── test_samples_avqa.json
	├── audio_data/
	│   └── *
	└── video_data/
		└── *
```

Compatibility note:

1. Current training/inference loaders are migrated to use `prepared_datasets` paths.
2. Smoke inference still prioritizes `smoke_test_data/AVE_data` when that folder exists.
3. Keep all dataset files under `prepared_datasets` for normal pretrain/finetune runs.

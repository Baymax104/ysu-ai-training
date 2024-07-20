数据集：Libri2Mix

通过LibriSpeech-dev-clean数据集生成，不添加wham噪声，参考[LibriMix](https://github.com/JorisCos/LibriMix)

使用`torchaudio.datasets.librimix.LibriMix`加载

- n_src: 2
- freqs: 8k
- modes: min
- types: mix_clean

数据集目录

```
.
└── Libri2Mix
    └── wav8k
        └── min
            ├── dev
            │   ├── mix_clean
            │   ├── s1
            │   └── s2
            └── metadata
```
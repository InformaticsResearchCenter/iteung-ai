# ITeung AI
Indonesia ChatBot using Seq2Seq (LSTM and BiLSTM) with Bahdanau Attention Mechanism

# Environment
Python: 3.7.9\
CUDA Toolkit: 11.0.3 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)\
CuDNN: v8.0.5 [CuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

# Usage
1. Install all required depedency
```console
$ pip install -r requirements.txt
```

2. run preprocessing.py
```console
$ python preprocessing.py
```

3. run training.py
```console
$ python training.py
```
4. check output_dir get 4 file from there to iteung-ai plus daftar-slang from dataset
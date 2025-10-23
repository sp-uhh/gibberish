# Calculate Gibberishness Score

This is the code for the paper "Are These Even Words? Quantifying the Gibberishness of Generative Speech Models", D. de Oliveira, T. Peer, J. Rochdi, T. Gerkmann.

[[project page + dataset download]](https://uhh.de/inf-sp-gibberish)

## Installation

```
pip install torch torchaudio distillmos evaluate nemo_toolkit['asr'] git+https://github.com/sarulab-speech/UTMOSv2.git git+https://github.com/facebookresearch/fairseq.git
```

### Textless lib (TWIST)

The `textless` lib has very tight constraints, so they have to be loosened:

1. Run `git clone https://github.com/facebookresearch/textlesslib.git`
2. Change [L7-8 of `textlesslib/requirements.txt`](https://github.com/facebookresearch/textlesslib/blob/ba33d669d8284b4f7bfe81e7384e83ab799fe384/requirements.txt#L7C1-L8C14) to
```
numpy>=1.22.0
numba>=0.53.0
```
3. Run `pip install -e ./textlesslib`

### Fairseq (TWIST)

To use `fairseq` with Python 3.11+, follow [this Github issue post](https://github.com/facebookresearch/fairseq/issues/5012#issuecomment-3185202948), commenting out the block
```python
if f._field_type is _FIELD and f.default.__class__.__hash__ is None:
    raise ValueError(f'mutable default {type(f.default)} for field '
                        f'{f.name} is not allowed: use default_factory')
```
in `{PYTHON_BASE}/lib/python3.1X/dataclasses.py`, where `PYTHON_BASE` is wherever your Python is installed.

### UTMOSv2

To be able to run UTMOSv2 with files in speaker subfolders, change [L174 in `utmosv2/_core/model
/_common.py`](https://github.com/sarulab-speech/UTMOSv2/blob/3608fad976f18a0584e5837afa04fbc628572061/utmosv2/_core/model/_common.py#L174)

```python
if input_dir is not None:
    res = [
        DatasetSchema(
            file_path=p,
            dataset=predict_dataset,
        )
        for p in sorted(input_dir.glob("**/*.wav", recursive=True)) # This line
    ]
```

### SpeechLMScore

We had some issues running the original SpeechLMScore pipeline script, so we created a modified version that needs to be copied to the cloned repository. 
Additionally, we modified the Python script of the last stage to create an output csv file in the same format as our other scripts do.

```
git clone https://github.com/soumimaiti/speechlmscore_tool.git
cd speechlmscore_tool
pip install -r requirements.txt
./download_pretrained_models.sh
cp ../speechlmscore_hack/* .
```

### Huggingface's Evaluate (ASR + LLM)

If you wish to run the perplexity metric with batch size 1, the script needs a slight modification. After running the script once, it will download the perplexity module to `~/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--perplexity/<lengthy-download-folder-name>/perplexity.py`. [L122 of `perplexity.py`](https://github.com/huggingface/evaluate/blob/b3820eb820702611cd0c2247743d764f2a7fe916/metrics/perplexity/perplexity.py#L122) needs to be changed to

```python
if tokenizer.pad_token is None: # and batch_size > 1:  <= This part of the condition needs to be commented out
    existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
    # check that the model already has at least one special token defined
    assert (
        len(existing_special_tokens) > 0
    ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
    # assign one of the special tokens to also be the pad token
    tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})
```
otherwise the tokenizer complains, as it is always asked to pad (See [L143](https://github.com/huggingface/evaluate/blob/b3820eb820702611cd0c2247743d764f2a7fe916/metrics/perplexity/perplexity.py#L143)).

## How to run the evaluations

### TWIST

```
OMP_NUM_THREADS=1 python calc_twist.py --input_dir=data/earswham/test/gibberish --output_file=results/twist_earswham_gibberish.csv
```

### ASR + LLM

```
python calc_asr_llm.py --input_dir=data/earswham/test/gibberish --output_file=results/parakeet_gpt2_earswham_gibberish.csv --asr_model=parakeet
# or
python calc_asr_llm.py --input_dir=data/earswham/test/gibberish --output_file=results/quartznet_gpt2_earswham_gibberish.csv --asr_model=quartznet
```

### SpeechLMScore

```
cd speechlmscore_tool
./run_speechlmscore.sh ../data/earswham/test/gibberish ../results/speechlmscore_earswham_gibberish_test.csv
```

### DistillMOS + UTMOSv2

```
python calc_dnnmos.py --input_dir=data/earswham/test/gibberish --output_file=results/dnnmos_earswham_gibberish.csv
```

## License

This code is licensed under the [GNU AGPLv3 license](https://spdx.org/licenses/AGPL-3.0-or-later.html).
The dataset EARS-WHAM Gibberish Test is licensed under the [CC BY-NC 4.0 International license](https://www.creativecommons.org/licenses/by-nc/4.0/legalcode.en).

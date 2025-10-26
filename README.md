# Calculate Gibberishness Score

This is the code for the paper "Are These Even Words? Quantifying the Gibberishness of Generative Speech Models", D. de Oliveira, T. Peer, J. Rochdi, T. Gerkmann.

[[project page + dataset download]](https://uhh.de/inf-sp-gibberish)

> Significant research efforts are currently being dedicated to non-intrusive quality and intelligibility assessment, especially given how it enables curation of large scale datasets of in-the-wild speech data. However, with the increasing capabilities of generative models to synthesize high quality speech, new types of artifacts become relevant, such as generative hallucinations. While intrusive metrics are able to spot such sort of discrepancies from a reference signal, it is not clear how current non-intrusive methods react to high-quality phoneme confusions or, more extremely, gibberish speech. In this paper we explore how to factor in this aspect under a fully unsupervised setting by leveraging language models. Additionally, we publish LRS3-Gibberish, a dataset of high-quality synthesized gibberish speech for further development of measures to assess implausible sentences in spoken language, alongside code for calculating scores from a variety of speech language models.

With the scripts in this repository, you can compute perplexity scores with TWIST [1], SpeechLMScore (6th layer) [2] and ASR (Parakeet[3,4]/Quartznet[5]) + LLM (GPT-2[6]).
There is also a script to compute the non-intrusive quality/naturalness scores of DistillMOS [7] and UTMOSv2 [8], and a Jupyter notebook to compare distributions.

## Installation

```bash
./install.sh
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

## References

[1] M. Hassid, T. Remez, T. A. Nguyen, et al., "Textually pretrained speech language models," in Advances in Neural Inf. Proc. Systems (NeurIPS), vol. 36, 2023.

[2] S. Maiti, Y. Peng, T. Saeki, and S. Watanabe, "Speechlmscore: Evaluating speech generation using speech language model," in IEEE Int. Conf. on Acoustics, Speech and Signal Process. (ICASSP), 2023.

[3] D. Rekesh, N. R. Koluguri, S. Kriman, et al., "Fast conformer with linearly scalable attention for efficient speech recognition," in IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU), 2023.

[4] H. Xu, F. Jia, S. Majumdar, H. Huang, S. Watanabe, and B. Ginsburg, “Efficient sequence transduction by jointly predicting tokens and durations,” in Int. Conf. on Machine Learning (ICML), 2023.

[5] S. Kriman, S. Beliaev, B. Ginsburg, et al., "Quartznet: Deep automatic speech recognition with 1d time-channel separable convolutions," in IEEE Int. Conf. on Acoustics, Speech and Signal Process. (ICASSP), 2020.

[6] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al., "Language models are unsupervised multitask learners," OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[7] B. Stahl and H. Gamper, “Distillation and pruning for scalable self-supervised representation-based speech quality assessment,” in ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025.

[8] Baba, W. Nakata, Y. Saito, and H. Saruwatari, "The t05 system for the VoiceMOS Challenge 2024: Transfer learning from deep image classifier to naturalness MOS prediction of high-quality synthetic speech," in IEEE Spoken Language Technology Workshop (SLT), 2024.

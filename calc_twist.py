"""
Calculate SpeechLM perplexity for a directory and put the scores in a csv.

Copyright (C) 2025 Signal Processing Group University of Hamburg

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import logging
import warnings
import argparse
import pydantic
from tqdm import tqdm
from glob import glob
    
import numpy as np
import torch
import torchaudio

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('fairseq').setLevel(logging.WARNING)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pydantic.warnings.UnsupportedFieldAttributeWarning)

from textless.data.speech_encoder import SpeechEncoder
from textlesslib.examples.twist.speech_lm import build_speech_lm


def calc_perplexity(hubert_encoder, twist_model, speech_prompt):
    input_ids = twist_model.config.offset + hubert_encoder(speech_prompt)['units'].unsqueeze(0).type(torch.LongTensor).to(twist_model.device)
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = twist_model(input_ids, labels=target_ids)
        nll = outputs.loss.item()
    return nll

def main(args):
    # Load speech encoder and vocoder
    encoder = SpeechEncoder.by_name(
        dense_model_name = "mhubert-base-25hz",
        quantizer_model_name = "kmeans",
        vocab_size = 500,
        deduplicate=True,
        need_f0=False,
        add_bos_eos=False,
    ).eval()

    if torch.cuda.is_available():
        encoder = encoder.cuda()

    # Load twist model
    twist_model = build_speech_lm(args.twist_model)
    
    log_ppls = []
    with open(args.output_file, 'w') as f:
        f.write('filename,twist_log_ppl\n')
        files = sorted(glob(os.path.join(args.input_dir, '**/*.wav'), recursive=True))
        for input_file in tqdm(files):
            audio, sample_rate = torchaudio.load(input_file)
            assert sample_rate == 16000
            if audio.ndim == 2:
                audio = audio.mean(0)
            log_ppl = calc_perplexity(encoder, twist_model, audio)
            audio_id = os.path.join(os.path.basename(os.path.dirname(input_file)), os.path.basename(input_file))
            f.write(f'{audio_id},{round(log_ppl, 5)}\n')
            log_ppls.append(log_ppl)

        print("TWIST log ppl: ", round(np.mean(log_ppls), 3), " ± ", round(np.std(log_ppls), 3))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",   type=str, required=True, help="Input dir name")
    parser.add_argument("--output_file", type=str, required=True, help="Path where generated scores are saved")
    parser.add_argument("--twist_model", type=str, default="TWIST-350M", 
                                         choices=["TWIST-350M", "TWIST-1.3B", "TWIST-7B"], help="Name of TWIST model")
    args = parser.parse_args()
    main(args)
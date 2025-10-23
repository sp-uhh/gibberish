"""
Transcribe audio and calculate perplexity for a directory and put the scores in a csv.

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
import warnings
import logging
import argparse
from glob import glob

import numpy as np
import torch
import evaluate
import nemo.collections.asr as nemo_asr

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('nemo_logger').setLevel(logging.ERROR)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size

    if args.asr_model == 'quartznet':
        asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    elif args.asr_model == 'parakeet':
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    else:
        raise ValueError(f'Model not implemented: {args.asr_model}')

    perplexity = evaluate.load("perplexity", module_type="metric")
    
    files = sorted(glob(os.path.join(args.input_dir, '**/*.wav'), recursive=True))
    assert batch_size <= len(files)
    asr_results = asr_model.transcribe(files, batch_size=batch_size)
    
    # Select only files with a non-empty transcription
    asr_results = [asr_result.text for asr_result in asr_results if asr_result.text != '']
    valid_pairs = [(idx, transcription) for idx, transcription in enumerate(asr_results) if transcription != '']
    valid_files = {idx: valid_idx for valid_idx, (idx, transcription) in enumerate(valid_pairs)}

    ppl_results = perplexity.compute(model_id='gpt2', batch_size=batch_size, predictions=asr_results, device=device)

    log_ppls = []
    with open(args.output_file, 'w') as f:
        f.write(f'filename,{args.asr_model}_gpt2_log_ppl,{args.asr_model}_text\n')
        for idx, file in enumerate(files):
            audio_id = os.path.join(os.path.basename(os.path.dirname(file)), os.path.basename(file))
            if idx in valid_files.keys():
                valid_idx = valid_files[idx]
                transcription = asr_results[valid_idx]
                log_ppl = np.log(ppl_results['perplexities'][valid_idx])
            else:
                transcription = ''
                log_ppl = np.nan
            f.write(f'{audio_id},{round(log_ppl, 5)},"{transcription.strip()}"\n')
            log_ppls.append(log_ppl)

        print(args.asr_model.upper(), "+ GPT-2 log ppl: ", round(np.nanmean(log_ppls), 3), " ± ", round(np.nanstd(log_ppls), 3))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",   type=str, required=True, help="Input dir name")
    parser.add_argument("--output_file", type=str, required=True, help="Path where generated scores are saved")
    parser.add_argument("--asr_model",   type=str, default="parakeet", 
                                         choices=["parakeet", "quartznet"], help="Name of ASR model")
    parser.add_argument("--batch_size",  type=int, default=16, help="Batch_size for computations")    
    args = parser.parse_args()

    main(args)
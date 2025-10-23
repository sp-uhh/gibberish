"""
Calculate DistillMOS and UTMOSv2 for a directory and put the scores in a csv.

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
import pydantic
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
import torch
import torchaudio
import distillmos
import utmosv2

warnings.simplefilter(action='ignore', category=pydantic.warnings.UnsupportedFieldAttributeWarning)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    distillmos_model = distillmos.ConvTransformerSQAModel()
    distillmos_model.eval().to(device)

    utmosv2_model = utmosv2.create_model(pretrained=True)

    files = sorted(glob(os.path.join(args.input_dir, '**/*.wav'), recursive=True))

    distillmos_scores = []
    for filename in tqdm(files):
        with torch.no_grad():
            x, sr = torchaudio.load(filename)
            x = x[0, None, :].to(device)
            if sr != 16000:
                x = torchaudio.transforms.Resample(sr, 16000)(x)
            distillmos_scores.append(distillmos_model(x).item()) 

    utmosv2_preds = utmosv2_model.predict(input_dir=args.input_dir)
    utmosv2_preds = {x['file_path']: x['predicted_mos'] for x in utmosv2_preds}

    with open(args.output_file, 'w') as f:
        f.write('filename,distillmos,utmosv2\n')
        for filename, distillmos_pred in zip(files, distillmos_scores):
            audio_id = os.path.join(os.path.basename(os.path.dirname(filename)), os.path.basename(filename))
            f.write(f'{audio_id},{round(distillmos_pred, 5)},{round(utmosv2_preds[filename], 5)}\n')

    print("DistillMOS: ", round(np.mean(distillmos_scores), 3), " ± ", round(np.std(distillmos_scores), 3))
    utmosv2_scores = list(utmosv2_preds.values())
    print("UTMOSv2: ", round(np.mean(utmosv2_scores), 3), " ± ", round(np.std(utmosv2_scores), 3))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",   type=str, required=True, help="Input dir name")
    parser.add_argument("--output_file", type=str, required=True, help="Path where generated scores are saved")
    args = parser.parse_args()

    main(args)
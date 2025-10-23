#!/usr/bin/env bash
set -Eeuo pipefail

audio_dir=$1 # directory where the audio files are located
output_file=$2 # output csv path for the scores
l=6 # using layer 6, the one that the clustering method was trained on
ext='.wav'

# Configurations
data_basedir="data" # intermediate stage files and logs will be placed here
hubert_path="facebook/hubert-base-ls960"
km_path="models/hubert/km.bin"
ulm_path="models/ulm/"
token_list="models/ulm/tokens.txt"
nj=1

echo "Audio dir: ${audio_dir}"

base="$(basename "${audio_dir}")"
# If basename starts with '-', replace the *leading* '-' with a safe prefix like 'neg'
if [[ "$base" == -* ]]; then
  base="neg${base#-}"
fi
split="${base}_${l}"

echo "${split}"

data_dir="${data_basedir}/${split}"
mkdir -p "${data_dir}/log"

# 0) Quick input checks
if [ ! -d "${audio_dir}" ]; then
  echo "ERROR: audio_dir does not exist: ${audio_dir}" >&2; exit 1
fi

# 1) Get file list
echo "Getting file list."
python 01a_gen_list.py -a "${audio_dir}" -o "${data_dir}/${split}_file_list" --ext "${ext}" > "${data_dir}/log/log" 2>&1

# 2) Gen TSV
(python 01b_gen_tsv.py -i "${data_dir}/${split}_file_list" -o "${data_dir}/${split}.tsv") &>> "${data_dir}/log/log" || {
  echo "ERROR: 01b_gen_tsv.py failed. See ${data_dir}/log/log" >&2; exit 1; }

# --- FIX TSV root ---
tsv="${data_dir}/${split}.tsv"
tmp="${tsv}.tmp"

# Wait for TSV to exist & be non-empty (up to 30s)
for i in $(seq 1 30); do [ -s "$tsv" ] && break; sleep 1; done
if [ ! -s "$tsv" ]; then echo "ERROR: TSV not created: $tsv" >&2; exit 1; fi

# Normalize root line based on whether paths are abs/rel
first_row_path="$(sed -n '2p' "$tsv" | awk -F'\t' '{print $1}')"
if printf '%s' "$first_row_path" | grep -qE '^/'; then
  { echo "/"; awk 'BEGIN{FS=OFS="\t"} NR>1 { $1 = gensub(/^\/+/, "", "g", $1); print }' "$tsv"; } > "$tmp" && mv "$tmp" "$tsv"
else
  root="$(realpath "$audio_dir")"
  { echo "$root"; tail -n +2 "$tsv"; } > "$tmp" && mv "$tmp" "$tsv"
fi
# --- end fix ---

# 3) Get HuBERT features (GPU)
echo "Getting features."
(python 02a_dump_feature.py --tsv_dir "${data_dir}" \
    --split "${split}" --ckpt_path "${hubert_path}" --layer "${l}" \
    --feat_dir "${data_dir}") &> "${data_dir}/log/dump_features.log" || {
  echo "ERROR: 02a_dump_feature.py failed. See ${data_dir}/log/dump_features.log" >&2; exit 1; }

# 4) Wait for .len to be complete
len_path="${data_dir}/${split}.len"
expected=$(($(wc -l < "$tsv") - 1))  # number of data lines
echo "Waiting for ${len_path} (expect ${expected} lines)..."
for i in $(seq 1 180); do
  if [ -s "$len_path" ] && [ "$(wc -l < "$len_path")" -eq "$expected" ]; then
    echo "Found ${len_path} with ${expected} lines."; break
  fi; sleep 1
done
if [ ! -s "$len_path" ] || [ "$(wc -l < "$len_path")" -ne "$expected" ]; then
  echo "ERROR: ${len_path} missing or incomplete. Check ${data_dir}/log/dump_features.log" >&2; exit 1
fi

# 5) K-means labels (GPU)
echo "Getting k-means labels."
(python 02b_dump_km_label.py --feat_dir "${data_dir}" \
    --split "${split}" --km_path "${km_path}" \
    --lab_dir "${data_dir}" --use_gpu) &> "${data_dir}/log/dump_km_label.log" || {
  echo "ERROR: 02b_dump_km_label failed. See ${data_dir}/log/dump_km_label.log" >&2; exit 1; }

# 6) Build key lists
sed '1d' "${data_dir}/${split}.tsv" \
| awk '{n=split($1, lst, "/"); folder=lst[n-1]; fname=lst[n]; gsub(/\.(wav|flac)/, "", fname); print folder "/" fname }' \
> "${data_dir}/${split}.keys.prompt"

sed '1d' "${data_dir}/${split}.tsv" \
| awk '{n=split($1, lst, "/"); folder=lst[n-1]; fname=lst[n]; gsub(/\.(wav|flac)/, "", fname); print folder "/" fname "," }' \
> "${data_dir}/${split}.keys.csv"

# 7) Sanity checks before composing prompts
km_file="${data_dir}/${split}.km"
keys_file="${data_dir}/${split}.keys.prompt"

if [ ! -s "$km_file" ]; then echo "ERROR: Missing or empty $km_file" >&2; exit 1; fi
if [ ! -s "$keys_file" ]; then echo "ERROR: Missing or empty $keys_file" >&2; exit 1; fi

n_km=$(wc -l < "$km_file")
n_keys=$(wc -l < "$keys_file")
if [ "$n_km" -ne "$n_keys" ]; then
  echo "ERROR: Line count mismatch: km=${n_km}, keys=${n_keys}" >&2
  exit 1
fi
if [ "$n_km" -eq 0 ]; then
  echo "ERROR: No items to score (0 lines in km/keys)" >&2
  exit 1
fi

# Check for empty token sequences (blank .km lines)
if awk 'NF==0' "$km_file" | head -n1 | grep -q .; then
  echo "ERROR: Found empty token sequences in ${km_file}" >&2
  awk 'NF==0 {print "Empty token line at:", NR; exit}' "$km_file" >&2
  exit 1
fi

# 8) Compose prompts with a GUARANTEED TAB delimiter and filter blanks defensively
paste -d $'\t' "${keys_file}" "${km_file}" \
| awk -F'\t' 'NF==2 && $2!=""' \
> "${data_dir}/${split}.txt"

prompts="${data_dir}/${split}.txt"
n_prompts=$(wc -l < "$prompts")
if [ "$n_prompts" -eq 0 ]; then
  echo "ERROR: Prompts file ended up empty after filtering: ${prompts}" >&2
  exit 1
fi

# Optional: verify exactly 2 TAB-separated fields on a sample
if ! head -n 3 "$prompts" | awk -F'\t' 'NF!=2 {exit 1}'; then
  echo "ERROR: Prompts sample not TAB-delimited with 2 fields" >&2
  exit 1
fi

# 9) Calculate perplexity (logs to file, fail clearly if it errors)
batch_size=16
echo "Calculating perplexity on ${n_prompts} utterances..."
(python 03_calc_perplexity.py --prompts "${prompts}" \
    --token_list "${token_list}" \
    --config_file "${ulm_path}/config.yaml" --ckpt_path "${ulm_path}/valid.loss.best.pth" \
    --batch_size "${batch_size}" --out_dir "${data_dir}" -n "${nj}" --ext "${ext}") &> "${data_dir}/log/calc_ppl.log" || {
  echo "ERROR: 03_calc_perplexity failed. See ${data_dir}/log/calc_ppl.log" >&2
  # Surface last lines to console for quick diagnosis:
  tail -n 50 "${data_dir}/log/calc_ppl.log" >&2
  exit 1
}
echo "Copying ${data_dir}/speechlmscore.csv to $output_file"
cp "${data_dir}/speechlmscore.csv" "$output_file"

echo "Perplexity calculation finished. See ${data_dir}/log/calc_ppl.log"
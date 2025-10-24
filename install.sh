#!/bin/bash

set -e
LOG_FILE='./install.log'
exec 3>&1
trap 'echo "Installation script failed at line ${LINENO}. Check ${LOG_FILE} for more details." >&3' ERR
rm -f $LOG_FILE

# This repo uses several publicly available models
# Some requiremnts are simple, some require hacks 

# Start with the simple ones
SIMPLE_REQS='torch torchaudio torchcodec distillmos evaluate nemo_toolkit[asr]'
echo -n 'Installing simple requirements (pip)...'
pip install ${SIMPLE_REQS} >>$LOG_FILE 2>&1
echo "Done."

EXTERNAL_LIBS='./external_libs'

# Textlesslib's requirements.txt is needlessly strict, so we patch it before installing
TEXTLESS_REPO='https://github.com/facebookresearch/textlesslib.git'
TEXTLESS_COMMIT='ba33d669d8284b4f7bfe81e7384e83ab799fe384'

if [[ -d textlesslib ]]; then 
    echo 'Removing existing textlesslib directory.'
    rm -rf textlesslib
fi
echo -n 'Cloning textlesslib...'
{
    git clone ${TEXTLESS_REPO}
    cd textlesslib; git checkout ${TEXTLESS_COMMIT}; cd ..
} >>$LOG_FILE 2>&1 
echo "Done."

echo -n 'Patching textlesslib...'
sed -i '/^numpy==/s/==/>=/; /^numba==/s/==/>=/' ./textlesslib/requirements.txt
echo "Done."

echo -n 'Installing textlesslib (pip)...'
pip install -e ./textlesslib >>$LOG_FILE 2>&1
echo "Done."

# Fairseq is broken on python>=3.11
# See: https://github.com/facebookresearch/fairseq/issues/5012
# We solve this (for now) by installing a fork with a bugfix
PY_MINOR_VERSION=$(python -c 'import sys; print(sys.version_info.minor)')
if [[ $PY_MINOR_VERSION -lt 11 ]]; then
    echo 'Python<3.11 => Using official fairseq repo'
    FAIRSEQ_REPO='https://github.com/facebookresearch/fairseq.git'
else
    echo 'Python>=3.11 => Using forked fairseq repo'
    FAIRSEQ_REPO='https://github.com/liyaodev/fairseq.git@b963eac7a04c539ad59fb1e23277f2ff7ee29e74'
fi

echo -n 'Installing fairseq (pip)...'
pip install 'git+'${FAIRSEQ_REPO} >>$LOG_FILE 2>&1
echo "Done."

# UTMOSv2 needs to be slightly patched to work with our data structure
UTMOSV2_REPO='https://github.com/sarulab-speech/UTMOSv2.git'
UTMOSV2_COMMIT='3608fad976f18a0584e5837afa04fbc628572061'

if [[ -d UTMOSv2 ]]; then 
    echo 'Removing existing UTMOSv2 directory.'
    rm -rf UTMOSv2
fi
echo -n 'Cloning UTMOSv2...'
{
    git clone ${UTMOSV2_REPO}
    cd UTMOSv2; git checkout ${UTMOSV2_COMMIT}; cd ..
} >>$LOG_FILE 2>&1
echo "Done."

echo -n 'Patching UTMOSv2...'
sed -i '174s/\*\.wav/\*\*\/\*\.wav/' UTMOSv2/utmosv2/_core/model/_common.py
echo "Done."

echo -n 'Installing UTMOSv2 (pip)...'
pip install -e ./UTMOSv2 >>$LOG_FILE 2>&1
echo "Done."

# SpeechLMScore needs a modified evaluation script
SPEECHLMSCORE_REPO='https://github.com/soumimaiti/speechlmscore_tool.git'
SPEECHLMSCORE_COMMIT='ffcfd7f3c1a3d14a2d1e9898c12753578c325347'

if [[ -d speechlmscore_tool ]]; then 
    echo 'Removing existing speechlmscore_tool directory.'
    rm -rf speechlmscore_tool
fi
echo -n 'Cloning SpeechLMScore...'
{
    git clone ${SPEECHLMSCORE_REPO} 
    cd speechlmscore_tool; git checkout ${SPEECHLMSCORE_COMMIT}; cd ..
} >>$LOG_FILE 2>&1
echo "Done."

echo -n 'Patching and installing SpeechLMScore...'
{
    cd speechlmscore_tool
    pip install -r requirements.txt
    ./download_pretrained_models.sh
    cp ../speechlmscore_hack/* .
    cd ..
} >>$LOG_FILE 2>&1

echo "Done."
echo
echo "Installation complete."
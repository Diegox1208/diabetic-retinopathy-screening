#!/usr/bin/env bash
#### created by Diego G. Salas
#### adapted from the Colab notebook notebooks/RD_02_APTOS_EyePACS_v3_checkpoint.ipynb
#
# Runs the whole pipeline end to end.
#   bash run.sh                               -> synthetic test dataset in tests/, CPU, about a minute
#   CONFIG=config/config.sh bash run.sh       -> APTOS 2019 + EyePACS, training needs a GPU
#   DRY_RUN=1 bash run.sh                     -> print every command, run nothing
set -euo pipefail

#---- ##-- Configuration ----
export CONFIG="${CONFIG:-tests/config.test.sh}"
export DRY_RUN="${DRY_RUN:-0}"
source "${CONFIG}"

##### Set variables
manifest="${workDir}/manifest.tsv"
manifestPrep="${workDir}/manifest_${imageSize}px.tsv"
splitFile="${workDir}/split_seed${seed}.tsv"
prefix="${resultsDir}/${version}"

##### Build, log, run one stage
run_stage() {
  local stage="$1"; shift
  local cmd="$*"
  echo "[${stage}] ${cmd}"
  echo -e "${version}\t${stage}\t${cmd}\t${date_}" >> "${logFile}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    eval "time ${cmd}"
  fi
}

#---- ##-- Stages ----
bash workflow/01_download.sh

run_stage 02_manifest python workflow/02_manifest.py \
  --raw "${rawDir}" --info "${infoFile}" \
  --out "${manifest}" --summary "${resultsDir}/dataset_summary.tsv"

run_stage 03_preprocess python workflow/03_preprocess.py \
  --manifest "${manifest}" --out-dir "${workDir}/prep" --out "${manifestPrep}" \
  --size "${imageSize}" --clahe "${useClahe}" --workers "${threads}"

run_stage 04_split python workflow/04_split.py \
  --manifest "${manifestPrep}" --out "${splitFile}" --summary "${resultsDir}/split_summary.tsv" \
  --test-folds "${testFolds}" --val-folds "${valFolds}" --seed "${seed}"

run_stage 05_train python workflow/05_train.py \
  --split "${splitFile}" --model-dir "${modelDir}" --history "${prefix}_history.tsv" \
  --version "${version}" --architecture "${architecture}" --pretrained "${pretrained}" \
  --batch-size "${batchSize}" --epochs "${epochs}" --lr "${learningRate}" --seed "${seed}" \
  --num-workers "${numWorkers}" --device "${device}"

run_stage 06_evaluate python workflow/06_evaluate.py \
  --split "${splitFile}" --model-dir "${modelDir}" --prefix "${prefix}" \
  --version "${version}" --architecture "${architecture}" \
  --target-sensitivity "${targetSensitivity}" --batch-size "${batchSize}" \
  --bootstrap "${bootstrapN}" --num-workers "${numWorkers}" --device "${device}"

echo "done: results in ${resultsDir}, log in ${logFile}"

#!/usr/bin/env bash
#### created by Diego G. Salas
#### adapted from notebooks/RD_02_APTOS_EyePACS_v3_checkpoint.ipynb, cells "Paso 1"
#
# Stage 01 downloads each dataset in the info file from Kaggle into ${rawDir}/<name>.
# Needs ~/.kaggle/kaggle.json and the APTOS competition rules accepted on kaggle.com.
set -euo pipefail

#---- ##-- Configuration ----
source "${CONFIG:-tests/config.test.sh}"
DRY_RUN="${DRY_RUN:-0}"

##### Set variables
stage="01_download"

#---- ##-- Loop over datasets ----
while IFS=$'\t' read -r dataset name accession source; do
  [[ "${dataset}" == "dataset" || -z "${dataset}" ]] && continue      # header or blank line

  outDir="${rawDir}/${name}"
  if [[ -d "${outDir}" ]] && [[ -n "$(ls -A "${outDir}")" ]]; then
    echo "[${stage}] ${outDir} already has files, skipping ${name}" >&2
    continue
  fi
  if [[ "${accession}" == "synthetic" ]]; then
    echo "[${stage}] ${name} is synthetic: run python tests/make_test_data.py first" >&2
    continue
  fi

  ##### Build, log, run
  case "${accession}" in
    competitions/*)
      slug="${accession#competitions/}"
      cmd="kaggle competitions download -c ${slug} -p ${outDir} && unzip -q -o ${outDir}/${slug}.zip 'train_images/*' train.csv -d ${outDir} && rm -f ${outDir}/${slug}.zip" ;;
    datasets/*)
      cmd="kaggle datasets download -d ${accession#datasets/} -p ${outDir} --unzip" ;;
    *)
      echo "[${stage}] unknown accession ${accession}, skipping ${name}" >&2
      continue ;;
  esac
  echo "[${stage}] ${cmd}"
  echo -e "${name}\t${stage}\t${cmd}\t${date_}" >> "${logFile}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${outDir}"
    eval "time ${cmd}"
  fi
done < "${infoFile}"

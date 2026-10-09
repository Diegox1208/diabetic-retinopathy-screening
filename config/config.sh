#### created by Diego G. Salas
#### Full-data configuration (APTOS 2019 + EyePACS). Sourced by run.sh and by every workflow script.
#### All paths are relative to the repository root. Never edit a workflow script to change a path.
#### These values reproduce the Colab run model_v3. Training needs a GPU.

##### Set variables
repoDir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataDir="${repoDir}/data"
rawDir="${dataDir}/raw"
workDir="${dataDir}/work"
modelDir="${repoDir}/models"
resultsDir="${repoDir}/results"
logDir="${repoDir}/logs"
infoFile="${repoDir}/config/datasets.tsv"

##### Model parameters
version="model_v3"
architecture="efficientnet_b0"
pretrained=1                 # ImageNet weights from timm
imageSize=384
batchSize=32
epochs=12
learningRate=3e-4
seed=42
useClahe=0
targetSensitivity=0.90       # threshold chosen on validation to reach this sensitivity
testFolds=7                  # StratifiedGroupKFold, 1 of 7 folds is test (~14%)
valFolds=6                   # then 1 of 6 remaining folds is validation (~14%)
bootstrapN=1000
device="auto"                # cuda if available, else cpu

##### Global parameters
threads=8                    # preprocessing workers
numWorkers=2                 # DataLoader workers

##### Timestamp and log
day="$(date +%Y%m%d)"
time_="$(date +%H%M%S)"
date_="${day}_${time_}"
logFile="${logDir}/run.tsv"

mkdir -p "${resultsDir}" "${logDir}" "${modelDir}" "${workDir}"

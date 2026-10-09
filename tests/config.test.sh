#### created by Diego G. Salas
#### Test configuration with the same variables as config/config.sh, pointing at tests/data (< 1 MB, synthetic).
#### Small images, two epochs, no pretrained weights. It checks that every stage runs, not that the model learns.

##### Set variables
repoDir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataDir="${repoDir}/tests/data"
rawDir="${dataDir}/raw"
resultsDir="${repoDir}/tests/results"
workDir="${resultsDir}/work"
modelDir="${resultsDir}/models"
logDir="${repoDir}/logs"
infoFile="${repoDir}/tests/datasets.tsv"

##### Model parameters
version="test_v0"
architecture="efficientnet_b0"
pretrained=0
imageSize=64
batchSize=8
epochs=2
learningRate=3e-4
seed=42
useClahe=0
targetSensitivity=0.90
testFolds=7
valFolds=6
bootstrapN=200
device="cpu"

##### Global parameters
threads=2
numWorkers=0

##### Timestamp and log
day="$(date +%Y%m%d)"
time_="$(date +%H%M%S)"
date_="${day}_${time_}"
logFile="${logDir}/test.tsv"

mkdir -p "${resultsDir}" "${logDir}" "${modelDir}" "${workDir}"

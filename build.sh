set -euo pipefail

git add -A ./
git commit -m f
git push origin main
ssh -i ~/aosp.pem gitlab-runner@172.202.112.196 'cd /home/gitlab-runner/kyutai/pocket-tts-onnx-export && git pull origin main && source ../venv/bin/activate && pip3 install onnxconverter_common && python3 export.py'

scp -i ~/aosp.pem gitlab-runner@172.202.112.196:/home/gitlab-runner/kyutai/pocket-tts-onnx-export/onnx/mimi_encoder_fp16.onnx ~/Downloads/mimi_encoder.onnx
scp -i ~/aosp.pem gitlab-runner@172.202.112.196:/home/gitlab-runner/kyutai/pocket-tts-onnx-export/onnx/mimi_decoder_fp16.onnx ~/Downloads/mimi_decoder.onnx

adb -s WT02082500015 push ~/Downloads/mimi_encoder.onnx /data/mimi_encoder.onnx
adb -s WT02082500015 push ~/Downloads/mimi_decoder.onnx /data/mimi_decoder.onnx
set -euo pipefail

git add -A ./
git commit -m f
git push origin main
ssh -i ~/aosp.pem gitlab-runner@172.202.112.196 'cd /home/gitlab-runner/kyutai/pocket-tts-onnx-export && git pull origin main && source ../venv/bin/activate && python3 export.py'

rm -f ./output.pcm
rm -f ./output.wav
adb -s WT02082500015 pull /data/output.pcm ./output.pcm
ffmpeg -y -f s16le -ar 24000 -ac 1 -i ./output.pcm output.wav

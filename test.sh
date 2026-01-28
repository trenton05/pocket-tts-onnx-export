adb -s WT02082500015 pull /data/data/com.is360mobile.walt/cache/music.pcm ./music.pcm
ffmpeg -y -f s16le -ar 24000 -ac 1 -i ./music.pcm music.wav

adb -s WT02082500015 pull /data/data/com.is360mobile.walt/cache/readout.pcm ./readout.pcm
ffmpeg -y -f s16le -ar 24000 -ac 1 -i ./readout.pcm readout.wav


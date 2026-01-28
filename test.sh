adb -s WT02082500015 pull /data/data/com.is360mobile.walt/cache/output.pcm ./output.pcm
ffmpeg -f s16le -ar 24000 -ac 1 -i ./output.pcm output.wav


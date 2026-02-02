# Copyright (C) 2016 Verizon. All Rights Reserved.
#!/bin/bash

set -euo pipefail

export ANDROID_NDK="~/Library/Android/sdk/ndk/28.2.13676358"

rm -fR build
mkdir -p build
cd build

export LD_LIBRARY_PATH=/data

cmake ../ -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_NDK=$ANDROID_NDK \
    -DANDROID_ABI="arm64-v8a" \

make -j8

cd ..

adb push build/mimi /data/mimi
adb shell "export LD_LIBRARY_PATH=/data && /data/mimi"

rm -f ./output.pcm
rm -f ./output.wav
adb -s WT02082500015 pull /data/output.pcm ./output.pcm
ffmpeg -y -f s16le -ar 24000 -ac 1 -i ./output.pcm output.wav

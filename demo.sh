#!/bin/bash
echo "demo"
python demo.py
mkdir -p normal
mkdir -p attenuation
rm -f normal/* attenuation/*

for i in $(seq 0 121); do

  echo "convert frame $i"
  n=$(echo $i |xargs printf "%08d")

  convert -scale 400 -antialias  \
    demo_normal/demo_normal${n}.png \
    gen_proc_normal/gen_proc_normal${n}.png \
    gen_mod_normal/gen_mod_normal${n}.png \
    -append normal/normal${n}.png

  convert -scale 400 -antialias  \
    demo_attenuation/demo_attenuation${n}.png \
    gen_proc_attenuation/gen_proc_attenuation${n}.png \
    gen_mod_attenuation/gen_mod_attenuation${n}.png \
    -append attenuation/attenuation${n}.png
done
echo "videos"
convert -loop 0 -delay 5 normal/* normal.gif
convert -loop 0 -delay 5 attenuation/* attenuation.gif
echo "clear"
rm -r gen_* demo_* attenuation normal

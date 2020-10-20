#!/bin/bash

set -e

MAIN_DIR="$(dirname $(realpath $0) | sed -e 's/\utils//')"

cd $MAIN_DIR
cd src

TYPE1=normal
TYPE2=large
TYPE3=still

echo "demo"
python demo.py
mkdir -p ${TYPE1} ${TYPE2} ${TYPE3}
rm -f ${TYPE1}/* ${TYPE2}/* ${TYPE3}/*
for i in $(seq 0 121); do

  echo "convert frame $i"
  n=$(echo $i |xargs printf "%08d")

  for type in $TYPE{1,2,3}; do
    convert -scale 400 -antialias  \
    demo_${type}/demo_${type}${n}.png \
    gen_proc_${type}/gen_proc_${type}${n}.png \
    gen_mod_${type}/gen_mod_${type}${n}.png \
    prederr_${type}/prederr_${type}${n}.png \
    -append ${type}/${type}${n}.png
  done

done
echo "videos"
for type in $TYPE{1,2,3}; do
  convert -loop 0 -delay 20 ${type}/${type}* ${MAIN_DIR}/pics/${type}.gif
done

echo "clear"
rm -r gen_* demo_* $TYPE{1,2,3} prederr_*

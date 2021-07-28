#!/bin/bash

set -e
MAIN_DIR="$(dirname $(realpath $0) | sed -e 's/\utils//')"

function demo() {

  cd $MAIN_DIR
  cd src

  TYPE=$1

  echo "demo"
  python demo.py -t $TYPE
  mkdir -p ${TYPE}
  rm -f ${TYPE}/*
  N=$(echo "$(ls demo_${TYPE}/demo_*| wc -l) -1"| bc)
  for i in $(seq 0 $N); do
    echo "convert frame $i"
    n=$(echo $i |xargs printf "%08d")
      mogrify -rotate 180 demo_${TYPE}/demo_${TYPE}${n}.png
      convert -scale 400 -antialias  \
      demo_${TYPE}/demo_${TYPE}${n}.png \
      gen_proc_${TYPE}/gen_proc_${TYPE}${n}.png \
      gen_mod_${TYPE}/gen_mod_${TYPE}${n}.png \
      prederr_${TYPE}/prederr_${TYPE}${n}.png \
      -append ${TYPE}/${TYPE}${n}.png
  done

  # screenshots
  cp ${TYPE}/${TYPE}*${N}.png ${MAIN_DIR}/pics/${TYPE}.png

  echo "videos"
  convert -loop 0 -delay 10 ${TYPE}/${TYPE}* ${MAIN_DIR}/pics/${TYPE}.gif

  echo "clear"
  rm -r gen_* demo_* $TYPE prederr_*
}

demo still
demo normal
demo large

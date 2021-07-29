# Copyright (c) 2021 Francesco Mannella, Federico Maggiore
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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

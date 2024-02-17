#!/bin/bash

file=output/profiling/times_$(git log --oneline | head -n 1 | awk '{ print $1 }')_autogen.csv
version=$1

echo Preparing to test $version

if [ ! -d output/profiling ]; then
  mkdir -p output/profiling
fi

echo Building...

cmake --build build/$version --parallel -t performance

echo Testing...
echo

echo -e "scene count,$version simulation time,$version total time" > $file
echo -e "scene count\t$version simulation time\t$version total time"

for sim_count in {10..82}
  do
    sim_count=$(calc 'int(2^('$sim_count'/5))' | awk '{print $1}')
    ./build/$version/tests/performance -m 0 -s $sim_count \
      | grep "Kernel took:" \
      | awk '{printf("'$sim_count',%s,%s\n",$3,$6)}' \
      | tee -a $file | awk -F ',' '{print $1,"\t",$2,"\t",$3}' \
      || (echo "Error! $sim_count"; exit 1)
  done

echo
echo Done

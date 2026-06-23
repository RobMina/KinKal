#!/usr/bin/env bash
set -euo pipefail

src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${src_dir}/build"
kinkal_src_dir="${src_dir}/../.."
kinkal_build_dir="${src_dir}/../../KinKal_build"
mkdir -p "${build_dir}"

cxx="${CXX:-c++}"
out="${build_dir}/ThickSlabG4"
g4_libs="$(geant4-config --libs)"
root_libs="$(root-config --libs)"
rpaths=()
for token in ${g4_libs}; do
  if [[ "${token}" == -L* ]]; then
    rpaths+=("-Wl,-rpath,${token#-L}")
  fi
done
rpaths+=("-Wl,-rpath,${kinkal_build_dir}/lib")
kinkal_libs=(
  "${kinkal_build_dir}/lib/libKinKal_Detector.so.3.6.0"
  "${kinkal_build_dir}/lib/libKinKal_Fit.so.3.6.0"
  "${kinkal_build_dir}/lib/libKinKal_Trajectory.so.3.6.0"
  "${kinkal_build_dir}/lib/libKinKal_Geometry.so.3.6.0"
  "${kinkal_build_dir}/lib/libKinKal_MatEnv.so.3.6.0"
  "${kinkal_build_dir}/lib/libKinKal_General.so.3.6.0"
)

${cxx} -std=c++17 -O3 -Wall -Wextra \
  -I"${kinkal_src_dir}" \
  "${src_dir}/ThickSlabG4.cc" \
  $(geant4-config --cflags) \
  $(root-config --cflags) \
  "${kinkal_libs[@]}" \
  ${g4_libs} \
  ${root_libs} \
  "${rpaths[@]}" \
  -o "${out}"

echo "${out}"

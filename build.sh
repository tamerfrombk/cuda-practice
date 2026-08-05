#!/usr/bin/env bash
set -euo pipefail

src=${1:?usage: ./build.sh <file.cu> [extra nvcc args...]}
shift

mkdir -p bin
nvcc12 "$src" -o "bin/$(basename "${src%.cu}")" "$@"

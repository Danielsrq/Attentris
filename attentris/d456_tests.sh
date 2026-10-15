#!/bin/sh
# depth 4, 5, 6 (beam 8) tests of the d=128 dagger5b-ft model, one after the other (d3_tests.sh)
DIR=$(cd "$(dirname "$0")" && pwd)
for dp in 4 5 6; do MODEL=v51_d128_dagger5b_ft DEPTH=$dp sh "$DIR/d3_tests.sh"; done

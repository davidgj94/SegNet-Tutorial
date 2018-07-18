#!/bin/bash
solver_dir=$1
exp_name=$2

sed -ri "s|net: \"/home/davidgj/projects_v2/SegNet-Tutorial/Models/.*/train.prototxt\"|net: \"${solver_dir}/train.prototxt\"|" "${solver_dir}"/solver.prototxt
sed -ri "s|(snapshot_prefix: \"/home/davidgj/projects_v2/SegNet-Tutorial/Models/Training/).*(/snapshot\")|\1${exp_name}\2|" "${solver_dir}"/solver.prototxt

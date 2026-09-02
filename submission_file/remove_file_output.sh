#!/bin/bash
Type=${1}

echo $"id main db: ${Type}" 
echo "Path is ${PWD}"
rm -rf rm -rf output_bash/*_"${Type}" 
echo $"folder out and err for ${Type} are removed"

USAGE="usage: myscript type, with type 0-5, where:\n 0 - no shear, k=3.1;\n 1 - no shear, k=5.0;\n 2 - no shear, non-linear;\n 3 - yes shear, k=3.1;\n 4 - yes shear, k=5.0;\n 5 - yes shear, non-linear"
USAGE2="Type must be an integer between 0-5"
Type=${1:?$(printf '%b' "$USAGE")}

if [ "$Type" -gt 5 ] ||  [ "$Type" -lt 0 ]; then
    printf '%b\n' "$USAGE2" >&2
    exit 1
fi 

echo $"id main db: ${Type}" 
rm -rf output_bash/*_${Type} 
echo $"folder out and err for ${Type} are removed"
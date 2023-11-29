#!/bin/bash

# Check if all three arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <filename> <find_string> <replace_string>"
    exit 1
fi

filename="$1"
find_string="$2"
replace_string="$3"

# Check if the file exists
if [ ! -f "$filename" ]; then
    echo "Error: File '$filename' not found."
    exit 1
fi

# Use sed to replace all occurrences of find_string with replace_string in the file
sed -i "s/$find_string/$replace_string/g" "$filename"

echo "Replacement complete in '$filename'."
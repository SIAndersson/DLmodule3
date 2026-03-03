#!/bin/bash

echo "=== Finding files in directories starting with 'unet_' ==="
find model_checkpoints -type f -path "*/unet_*/*" -print

file_count=$(find model_checkpoints -type f -path "*/unet_*/*" | wc -l)
echo -e "\nFound $file_count files in unet_* directories"

if [ $file_count -gt 0 ]; then
    echo -e "\n=== Creating tar.xz archive with maximum compression ==="
    find model_checkpoints -type f -path "*/unet_*/*" -print0 | \
    XZ_OPT=-9e tar -cJvf unet_checkpoints.tar.xz --null -T -
    
    echo -e "\n=== Archive Information ==="
    ls -lh unet_checkpoints.tar.xz
    
    echo -e "\n=== Archive Contents ==="
    echo "Directories in archive:"
    tar -tJf unet_checkpoints.tar.xz | sed 's|/[^/]*$||' | sort -u
    
    echo -e "\nAll files in archive:"
    tar -tJvf unet_checkpoints.tar.xz
    
    echo -e "\n=== Testing Archive Integrity ==="
    tar -tJf unet_checkpoints.tar.xz > /dev/null && echo "Archive integrity: OK" || echo "Archive integrity: FAILED"
else
    echo "No files found in unet_* directories"
fi

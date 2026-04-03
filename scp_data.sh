#! /bin/bash

echo "Copy start"

TARGET_HOST="user@192.168.233.168"
# TARGET_HOST="user@192.168.70.197"
SSH_CTL_PATH="$HOME/.ssh/cm-%r@%h:%p"
SSH_OPTS="-o ControlMaster=auto -o ControlPersist=10m -o ControlPath=${SSH_CTL_PATH}"

while IFS= read -r name; do
  rsync -avz \
    -e "ssh ${SSH_OPTS}" \
    --rsync-path="mkdir -p /data/suitesparse_collection/${name} && rsync" \
    /data/suitesparse_collection/${name}/${name}.mtx \
    "${TARGET_HOST}:/data/suitesparse_collection/${name}/"
done < /data/lsl/SparseOps/testdatasets.txt

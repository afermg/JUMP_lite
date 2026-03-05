# Setup Nahual servers

# ViT-based models
for i in {0..20}; do
	for model in morphem openphenom; do
		session_name="${model}_${i}"
		ipc_addr="ipc:///tmp/${session_name}.ipc"
		echo "Starting ${model} instance in screen session '${session_name}'"
		screen -S "${session_name}" -dm bash -c "nix run github:afermg/nahual_vit#${model} '${ipc_addr}'"
	done
done

# Subcell
for i in {0..20}; do
	session_name="subcell_${i}"
	ipc_addr="ipc:///tmp/subcell_${i}.ipc"
	echo "Starting subcell instance in screen session '${session_name}'"
	screen -S "${session_name}" -dm bash -c "nix run github:afermg/SubCellPortable ${ipc_addr}"
done

# Dinov2 Random
for i in {0..20}; do
	session_name="dinov2_${i}"
	ipc_addr="ipc:///tmp/dinov2_random_${i}.ipc"
	echo "Starting dinov2 instance in screen session '${session_name}'"
	# Temporary commit, until cache expires
	screen -S "${session_name}" -dm bash -c "nix run github:afermg/dinov2 ${ipc_addr}"
done

# Dinov2
for i in {0..20}; do
	session_name="dinov2_${i}"
	ipc_addr="ipc:///tmp/dinov2_${i}.ipc"
	echo "Starting dinov2 instance in screen session '${session_name}'"
	# Temporary commit, until cache expires
	screen -S "${session_name}" -dm bash -c "nix run github:afermg/dinov2 ${ipc_addr}"
done

echo "All instances started in detached screen sessions."
echo "Use 'screen -ls' to list sessions and 'screen -r \$MODEL_\$InstanceID' to attach one in particular."
echo "To kill them all, run: screen -ls | awk -F'.' '/\S+_[0-9]/ {print $1}' | xargs kill"

# Run aliby that uses servers for embeddings
# python aliby_featurize.py

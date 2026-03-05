for i in {0..8}; do
	# for model in morphem openphenom; do
	session_name="cellpose_${i}"
	ipc_addr="ipc:///tmp/${session_name}.ipc"
	echo "Starting ${model} instance in screen session '${session_name}'"
	# screen -S "${session_name}" -dm bash -c "nix run github:afermg/cellpose '${ipc_addr}'"
	screen -S "${session_name}" -dm bash -c "nix run github:afermg/cellpose/2146b5ee3b2c7eb2c826efe7a24b3b289432500b '${ipc_addr}'"
done

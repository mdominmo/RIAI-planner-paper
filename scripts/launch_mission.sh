#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

trap handle_exit EXIT
function handle_exit() {
    echo "Exiting simulation."
    echo "${PIDS[@]}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid"
        fi
    done 
    exit 0
}

source ../install/setup.bash

ros2 run mission mission_node mission_node 

echo "Setting up the simulation environment..."
PIDS=()
conf_file="riai_planner_paper_world.yaml"

export PX4_GZ_WORLD=$(yq '.world.type' "$conf_file")
export GZ_SIM_RESOURCE_PATH="${HOME}/repositories/ai_safe/ai-safe-sim/gz_assets/models/:${HOME}/repositories/ai_safe/ai-safe-sim/PX4-Autopilot/Tools/simulation/gz/models/"
PX4_FOLDER="${HOME}/repositories/ai_safe/ai-safe-sim/PX4-Autopilot"
declare -A swarm_pose

while IFS=": " read -r key value; do
    swarm_pose["$key"]=$value
done < <(yq '.UAS.swarm_pose | to_entries | map([.key, .value] | join(": ")) | .[]' "$conf_file")
y_0=${swarm_pose["y"]}

declare -A model
vehicle_instance=1
uavs=$(yq '.UAS.models[].name' "$conf_file")

echo "starting gz server..."
run_cmd "gz sim -r $PX4_GZ_WORLD.sdf"

MODE=$1
if [ "$MODE" != "-e" ]; then
    
    sleep 10
    for uav in $uavs; do
        while IFS=": " read -r key value; do
            model["$key"]=$value
        done < <(yq ".UAS.models[] | select(.name == \"$uav\") | to_entries | map([.key, .value] | join(\": \")) | .[]" "$conf_file")

        export PX4_SIM_MODEL="${model["name"]}"
        export PX4_SYS_AUTOSTART=${model["autostart"]}

        for ((i = 1; i <= model["number"]; i++)); do
            camera_topic="/world/${PX4_GZ_WORLD}/model/${PX4_SIM_MODEL}_${vehicle_instance}/link/mono_cam/base_link/sensor/camera/image"
            camera_topics="$camera_topics $camera_topic"
        
            swarm_pose["y"]=$((y_0 - vehicle_instance * 2))
            export PX4_UXRCE_DDS_NS="px4_${vehicle_instance}"
            export PX4_GZ_MODEL_POSE="${swarm_pose["x"]},${swarm_pose["y"]},${swarm_pose["z"]},${swarm_pose["R"]},${swarm_pose["P"]},${swarm_pose["Y"]}"

            run_cmd "${PX4_FOLDER}/build/px4_sitl_default/bin/px4 -i $vehicle_instance"
            sleep 5

            vehicle_instance=$((vehicle_instance+=1))
        done
    done

    run_cmd "ros2 run ros_gz_image image_bridge $camera_topics"
    sleep 5

    run_cmd "MicroXRCEAgent udp4 -p 8888"
    sleep 8
fi

echo "Simulation started."
while true; do
    read -n 1 -s key
    case "$key" in
        q)
            echo "Exitting simulator"
            handle_exit
            break
            ;;
        *)
            ;;
    esac
done


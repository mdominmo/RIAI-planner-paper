#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../install/setup.bash

cleanup() {
    echo "CTRL+C detected. Killing launch and all child processes..."
    if [[ -n "$LAUNCH_PID" ]]; then
        pkill -9 -P $LAUNCH_PID 2>/dev/null || true
        kill -9 $LAUNCH_PID 2>/dev/null || true
    fi
    if [[ -n "$SIMULATOR_PID" ]]; then
        # Enviar 'q' al simulador para que haga su propia limpieza
        echo "q" > /proc/$SIMULATOR_PID/fd/0 2>/dev/null || true
        sleep 2
        kill -9 $SIMULATOR_PID 2>/dev/null || true
    fi
    pkill -9 -f "px4_gz_positioning_node" 2>/dev/null || true
    pkill -9 -f "tracking_node" 2>/dev/null || true
    pkill -9 -f "mission_node" 2>/dev/null || true
    pkill -9 -f "aruco_detector_node" 2>/dev/null || true
    pkill -9 -f "fusion_node" 2>/dev/null || true
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f "MicroXRCEAgent" 2>/dev/null || true
    pkill -9 -f "px4" 2>/dev/null || true
    pkill -9 -f "launch_simulato" 2>/dev/null || true 
    pkill -9 -f "image_bridge" 2>/dev/null || true 
    pkill -9 -f "ros2" 2>/dev/null || true 
    exit 1
}
trap cleanup SIGINT SIGTERM

kill_simulator() {
    echo "Killing simulator..."
    if [[ -n "$SIMULATOR_PID" ]] && kill -0 $SIMULATOR_PID 2>/dev/null; then
        kill -TERM $SIMULATOR_PID 2>/dev/null || true
        sleep 2
        kill -9 $SIMULATOR_PID 2>/dev/null || true
    fi
    # Matar procesos del simulador
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f "MicroXRCEAgent" 2>/dev/null || true
    pkill -9 -f "px4" 2>/dev/null || true
    pkill -9 -f "ros_gz_image" 2>/dev/null || true
}

wait_for_simulator() {
    echo "Waiting for simulator to be ready..."
    local max_wait=60
    local elapsed=0
    
    sleep 30
    while [ $elapsed -lt $max_wait ]; do
        if pgrep -f "gz sim" > /dev/null && pgrep -f "MicroXRCEAgent" > /dev/null; then
            echo "Simulator ready!"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    
    echo "WARNING: Simulator did not start properly"
    return 1
}

# Simulation parameters
#_______________________________________________________________
PLAN_TYPE=("1" "2" "3") # RRT RRT* RRT-HUNGARIAN RANDOM
NUM_VEHICLES=("5" "6" "7" "8")
N_POINTS=500
MISSION_FRAME="[108.28299713134766,-94.181564331054688,15.0]"
MISSION_RADIUS=("4.0" "15.0" "6.0")
MISSION_HEIGHT=("5.0" "15.0" "7.0")
STEP_SIZE=("0.5" "1.0" "2.0")
N_STEPS=("2000 3000")
SPACE_COEF=(".5" ".8" "1.0")
TIME_COEF=(".5" ".2" ".0")
AVG_SPEED=("1.0" "2.0")
SPATIAL_TOL=("1.5")
TIME_TOL=("100")
CYLINDER_HEIGHT=3.0
CYLINDER_RADIUS=1.2
MAX_ATTEMPS=5
MAX_MINUTES=120
NUM_SIMULATIONS=5
#_______________________________________________________________

DRY_EXECUTION=false
LAUNCH_FILE="riai_launch riai.launch.py"
SIMULATOR_SCRIPT="./launch_simulator.sh"

for vehicles in "${NUM_VEHICLES[@]}"; do
    for plan in "${PLAN_TYPE[@]}"; do
        for spatial_tol in "${SPATIAL_TOL[@]}"; do
            for ((n=0; n<NUM_SIMULATIONS; n++)); do
                echo "=============================="
                echo "Starting simulation $n of $NUM_SIMULATIONS..."
                echo "Num vehicles: $vehicles"
                echo "Plan type: $plan"
                echo "Spatial tolerance: $spatial_tol"
                echo "=============================="
                
                if [ $DRY_EXECUTION == false ]; then
                    attempt=1
                    success=false
                    while [ $attempt -le $MAX_ATTEMPS ] && [ $success == false ]; do
                        echo "Attempt $attempt of $MAX_ATTEMPS..."
                        echo "Launching simulator..."
                        $SIMULATOR_SCRIPT $vehicles &
                        SIMULATOR_PID=$!
                        echo "Simulator PID: $SIMULATOR_PID"
                        if ! wait_for_simulator; then
                            echo "Failed to start simulator. Retrying..."
                            kill_simulator
                            sleep 3
                            attempt=$((attempt + 1))
                            continue
                        fi
                        sleep 5  
                        echo "Launching mission..."
                        ros2 launch $LAUNCH_FILE \
                            num_vehicles:="${vehicles}" \
                            plan_type:="${plan}" \
                            spatial_tol:="${spatial_tol}" \
                            mission_radius:="${MISSION_RADIUS[1]}" \
                            mission_height:="${MISSION_HEIGHT[1]}" &
                        LAUNCH_PID=$!
                        sleep 2
                        elapsed=0
                        timeout_seconds=$((MAX_MINUTES * 60))
                        while [ $elapsed -lt $timeout_seconds ]; do
                            if ! ros2 node list 2>/dev/null | grep -q "/mission_1"; then
                                echo "Mission node finished successfully!"
                                success=true
                                break
                            fi
                            sleep 2
                            elapsed=$((elapsed + 2))
                        done
                        if [ $elapsed -ge $timeout_seconds ]; then
                            echo "TIMEOUT reached (${MAX_MINUTES} min). Killing processes..."
                        fi
                        echo "Killing mission processes..."
                        pkill -9 -P $LAUNCH_PID 2>/dev/null || true
                        kill -9 $LAUNCH_PID 2>/dev/null || true
                        pkill -9 -f "px4_gz_positioning_node" 2>/dev/null || true
                        pkill -9 -f "tracking_node" 2>/dev/null || true
                        pkill -9 -f "mission_node" 2>/dev/null || true
                        pkill -9 -f "aruco_detector_node" 2>/dev/null || true
                        pkill -9 -f "fusion_node" 2>/dev/null || true
                        wait $LAUNCH_PID 2>/dev/null || true
                        kill_simulator
                        wait $SIMULATOR_PID 2>/dev/null || true
                        echo "Waiting 5 seconds before next attempt..."
                        sleep 5
                        if [ $success == false ]; then
                            echo "Attempt $attempt failed. Retrying..."
                            attempt=$((attempt + 1))
                        else
                            echo "Simulation completed successfully!"
                        fi
                    done
                    if [ $success == false ]; then
                        echo "WARNING: Simulation failed after $MAX_ATTEMPS attempts!"
                    fi
                fi
            done 
        done
    done
done
echo "Batch execution finished."
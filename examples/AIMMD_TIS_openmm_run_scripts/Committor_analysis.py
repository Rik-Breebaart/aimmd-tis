import os
import sys
import numpy as np
from pathlib import Path
from multiprocessing import Process
from simtk import unit
import openpathsampling as paths
from openpathsampling.experimental.storage import Storage, monkey_patch_all
import openpathsampling.engines.openmm as ops_openmm
import time

# Patch OPS to use experimental storage features
paths = monkey_patch_all(paths)
# current_directory = os.path.dirname(os.path.abspath(os.getcwd()))
current_directory = "/home/rbreeba/Projects/-RE-TIS-AIMMD/TIS_AIMMD_biosystems/Host-Guest-System-NPT"
# Get the current directory and add it to the system path
sys.path.append(current_directory)

# Get the parent and grandparent directories and add them to the system path
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
parent_parent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
sys.path.append(parent_directory)
sys.path.append(parent_parent_directory)


from examples.AIMMD_TIS_openmm_run_scripts.HostGuest.setup_utilities import TPS_setup, create_parser, global_arguments

 
def run_committor_analysis_for_frame(frame_index, input_path: Path = None, TPS_config_path=None, n_shots=None, traj_path=None, output_path=None, 
                             system_resource_directory=None):
    # Load setup
    print("Process committor analysis frame {}: started".format(frame_index))
    TPS_config_path = Path(input_path/ TPS_config_path).with_suffix(".json")
    # Setup TPS utilities
    TPS_utils = TPS_setup(TPS_config_path, print_config=True, resource_directory=system_resource_directory)

    # Set up engine and beta
    engine = TPS_utils.md_engine
    if hasattr(engine, 'integrator') and callable(getattr(engine.integrator, 'getTemperature', None)):
        beta = 1 / (engine.integrator.getTemperature() * unit.BOLTZMANN_CONSTANT_kB)
    elif isinstance(engine, paths.engines.toy.Engine):
        beta = engine.options["integ"].beta
    else:
        raise AttributeError("Engine type not supported for beta calculation.")

    # Load trajectory
    print("Trajectory path {}".format(traj_path))
    old_store = Storage(traj_path,"r")
    init_traj = old_store.trajectories[-1]
    mdtraj_traj = init_traj.to_mdtraj()
    old_store.close()
    init_traj = ops_openmm.trajectory_from_mdtraj(mdtraj_traj)
    frame = init_traj[frame_index]

    # Setup randomizer
    modifier = paths.RandomVelocities(beta=beta, engine=engine)

    # Setup committor simulation
    output_path_name= Path(output_path/ f"committor_analysis_frame_{frame_index:04d}")
    output_path_ops_storage = output_path_name.with_suffix(".db")
    output_storage = Storage(output_path_ops_storage, "w")

    simulation = paths.CommittorSimulation(
        storage=output_storage,
        engine=engine,
        states=TPS_utils.states,
        randomizer=modifier,
        initial_snapshots=[frame]
    )

    # Manual run loop with timing
    n_done = 0
    log_interval = max(1, n_shots // 10)  # print progress 10 times at most
    start_time = time.time()

    while n_done < n_shots:
        remaining = n_shots - n_done
        this_batch = min(log_interval, remaining)

        simulation.run(n_per_snapshot=this_batch)

        n_done += this_batch
        elapsed = time.time() - start_time
        avg_time_per_shot = elapsed / n_done
        est_remaining = avg_time_per_shot * (n_shots - n_done)
        print("\n")
        print(f"[Frame {frame_index}] Completed {n_done}/{n_shots} shots "
            f"(~{avg_time_per_shot:.2f}s/shot), "
            f"Estimated time remaining: {est_remaining/60:.2f} min")


    results = paths.ShootingPointAnalysis(steps=output_storage.steps, states=TPS_utils.states, error_if_no_state=False)
    df = results.to_pandas()
    # Make sure the states you're interested in are present, fill missing ones with 0
    expected_states = ["bound", "unbound"]
    df = df.reindex(columns=expected_states, fill_value=0)

    # Compute p_B = unbound / (bound + unbound)
    total = df.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_B = np.where(total > 0, df["unbound"] / total, np.nan)  # Avoid division by zero

    df.to_csv(output_path_name.with_suffix(".csv"))

    output_storage.close()
    print(f"Finished frame {frame_index} with a predicted p_B of {p_B}")



def parse_arguments():
    arguments_list = [global_arguments["directory"],
        global_arguments["config_file_TPS"],
        (["-shots", "--shots"], int, True, "The number of committor analysis shots performed."),
        global_arguments["trajectory_path"],
        global_arguments["output_path"],
        global_arguments["system_resource_directory"],
        (["-frame_index", "--frame_index"], int, True, "The index of the frame on which committor analysis is performed.", None, "+"),
    ]
    parser = create_parser(arguments_list)
    return parser.parse_args()



def main():
    args = parse_arguments()

    frame_indices = args.frame_index
    # Assign parsed arguments directly to variables 
    input_path = args.directory
    config_path = args.config_file_TPS
    traj_path = args.trajectory_path
    output_path = args.output_path or input_path  # Default to input path if output path is None
    system_resource_directory = args.system_resource_directory
    n_shots = args.shots
    


    processes = []

    # Initialize and start processes for each interface value
    processes = []
    for i, frame_idx in enumerate(frame_indices):
        p = Process(
            target=run_committor_analysis_for_frame,
            args=(frame_idx, input_path, config_path, n_shots, traj_path, output_path, system_resource_directory)
        )
        processes.append(p)
        p.start()
    # Final join for remaining
    for proc in processes:
        proc.join()

if __name__ == "__main__":
    main()


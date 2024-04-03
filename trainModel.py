'''
Falcon job codes
PD     Pending. Job is waiting for resource allocation
R      Running. Job has an allocation and is running
S      Suspended. Execution has been suspended and resources have been released for other jobs
CA     Cancelled. Job was explicitly cancelled by the user or the system administrator
CG     Completing. Job is in the process of completing. Some processes on some nodes may still be active
CD     Completed. Job has terminated all processes on all nodes with an exit code of zero
F      Failed. Job has terminated with non-zero exit code or other failure condition
'''

import sys
import os
import subprocess
import re
from time import sleep

output_file = "squeueOutput.txt"

mem_per_cpu_max = 30
mem_per_cpu_min = 10
mem_per_cpu_step = 5


class Machine:
    def __init__(self, name, gpu, partition):
        self.name = name
        self.gpu = gpu
        self.partition = partition


machines = {}
for machine in [
    Machine("PEREGRINE-80","a100-sxm4-80gb","peregrine-gpu"),
    Machine("PEREGRINE-40","nvidia_a100_3g.39gb","peregrine-gpu"),
    Machine("KESTREL","3090","kestrel-gpu")
]:
    machines[machine.name] = machine


'''
Main is a recursive function so that we can re-try if we get a stalled, cancelled, failed, or pending job status
'''


def main(start_index: int, email: str, username: str, current_mem_per_cpu: int, machine: Machine):
    run_squeue()
    print(f"Attempting to train on {machine.name} {machine.gpu} with {current_mem_per_cpu}gb CPU memory")
    new_job_id = run_job(start_index, email, username, current_mem_per_cpu, machine)  # Run our job, get its id
    print(f"Waiting for job status to update...")
    sleep(15)  # Wait for job status to update
    print(f"Finished waiting for job status to update")
    error = check_for_error_code(new_job_id)  # Check for an error code
    if error:  # If we find an error code, handle it
        handle_error(username, email, start_index, current_mem_per_cpu, new_job_id, machine)  # This is where our recursive call lives
    else:
        print(f"No error code found. Job should be running.")


'''
start_index: the index we're starting at
email: the user's email
current_mem_per_cpu: the current memory that we're building our bash script with
Builds the bash script from parameters, writes it to a file, then runs it, returns id of new job
'''


def run_job(start_index: int, email: str, username: str, current_mem_per_cpu: int, machine: Machine):
    ids = set(get_job_id_set(username))  # Get all current job ids
    bash_file = get_bash_string(start_index, email, username, current_mem_per_cpu, machine)  # Get the bash file as a string
    print(f"Writing out generated bash script as .sh file...")

    current_working_directory = os.getcwd()
    script_name = f"ds_model_{start_index}.sh"
    script_path = f"{current_working_directory}/{script_name}"

    with open(script_name, "w") as f:
        f.write(bash_file)  # Write the bash file out to a file
    current_working_directory = os.getcwd()

    subprocess.run(["chmod", "u+x", script_path])

    print(f"Submitting job for '{script_name}'")
    subprocess.run(["sbatch", script_path])  # Run the bash file

    print(f"Waiting for job to start...")
    sleep(10)  # Wait for job to start
    print(f"Done waiting.")

    new_ids = set(get_job_id_set(username))  # Get new job ids
    new_job_id = get_id(ids, new_ids)  # Find the correct job id
    print(f"Found new job id: {new_job_id}")
    if new_job_id == -1:  # Abort if we couldn't find our job id
        print("Failed to find job id. Aborting.")
        exit(1)
    return new_job_id


'''
new_job_id: job id associated with our new job
Returns True or False depending on the result of a regex search for error codes
Looks through our output file (this is the output of an `squeue` command and matches the job id. Then do regex on
    that line and look for an error code. Return true if found, else false
'''


def check_for_error_code(new_job_id: str):
    print(f"Checking for error codes...")
    run_squeue()  # Run squeue so we can check for errors in the table
    with open(output_file, 'r') as f:  # Open our output file
        lines = f.readlines()  # Get all lines in file
        for line in lines:
            if new_job_id in line:  # Match the correct job id
                match = re.search(r"\b(PD|CA|S|F)\b", line)  # Regex for error codes
                if match is not None:  # If we find an error code, report by returning true
                    print(f"Found error code {match.group(0)}")
                    return True
    print(f"No error code found")
    return False  # No error codes found


'''
current_mem_per_cpu: current cpu memory allocation
new_job_id: id of relevant job
Decrements current_mem_per_cpu, checks if we're at the memory floor, and does recursion
'''


def handle_error(username:str, email:str, start_index:int, current_mem_per_cpu: int, new_job_id: int, machine: Machine):
    print(f"Handling error code...")
    current_mem_per_cpu -= mem_per_cpu_step  # Decrement the current memory per cpu
    if current_mem_per_cpu >= mem_per_cpu_min:  # If we're still at/above our floor, carry on
        # subprocess.run(["skill", new_job_id])  # Kill the job
        subprocess.run(["scancel", new_job_id])
        sleep(3)
        print(f"Found error code, retrying with {current_mem_per_cpu}G mem_per_cpu")
        main(start_index, email, username, current_mem_per_cpu, machine)  # Try again
    else:  # If we've breached the lower bound, abort the program
        print("Reached minimum memory requirement. Aborting.")
        exit(1)


'''
initial_set: this is a set() or all job id's before we started our new one
second_set: this is a set() or all job id's after we started our new one
Returns: the single job id that is present in second_set and absent in initial_set, -1 if no such id is found
'''


def get_id(initial_set: set, second_set: set):
    for item in second_set:  # Iterate second set
        if item not in initial_set:  # Find first element from second set that's not in first set
            return item  # Return it
    return -1  # If we didn't find it, return -1


'''
username: username of whoever is running this script. Comes in from CLI
Returns list of all job ID's associated with the username
This is a helper for finding the relevant job id
'''


def get_job_id_set(username: str):
    run_squeue()  # Run the squeue command, write the output to a file
    sleep(3)
    all_ids = []
    with open(output_file, 'r') as f:  # Open the file we just wrote to
        lines = f.readlines()  # Read all lines of file
        for line in lines:
            if username in line:  # Only consider lines associated with the correct user
                ids = re.findall(r"\d{4}", line)  # Regex match for 4 consecutive digits
                if len(ids) > 0:
                    all_ids.append(ids[0])
    return all_ids


'''
Runs the `squeue` command and writes the output to our output_file
'''


def run_squeue():
    output = subprocess.run(["squeue"], stdout=subprocess.PIPE).stdout.decode("utf-8")  # Run the squeue command
    with open(output_file, "w") as f:
        f.write(output)  # Write the output of squeue to the output file


'''
Builds a bash file as a string from parameters
'''


def get_bash_string(start_index: int, email: str, username: str, current_mem_per_cpu: int, machine: Machine):
    # ToDo update --cpus-per-task is we run into Resource issues. Start w/ 5, lower limit=1
    # ToDo reduce cpus-per-task FIRST upper bound=5, lower bound=1, THEN reduce mem-per-cpu

    logfile_name = f"log_{username}_{start_index}.out"
    return f'''#!/bin/bash
#SBATCH --job-name="model_1"
#SBATCH --partition={machine.partition}
#SBATCH --qos=gpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu={current_mem_per_cpu}G
#SBATCH --time=10-00:00:00
#SBATCH --output={logfile_name}
#SBATCH --error={logfile_name}
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user={email}@colostate.edu
#SBATCH --gres=gpu:{machine.gpu}:1
srun python3 /s/lovelace/f/nobackup/shrideep/sustain/everett/cl_3.8/deepSoil/models/sm_model_ev.py {start_index}'''

# source /s/lovelace/f/nobackup/shrideep/sustain/matt/cl_3.8/venv/bin/activate

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Invalid usage. Please run 'python3 trainModel.py <start_index> <your_email_username> <eid> <machine>'"
              "\nMachines: PEREGRINE-80 | PEREGRINE-40 | KESTREL"
              "\nEX: python3 trainModel.py 101 asterix asterix PEREGRINE-80"
              "\nExplanation: My email is 'asterix@rams.colostate.edu' and my eid is 'asterix'.")
        exit(1)
    if sys.argv[4] not in machines:
        print(f"'{sys.argv[4]}' not a valid machine. Valid options are <PEREGRINE-80 | PEREGRINE-40 | KESTREL>")
    start_index = int(sys.argv[1])
    email = sys.argv[2]
    username = sys.argv[3]
    machine = machines[sys.argv[4]]
    main(start_index, email, username, mem_per_cpu_max, machine)

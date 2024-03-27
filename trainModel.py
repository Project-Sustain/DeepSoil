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
import subprocess
import re
from time import sleep

output_file = "squeueOutput.txt"

mem_per_cpu_max = 30
mem_per_cpu_min = 10
mem_per_cpu_step = 5
gpu_size = 80


'''
Main is a recursive function so that we can re-try if we get a stalled, cancelled, failed, or pending job status
'''


def main(start_index: int, email: str, username: str, current_mem_per_cpu: int):
    new_job_id = run_job(start_index, email, username, current_mem_per_cpu)  # Run our job, get its id
    sleep(15)  # Wait for job status to update
    error = check_for_error_code(new_job_id)  # Check for an error code
    if error:  # If we find an error code, handle it
        handle_error(current_mem_per_cpu, new_job_id)  # This is where our recursive call lives
    else:
        print(f"No error code found. Job should be running.")


'''
start_index: the index we're starting at
email: the user's email
current_mem_per_cpu: the current memory that we're building our bash script with
Builds the bash script from parameters, writes it to a file, then runs it, returns id of new job
'''


def run_job(start_index: int, email: str, username: str, current_mem_per_cpu: int):
    ids = set(get_job_id_set(username))  # Get all current job ids
    bash_file = get_bash_string(start_index, email, username, current_mem_per_cpu)  # Get the bash file as a string
    with open("ds_model.sh", "w") as f:
        f.write(bash_file)  # Write the bash file out to a file
    subprocess.run(["ds_model.sh"])  # Run the bash file
    sleep(10)  # Wait for job to start
    new_ids = set(get_job_id_set(username))  # Get new job ids
    new_job_id = get_id(ids, new_ids)  # Find the correct job id
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
    run_squeue()  # Run squeue so we can check for errors in the table
    with open(output_file, 'r') as f:  # Open our output file
        lines = f.readlines()  # Get all lines in file
        for line in lines:
            if new_job_id in line:  # Match the correct job id
                error_codes = re.findall(r"\b(PD|CA|S|F)\b", line)  # Regex for error codes
                if len(error_codes) > 0:  # If we find an error code, report by returning true
                    return True
    return False  # No error codes found


'''
current_mem_per_cpu: current cpu memory allocation
new_job_id: id of relevant job
Decrements current_mem_per_cpu, checks if we're at the memory floor, and does recursion
'''


def handle_error(current_mem_per_cpu: int, new_job_id: int):
    current_mem_per_cpu -= mem_per_cpu_step  # Decrement the current memory per cpu
    if current_mem_per_cpu >= mem_per_cpu_min:  # If we're still at/above our floor, carry on
        subprocess.run(["skill", new_job_id])  # Kill the job
        sleep(3)
        print(f"Found error code, retrying with {current_mem_per_cpu}G mem_per_cpu")
        main(start_index, email, username, current_mem_per_cpu)  # Try again
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


def get_bash_string(start_index: int, email: str, username: str, current_mem_per_cpu: int):
    logfile_name = f"log_{username}_{start_index}.out"
    return f'''#!/bin/bash
#SBATCH --job-name="model_1"
#SBATCH --partition=peregrine-gpu
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
#SBATCH --mail-user=<{email}>@colostate.edu
#SBATCH --gres=gpu:a100-sxm4-{gpu_size}gb:1
srun python3 /cl_3.8/deepSoil/models/sm_model_ev.py {start_index}'''


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Invalid usage. Please run 'python3 trainModel.py <start_index> <your_email_username> <eid>'"
              "\nEX: python3 trainModel.py 100 asterix asterix"
              "\nExplanation: My email is 'asterix@rams.colostate.edu' and my eid is 'asterix'.")
        exit(1)
    start_index = int(sys.argv[1])
    email = sys.argv[2]
    username = sys.argv[3]
    main(start_index, email, username, mem_per_cpu_max)

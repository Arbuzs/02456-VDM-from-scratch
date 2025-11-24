import wandb
import subprocess
import yaml
import sys
import os

# --- Configuration ---
SWEEP_CONFIG_FILE = "project/sweep_config.yaml"
# You might want to run multiple agents in parallel on the HPC
NUM_AGENTS = 4 

# --- 1. Define the Sweep ---

def run_sweep():
    # 1. Define the sweep using the YAML file
    with open(SWEEP_CONFIG_FILE, 'r') as f:
        sweep_config = yaml.safe_load(f)

    # 2. Create the sweep and get the sweep ID
    # Note: Replace 'your-entity' with your actual username/team name
    # We will try to infer the project name from main.py
    try:
        # Get project name (assuming it's VDM-from-scratch)
        project_name = "VDM-from-scratch" 
        sweep_id = wandb.sweep(
            sweep_config, 
            project=project_name
        )
        print(f"Successfully created sweep with ID: {sweep_id}")

        # 3. Launch the agents
        print(f"Launching {NUM_AGENTS} WandB agents...")
        
        # We start the agents using a subprocess command.
        # This command connects the agent to the sweep and starts work.
        # Note: You should typically run 'wandb agent' using LSF/HPC job submission 
        # (e.g., in a separate bsub job array) to allocate resources for each agent.
        
        for i in range(NUM_AGENTS):
            # Command to launch a single agent
            agent_command = f"wandb agent {sweep_id}"
            print(f"Starting Agent {i+1} with command: {agent_command}")
            # subprocess.Popen(agent_command, shell=True) 
            
            # Since this is an interactive session, we'll just print the command.
            # On an HPC, you would typically submit this command via a job scheduler.
            print(f"*** Run this command on your HPC node: {agent_command} ***")
            
    except Exception as e:
        print(f"An error occurred during sweep setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure wandb is initialized for the script that creates the sweep
    wandb.login() 
    run_sweep()
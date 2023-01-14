import subprocess

IDX = 1

for i in range(7):
    subprocess.run(f"python main_nce.py --fname=mass_shorttraj_05_{i + 7 * IDX} --data_config=short_trajectories.json --device={IDX}", shell=True)

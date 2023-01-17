import subprocess

IDX = 0

for i in range(50):
    subprocess.run(f"python main_nce.py --fname=mass_shorttraj_05_{i + 7 * IDX} --data_config=short_trajectories.json --device={IDX} --epochs=500", shell=True)

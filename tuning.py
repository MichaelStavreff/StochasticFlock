# %%
import stochastic_flock
import sys

params = stochastic_flock.Parameters()
params.kN_BIRDS = 50
# %% 2. Setup Random Generator and Simulation
rng = stochastic_flock.MT19937(50)
sim = stochastic_flock.Simulation2d(params, rng)
velocity_vec_x = []
velocity_vec_y = []
statuses = []
# %%
print(f"Starting simulation with {params.kN_BIRDS} birds...")

steps = 35000
for i in range(1, steps + 1):
    sim.update()
    velocity_vec_x.append(sim.velocities[0][0])
    velocity_vec_y.append(sim.velocities[0][1])
    statuses.append(sim.status[0])
    if i % 50 == 0:
        print(f"Step {i} completed...")

print("Simulation finished.")
sys.exit(0)

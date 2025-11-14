#Code to run simulations in batches
#2025-06-19 Merlin Tétrault-Leclerc
#_________________________________________________________________________________
#_________________________________________________________________________________


#LIBRARIES
#_________________________________________________________________________________
import numpy as np
import Module_dynamics as dyn
import multiprocessing as mp

#CODE
#_________________________________________________________________________________
def run_sim(params):
    mass, capacity, solararea, power, simtime = params

    # Initialize vehicle and environment
    path = dyn.Path(name='CGV', type='A2A')
    env = dyn.Environment(path, StartDateTimeLocal='2025-06-15 00:00:00')
    rider = dyn.Rider()
    battery = dyn.Battery(capacity=capacity)
    solarpanel = dyn.SolarPanel(area=solararea)
    chassis = dyn.Chassis(mass=20, CdA=0.599631, Crr=0.004)
    chassis.cargo_mass = 30
    motor = dyn.Motor(RatedPower=power, efficiency=1)
    vehicle = dyn.Vehicle(env, rider, battery, solarpanel, chassis, motor)
    vehicle.mass = mass

    # Run simulation (will skip if results file exists)
    dyn.simulate_fixedtimestep(vehicle, simtime_s=simtime, dt=1, SOC_i=1, output = "light")

if __name__ == "__main__":
    simtime = 60*60*24*15

    mass_sweep = np.arange(100, 176, 25)
    capacity_sweep = np.arange(500, 1501, 500)
    solararea_sweep = np.arange(0.5, 1.51, 0.5)
    power_sweep = np.arange(250, 376, 125)

    # Prepare all parameter combinations
    param_grid = [
        (mass, capacity, solararea, power, simtime)
        for mass in mass_sweep
        for capacity in capacity_sweep
        for solararea in solararea_sweep
        for power in power_sweep
    ]

    # Use all available CPU cores
    # with mp.Pool(mp.cpu_count()) as pool:
    #     pool.map(run_sim, param_grid)

    # Use specified CPU cores
    with mp.Pool(processes=6) as pool:
        pool.map(run_sim, param_grid)
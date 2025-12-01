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
    name, mass, capacity, solararea, power, simtime, suffix = params

    # Initialize vehicle and environment
    path = dyn.Path(name=name, type='A2A')
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
    dyn.simulate_fixedtimestep(vehicle, simtime_s=simtime, dt=1, SOC_i=1, output = "full",
                               filename_suffix=suffix)

def run_sim_adaptative(params):
    name, mass, capacity, solararea, power, simtime, suffix = params

    # Initialize vehicle and environment
    path = dyn.Path(name=name, type='A2A')
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
    dyn.simulate_variabletimestep(vehicle, simtime_s=simtime, atol=[1, 0.01, 50.0], rtol=0.001,
                                  base_dt=0.1, max_dv=2,max_dt=10, ffwd_dt=100, SOC_i=0,
                                  output = "full", use_pbar=True,filename_suffix=suffix)

if __name__ == "__main__":
    simtime = 60*60*24*15

    # name_sweep = np.array(['CGV', 'PROTOUR'])
    # mass_sweep = np.arange(100, 176, 25)
    # capacity_sweep = np.arange(500, 1501, 500)
    # solararea_sweep = np.arange(0.5, 1.51, 0.5)
    # power_sweep = np.arange(250, 376, 125)

    mass_sweep = np.array([100])
    capacity_sweep = np.array([500])
    solararea_sweep = np.array([0.5])
    power_sweep = np.array([250])
    name_sweep = np.array(['CGV', 'PROTOUR'])

    # Prepare all parameter combinations
    param_grid = [
        (name, mass, capacity, solararea, power, simtime)
        for name in name_sweep
        for mass in mass_sweep
        for capacity in capacity_sweep
        for solararea in solararea_sweep
        for power in power_sweep
    ]

    param_grid = np.array(param_grid,dtype=object)

    with mp.Pool(processes=12) as pool:
        args_adapt = np.column_stack([param_grid, np.full(len(param_grid), 'Adaptive', dtype=object)])
        args_fixed = np.column_stack([param_grid, np.full(len(param_grid), 'Fixed', dtype=object)])

        res1 = pool.map_async(run_sim_adaptative, args_adapt)
        res2 = pool.map_async(run_sim, args_fixed)

        # force l’attente + remonte les exceptions
        res1.get()
        res2.get()



#! /usr/bin/env python3
import numpy as np
import sys, os
import h5py
from scipy.optimize import minimize

from multiprocessing import Pool, current_process
import threading
os.environ["OMP_NUM_THREADS"] = "1"
local_storage = threading.local()


sys.path.append("cpc_moist/python/")
sys.path.append("cpc_moist/bin/")
from canoe import def_species, load_configure, index_map
from canoe.snap import def_thermo
from canoe.athena import Mesh, ParameterInput, Outputs, MeshBlock
from scipy.optimize import minimize


def run_canoe_harp_fwd(theta, jindex):
    qH2O, RHmax = theta
    max_inter = 200
    # mb.construct_atmosphere(pin, qNH3, T0, RHmax, jindex, "pseudo", qH2O*1000., max_inter) ## perturb atm temperature with 
    mb.construct_atmosphere_Ts(pin,  qNH3,  Ts,  RHmax,  jindex, "pseudo", qH2O*1000) 

    # run forward RT model
    rad.cal_radiance(mb, mb.k_st, mb.j_st + jindex)
    tb = np.zeros(4 * nb)
    for ib in range(nb):
        toa = rad.get_band(ib).get_toa()[0]
        tb[ib * 4 : ib * 4 + 4] = toa
    return tb[::4][2:] ## [chn3, chn4, chn5, chn6]

def get_nh3_profile(jindex):
    aircolumn = mb.get_aircolumn(mb.k_st, mb.j_st + jindex, mb.i_st, mb.i_ed)   
    nh3_ppm = np.zeros(len(aircolumn))
    h2o_ppm = np.zeros(len(aircolumn))
    SH_NH3 = np.zeros(len(aircolumn))
    RH_NH3 = np.zeros(len(aircolumn))
    temp = np.zeros(len(aircolumn))
    ptemp = np.zeros(len(aircolumn))

    for i in range(len(aircolumn)):
        ap = aircolumn[i]
        SH_NH3[i] = ap.hydro()[iNH3]  ## specified mixing ratio kg/kg
        ap_mole = ap.to_mole_fraction()
        nh3_ppm[i] = ap_mole.hydro()[iNH3]*1.0E6   ## mole fraction [ppm]
        h2o_ppm[i] = ap_mole.hydro()[iH2O]*1.0E6   ## mole fraction [ppm]
        RH_NH3[i] = ap_mole.get_rh(iNH3) ## relative humidity [0-1]

        temp[i] = mb.get_temp(mb.k_st, mb.j_st + jindex, mb.i_st + i)
        ptemp[i] = mb.get_theta(1E5, mb.k_st, mb.j_st + jindex, mb.i_st + i)
    return nh3_ppm, h2o_ppm, SH_NH3, RH_NH3, temp, ptemp

def cost_function(theta, obs, jindex):
    qH2O, RHmax= theta
    simu = run_canoe_harp_fwd(theta, jindex)
    likelihood = 0.5 * np.sum(((obs[1:] - simu[1:]) / sigma) ** 2)

    H2O_prior = 0.5 * ((qH2O - 1.79) / 4) ** 2
    RHmax_prior = 0.5 * ((RHmax - 0.71) / 0.2) ** 2

    prior = RHmax_prior + H2O_prior
    if qH2O <= 0 or qH2O > 12 or RHmax < 0.01 or RHmax > 1.2:
        prior = np.inf
    return likelihood + prior

def retrieve_nh3_profile(obs, initial_guess, jindex):
    bounds = [(0.01, 12), (0.01, 1.2)]
    options = {'maxiter': 50}
    result = minimize(cost_function, initial_guess, args=(obs, jindex), method='L-BFGS-B', bounds=bounds, options=options)
    return result.x


nx2 = 30
global pin
pin = ParameterInput()
pin.load_from_file("juno_mwr.inp")

vapors = pin.get_string("species", "vapor").split(", ")
clouds = pin.get_string("species", "cloud").split(", ")
tracers = pin.get_string("species", "tracer").split(", ")

def_species(vapors=vapors, clouds=clouds, tracers=tracers)
def_thermo(pin)

config = load_configure("juno_mwr.yaml")

pin.set_boolean("job", "verbose", False)
pin.set_string("mesh", "nx2", f"{nx2}")

mesh = Mesh(pin)
mesh.initialize(pin)

global mb, rad, nb
mb = mesh.meshblock(0)
rad = mb.get_rad()
nb = rad.get_num_bands()

global iNH3, iH2O
pindex = index_map.get_instance()
iNH3 = pindex.get_vapor_id("NH3")
iH2O = pindex.get_vapor_id("H2O")
print(f"NH3 index: {iNH3}")
print(f"H2O index: {iH2O}")


global  qNH3
global T0
qNH3=354.83
T0 = 175.67
rhmax = 0.71
adlnNH3 =-0.06
pmin=1E-3
pmax=4.89
jindex=0

## set the background atmosphere
## retrieve bottom Ts
global Ts
Ts=mb.retrieve_Ts_given_T1bar(pin,  qNH3,  T0, rhmax,  jindex, "dry", 2500, 200)


tb_bg =run_canoe_harp_fwd([1.79, 0.71], jindex)

initial_guess = [1.79, 0.71]

global sigma
# sigma = [0.5,0.3,0.2,0.1]
sigma = 0.5

pvfile = h5py.File("observation/tb_anomaly_grided_PJ51_PJ61_FA02.h5", "r")
anomaly = pvfile["anomaly"][2:, :, :]  # (6,100,100)
pvfile.close()

# transform from (4,100,100) to (100,100,4)
res_region=anomaly.transpose(1,2,0)
print(f"Residual region shape: {res_region.shape}")
# print(f"tb_bg shape: {tb_bg.shape}")

print(tb_bg)


nx,ny,nch=res_region.shape

res_region = res_region.reshape(nx * ny, nch)

print(res_region.shape)

theta_total_1d = np.zeros((nx * ny, 2))
tbs_total_1d = np.zeros((nx * ny, 4))
chi_total_1d = np.zeros((nx * ny))
ppms_total_1d = np.zeros((nx * ny, 1600))
ppmh2o_total_1d = np.zeros((nx * ny, 1600))
shs_total_1d = np.zeros((nx * ny, 1600))
rhs_total_1d = np.zeros((nx * ny, 1600))
temp_total_1d = np.zeros((nx * ny, 1600))
ptemp_total_1d = np.zeros((nx * ny, 1600))                      

valid_index = ~np.isnan(res_region[:,0])
res = res_region[valid_index]
res[np.isnan(res)] = 0
tb0 = res + tb_bg

nvalid = tb0.shape[0]
nchunks = nvalid // nx2 + 1

print(f"Total valid points: {tb0.shape}")
print(f"Total valid points: {nvalid}")
print(f"Total valid points: {nx * ny}")

## load history theta file
# with h5py.File("thetas_in_chunk_pseudo_breakat211.h5", "r") as f:
#     thetas = f["thetas"][:]
#     tbs = f["tbs"][:]
thetas = np.zeros((nvalid, 2)) 
chi = np.zeros((nvalid,)) 
ppms = np.zeros((nvalid, 1600))
ppmh2o= np.zeros((nvalid, 1600))
shs = np.zeros((nvalid, 1600))
rhs = np.zeros((nvalid, 1600))
ptemp = np.zeros((nvalid, 1600))
temp = np.zeros((nvalid, 1600))
tbs = np.zeros((nvalid, 4))

def retrieve_nh3_profile_concurrent(args):
    obs, initial_guess, jindex = args
    return retrieve_nh3_profile(obs, initial_guess, jindex)

for ichunk in range(nchunks-1):
# for ichunk in range(0,2):
    print(f"Processing chunk {ichunk} of {nchunks}")
    # if ichunk == nchunks - 1:
    #     tb = tb0[ichunk * nx2 :]  
    #     nx2 = tb.shape[0]   
    # else:
    tb = tb0[ichunk * nx2 : (ichunk + 1) * nx2]
    # print(tb)

    with Pool(nx2) as pool:
        theta = pool.map(retrieve_nh3_profile_concurrent, [(tb[i], initial_guess, i) for i in range(nx2)])
    thetas[ichunk * nx2 : (ichunk + 1) * nx2] = theta

    for i, thet in enumerate(theta):
        print(f"Result for index {i}: {thet}")
        tbs[ichunk * nx2 + i]=run_canoe_harp_fwd(thet, i)
        chi[ichunk * nx2 + i] = cost_function(thet, tb[i], i)
        ppms[ichunk * nx2 + i], ppmh2o[ichunk * nx2 + i], shs[ichunk * nx2 + i], rhs[ichunk * nx2 + i], temp[ichunk * nx2 + i], ptemp[ichunk * nx2 + i]   = get_nh3_profile(i)
        thetas[ichunk * nx2 + i] = thet

    
    # for i, thet in enumerate(theta):
    #     print(f"Result for index {i}: {thet}", tbs[ichunk * nx2 +i] - tb0[ichunk * nx2 +i])

theta_total_1d[valid_index] = thetas
theta_total = theta_total_1d.reshape(nx, ny, 2)

chi_total_1d[valid_index] = chi
ppms_total_1d[valid_index] = ppms
ppmh2o_total_1d[valid_index] = ppmh2o
shs_total_1d[valid_index] = shs
rhs_total_1d[valid_index] = rhs
temp_total_1d[valid_index] = temp
ptemp_total_1d[valid_index] = ptemp
tbs_total_1d[valid_index] = tbs

chi_total = chi_total_1d.reshape(nx, ny)
tbs_total = tbs_total_1d.reshape(nx, ny, 4)
ppms_total = ppms_total_1d.reshape(nx, ny, 1600)
ppmh2o_total = ppmh2o_total_1d.reshape(nx, ny, 1600)
shs_total = shs_total_1d.reshape(nx, ny, 1600)
rhs_total = rhs_total_1d.reshape(nx, ny, 1600)
temp_total = temp_total_1d.reshape(nx, ny, 1600)
ptemp_total = ptemp_total_1d.reshape(nx, ny, 1600)

with h5py.File("thetas_in_grid_pseudo.h5", "w") as f:
    f.create_dataset(f"thetas", data=theta_total)
    f.create_dataset(f"tbs", data=tbs_total)

with h5py.File("profile_in_grid_pseudo.h5", "w") as f:
    f.create_dataset(f"ppms", data=ppms_total)
    f.create_dataset(f"shs", data=shs_total)
    f.create_dataset(f"rhs", data=rhs_total)
    f.create_dataset(f"chi", data=chi_total)
    f.create_dataset(f"temp", data=temp_total)
    f.create_dataset(f"ptemp", data=ptemp_total)

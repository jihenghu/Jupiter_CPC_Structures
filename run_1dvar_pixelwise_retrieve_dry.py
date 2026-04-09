#! /usr/bin/env python3
import numpy as np
import sys, os
import h5py
from scipy.optimize import minimize

from multiprocessing import Pool, current_process
import threading
os.environ["OMP_NUM_THREADS"] = "1"
local_storage = threading.local()


sys.path.append("cpc_dry/python/")
sys.path.append("cpc_dry/bin/")
from canoe import def_species, load_configure, index_map
from canoe.snap import def_thermo
from canoe.athena import Mesh, ParameterInput, Outputs, MeshBlock
from scipy.optimize import minimize

def set_equilibrium_aircolumn(RHmax: float=1.0, adlnNH3dlnP: float=0.0,
                 pmax: float = 0.0, jindex: int = 0):  
    # global xnh3, t1bar
    xH2O = 2500.
    max_inter = 500
    # qNH3 = xnh3
    # T0 = t1bar
    pmin: float = 0.0
    mb.construct_atmosphere(pin, qNH3, T0, RHmax, jindex, "dry", xH2O, max_inter)
    mb.modify_dlnNH3dlnP_rhmax(pin, adlnNH3dlnP, pmin, pmax*1E5, RHmax, jindex, "dry") 


def run_canoe_harp_fwd(theta, jindex):
    RHmax, adlnNH3dlnP, pmax = theta
    set_equilibrium_aircolumn(RHmax, adlnNH3dlnP, pmax, jindex)

    # run forward RT model
    rad.cal_radiance(mb, mb.k_st, mb.j_st + jindex)
    tb = np.zeros(4 * nb)
    for ib in range(nb):
        toa = rad.get_band(ib).get_toa()[0]
        tb[ib * 4 : ib * 4 + 4] = toa
    # print(f"TB: {tb[::4][2:]} ")
    return tb[::4][2:] ## [chn3, chn4, chn5, chn6]

def get_nh3_profile(jindex):
    aircolumn = mb.get_aircolumn(mb.k_st, mb.j_st + jindex, mb.i_st, mb.i_ed)   
    nh3_ppm = np.zeros(len(aircolumn))
    h2o_ppm = np.zeros(len(aircolumn))
    temp= np.zeros(len(aircolumn))  
    ptemp = np.zeros(len(aircolumn))

    SH_NH3 = np.zeros(len(aircolumn))
    RH_NH3 = np.zeros(len(aircolumn))


    for i in range(len(aircolumn)):
        ap = aircolumn[i]
        SH_NH3[i] = ap.hydro()[iNH3]  ## specified mixing ratio kg/kg
        ap_mole = ap.to_mole_fraction()
        nh3_ppm[i] = ap_mole.hydro()[iNH3]*1.0E6   ## mole fraction [ppm]
        h2o_ppm[i] = ap_mole.hydro()[iH2O]*1.0E6   ## mole fraction [ppm]
        RH_NH3[i] = ap_mole.get_rh(iNH3) ## relative humidity [0-1]
        temp[i] = mb.get_temp(mb.k_st, mb.j_st + jindex, mb.i_st + i)
        ptemp[i] = mb.get_theta(1E5, mb.k_st, mb.j_st + jindex, mb.i_st + i)
    return nh3_ppm, h2o_ppm, SH_NH3, RH_NH3,  temp, ptemp

def cost_function(theta, obs, jindex):
    RHmax, adlnNH3dlnP, pmax = theta
    simu = run_canoe_harp_fwd(theta, jindex)
    # print(simu - obs)
    # print(theta)
    likelihood = 0.5 * np.sum(((obs - simu) / sigma) ** 2)
    # print(theta, obs, simu, simu-obs, jindex)
    # Gaussian priors
    RHmax_prior = 0.5 * ((RHmax - 0.6) / 0.2) ** 2
    adlnNH3dlnP_prior = 0.5 * ((adlnNH3dlnP + 0.08) / 0.1) ** 2
    pmax_prior = 0.5 * ((pmax - 4.8) / 10) ** 2
    prior = RHmax_prior + adlnNH3dlnP_prior + pmax_prior
    # print(likelihood, prior)
    if RHmax < 0.27 or RHmax > 1.2 or adlnNH3dlnP < -0.3 or adlnNH3dlnP > 0.2 or pmax < 1 or pmax > 10:
        prior = np.inf

    return likelihood + prior


# def jacobian(theta, obs, jindex):
#     epsilon = 0.1
#     jac = np.zeros_like(theta)
#     for i in range(len(theta)):
#         theta_eps = np.copy(theta)
#         theta_eps[i] *= epsilon+1
#         cost_plus = cost_function(theta_eps, obs, jindex)
        
#         theta_eps[i] *= 2 * epsilon + 1
#         cost_minus = cost_function(theta_eps, obs, jindex)
        
#         jac[i] = (cost_plus - cost_minus) / (2 * epsilon)
#     return jac


def retrieve_nh3_profile(obs, initial_guess, jindex):
    bounds = [(0.27, 1.2), (-0.3, 0.2), (1, 10)]
    options = {'maxiter': 300}
    result = minimize(cost_function, initial_guess, args=(obs, jindex), method='L-BFGS-B', bounds=bounds, options=options)
    # print(result.x)
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


global qNH3
global T0
qNH3=354.83
T0 = 175.67
## set the background atmosphere
initial_guess = [0.71, -0.06, 4.89]
tb_bg = run_canoe_harp_fwd(initial_guess, 0)

global sigma
sigma = 0.5

pvfile = h5py.File("observation/tb_anomaly_grided_PJ51_PJ61_FA02.h5", "r")
anomaly = pvfile["anomaly"][2:, :, :]  # (6,100,100)
pvfile.close()

# transform from (4,100,100) to (100,100,4)
res_region=anomaly.transpose(1,2,0)
print(f"Residual region shape: {res_region.shape}")
# print(f"tb_bg shape: {tb_bg.shape}")

print(tb_bg)
# exit()


nx,ny,nch=res_region.shape

res_region = res_region.reshape(nx * ny, nch)

print(res_region.shape)

theta_total_1d = np.zeros((nx * ny, 3))
chi_total_1d = np.zeros((nx * ny))
ppms_total_1d = np.zeros((nx * ny, 1600))
ppms_h2O_total_1d = np.zeros((nx * ny, 1600))
shs_total_1d = np.zeros((nx * ny, 1600))
rhs_total_1d = np.zeros((nx * ny, 1600))
tbs_total_1d = np.zeros((nx * ny, 4))                        
temps_total_1d = np.zeros((nx * ny, 1600))  
ptemps_total_1d = np.zeros((nx * ny, 1600))

valid_index = ~np.isnan(res_region[:, 0])
tb0 = res_region[valid_index] + tb_bg

nvalid = tb0.shape[0]
nchunks = nvalid // nx2 + 1

print(f"Total valid points: {tb0.shape}")
print(f"Total valid points: {nvalid}")
print(f"Total valid points: {nx * ny}")

thetas = np.zeros((nvalid, 3)) 
chi = np.zeros((nvalid,)) 
ppms = np.zeros((nvalid, 1600))
ppms_h2O = np.zeros((nvalid, 1600))
temps = np.zeros((nvalid, 1600)) 
ptemps = np.zeros((nvalid, 1600))
shs = np.zeros((nvalid, 1600))
rhs = np.zeros((nvalid, 1600))
tbs = np.zeros((nvalid, 4))

def retrieve_nh3_profile_concurrent(args):
    obs, initial_guess, jindex = args
    return retrieve_nh3_profile(obs, initial_guess, jindex)

for ichunk in range(nchunks-1):
    print(f"Processing chunk {ichunk} of {nchunks}")
    # if ichunk == nchunks - 1:
    #     tb = tb0[ichunk * nx2 :]  
    #     nx2 = tb.shape[0]   
    # else:
    tb = tb0[ichunk * nx2 : (ichunk + 1) * nx2]

    with Pool(nx2) as pool:
        theta = pool.map(retrieve_nh3_profile_concurrent, [(tb[i], initial_guess, i) for i in range(nx2)])
        
    for i, thet in enumerate(theta):
        print(f"Result for index {i}: {thet}")
        tbs[ichunk * nx2 + i]=run_canoe_harp_fwd(thet, i)
        chi[ichunk * nx2 + i] = cost_function(thet, tb[i], i)
        ppms[ichunk * nx2 + i], ppms_h2O[ichunk * nx2 + i], shs[ichunk * nx2 + i], rhs[ichunk * nx2 + i], temps[ichunk * nx2 + i], ptemps[ichunk * nx2 + i] = get_nh3_profile(i)
        thetas[ichunk * nx2 + i] = thet

    # # thetas[ichunk * nx2 : (ichunk + 1) * nx2] = theta
    # with h5py.File("thetas_in_chunk_correct.h5", "w") as f:
    #     f.create_dataset(f"thetas", data=thetas)
    
    # with h5py.File("NH3_profile_in_chunk_correct.h5", "w") as f:
    #     f.create_dataset(f"ppms", data=ppms)
    #     f.create_dataset(f"shs", data=shs)
    #     f.create_dataset(f"rhs", data=rhs)

theta_total_1d[valid_index] = thetas
theta_total = theta_total_1d.reshape(nx, ny, 3)

tbs_total_1d[valid_index] = tbs


chi_total_1d[valid_index] = chi
ppms_total_1d[valid_index] = ppms
ppms_h2O_total_1d[valid_index] = ppms_h2O
shs_total_1d[valid_index] = shs
rhs_total_1d[valid_index] = rhs
temps_total_1d[valid_index] = temps
ptemps_total_1d[valid_index] = ptemps

tbs_total = tbs_total_1d.reshape(nx, ny, 4)

chi_total = chi_total_1d.reshape(nx, ny)
ppms_total = ppms_total_1d.reshape(nx, ny, 1600)
ppms_h2O_total = ppms_h2O_total_1d.reshape(nx, ny, 1600)
shs_total = shs_total_1d.reshape(nx, ny, 1600)
rhs_total = rhs_total_1d.reshape(nx, ny, 1600)
temps_total = temps_total_1d.reshape(nx, ny, 1600)
ptemps_total = ptemps_total_1d.reshape(nx, ny, 1600)

with h5py.File("thetas_in_grid_dry.h5", "w") as f:
    f.create_dataset(f"thetas", data=theta_total)
    f.create_dataset(f"tbs", data=tbs_total)

with h5py.File("profiles_in_grid_dry.h5", "w") as f:
    f.create_dataset(f"ppm_NH3", data=ppms_total)
    f.create_dataset(f"shs_nh3", data=shs_total)
    f.create_dataset(f"rhs_nh3", data=rhs_total)
    f.create_dataset(f"chi", data=chi_total)
    f.create_dataset(f"ppm_h2o", data=ppms_h2O_total)
    f.create_dataset(f"temp", data=temps_total)
    f.create_dataset(f"ptemp", data=ptemps_total)
    # f.create_dataset(f"press", data=ptemps_total)

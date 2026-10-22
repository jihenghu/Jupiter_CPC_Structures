"""CPC-mean retrieved NH3 profiles.

Uses the same CPC boxes / Tb-anomaly thresholds as run_fit.ipynb to build the
per-cyclone sample masks, then averages the 1DVar-retrieved specific-humidity
NH3 profiles (shs_nh3) of ../profiles_in_grid_dry.h5 over each CPC.
Output: cpcmean_nh3_profiles.h5
"""

import numpy as np
import h5py

# ---------------------------------------------------------------- grid / obs
with h5py.File("../observation/tb_anomaly_grided_PJ51_PJ61_FA02.h5", "r") as h5f:
    anomaly = h5f["anomaly"][:]
    lon2d = h5f["longitude"][:]
    lat2d = h5f["latitude"][:]

lon2d[lon2d > 180] -= 360

# ---------------------------------------------------------------- CPC boxes
cpcs_index = np.array([1, 2, 3, 4, 5, 6, 7, 8])
cpcs_lon_min = np.array([-30, -75, -130, -165, 145, 95, 60, 20])
cpcs_lon_max = np.array([10, -45, -90, -135, 180, 135, 90, 60])
cpcs_lat_min = np.array([80, 82, 80.5, 80.5, 81, 82.5, 79, 82.5])
cpcs_lat_max = np.array([85, 85, 85, 85, 85, 86, 83.5, 85.5])

thera = [1, 0.2, 1, 1, 0.35, 0.35, 1, 0.35]

# ---------------------------------------------------------------- profiles
with h5py.File("../profiles_in_grid_dry.h5", "r") as h5f:
    shsnh3 = h5f["shs_nh3"][:]          # (100, 100, 1600)

shsnh3[shsnh3 == 0] = np.nan            # unretrieved pixels

with h5py.File("../cpc_dry/bin/juno_forward_mwr_polarmean-main.nc", "r") as f:
    pressure = f["press"][0, :, 0, 0] * 1e-5     # bar

with h5py.File("../cpc_dry/bin/juno_mwr_forward_polarmean_profiles.h5", "r") as f:
    bg_nh3_sh = f["nh3_sh"][:]          # polar-mean background profile

nlev = shsnh3.shape[-1]

# ---------------------------------------------------------------- CPC means
ncpc = 8
cpc_mean = np.full((ncpc + 1, nlev), np.nan)
cpc_std = np.full((ncpc + 1, nlev), np.nan)
nsample = np.zeros(ncpc + 1, dtype=np.int32)

for i in range(ncpc):
    lon_mask = (lon2d >= cpcs_lon_min[i]) & (lon2d <= cpcs_lon_max[i]) & (anomaly[5, :, :] > thera[i])
    lat_mask = (lat2d >= cpcs_lat_min[i]) & (lat2d <= cpcs_lat_max[i]) & (anomaly[5, :, :] > thera[i])
    mask = lon_mask & lat_mask
    nsample[i] = np.count_nonzero(mask)

    cpc_mean[i, :] = np.nanmean(shsnh3[mask, :], axis=0)
    cpc_std[i, :] = np.nanstd(shsnh3[mask, :], axis=0)
    print(f"CPC {i+1}: {nsample[i]} samples")

## central polar cyclone
mask = lat2d >= 87.5
nsample[ncpc] = np.count_nonzero(mask)
cpc_mean[ncpc, :] = np.nanmean(shsnh3[mask, :], axis=0)
cpc_std[ncpc, :] = np.nanstd(shsnh3[mask, :], axis=0)
print(f"Polar Cyclone: {nsample[ncpc]} samples")

# ---------------------------------------------------------------- output
with h5py.File("cpcmean_nh3_profiles.h5", "w") as outhdf:
    d = outhdf.create_dataset("shs_nh3_mean", data=cpc_mean[:ncpc, :])
    d.attrs["description"] = "CPC-mean retrieved NH3 specific humidity profile, CPC 1-8"
    d = outhdf.create_dataset("shs_nh3_std", data=cpc_std[:ncpc, :])
    d.attrs["description"] = "standard deviation over the samples of each CPC"
    outhdf.create_dataset("nsample", data=nsample[:ncpc])
    outhdf.create_dataset("cpc_index", data=cpcs_index)
    outhdf.create_dataset("cpc_lon_min", data=cpcs_lon_min)
    outhdf.create_dataset("cpc_lon_max", data=cpcs_lon_max)
    outhdf.create_dataset("cpc_lat_min", data=cpcs_lat_min)
    outhdf.create_dataset("cpc_lat_max", data=cpcs_lat_max)
    outhdf.create_dataset("tb_threshold", data=np.array(thera))
    ## central polar cyclone (lat >= 87.5), kept separately from the eight CPCs
    outhdf.create_dataset("shs_nh3_mean_polar", data=cpc_mean[ncpc, :])
    outhdf.create_dataset("shs_nh3_std_polar", data=cpc_std[ncpc, :])
    d = outhdf.create_dataset("pressure", data=pressure)
    d.attrs["units"] = "bar"
    d = outhdf.create_dataset("shs_nh3_background", data=bg_nh3_sh)
    d.attrs["description"] = "polar-mean background NH3 specific humidity profile"

print("saved cpcmean_nh3_profiles.h5", cpc_mean[:ncpc, :].shape)

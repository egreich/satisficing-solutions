# competition_factor = α
# water_penalty = exp(-α * community_strategy)
# α = -ln(water_penalty) / community_strategy
import h5py
import numpy as np

def calculate_diurnal_range(data_series, time_axis):
    """
    Calculate the diurnal (daily) range from a time series.
    
    Parameters:
    data_series: Array of values (e.g., soil moisture)
    time_axis: Array of corresponding timestamps
    
    Returns:
    diurnal_range: Maximum - Minimum value in a day
    """
    # Convert to daily groups if needed
    if len(data_series) >= 8:  # SMAP typically has 3-hourly data (8 values/day)
        daily_groups = []
        for i in range(0, len(data_series), 8):
            daily_data = data_series[i:i+8]
            if len(daily_data) > 0:
                daily_range = np.nanmax(daily_data) - np.nanmin(daily_data)
                daily_groups.append(daily_range)
        return np.nanmean(daily_groups) if daily_groups else 0.0
    else:
        # For coarser data, just use overall range
        return np.nanmax(data_series) - np.nanmin(data_series)

def estimate_competition_factor(smap_data, aux_data):
    """
    Estimate competition factor from SMAP and ancillary data
    """
    # Calculate proxies from SMAP data
    soil_moisture_amplitude = calculate_diurnal_range(smap_data['soil_moisture'])
    evaporation_fraction = smap_data['ET'] / smap_data['PET']  # Actual/Potential ET
    vegetation_density = aux_data['LAI']  # Leaf Area Index from MODIS
    
    # Empirical relationship based on water stress patterns
    # Higher diurnal moisture amplitude → more competition
    # Lower evaporation fraction → more competition  
    # Higher vegetation density → more competition
    
    competition_factor = (
        0.5 * soil_moisture_amplitude / np.max(soil_moisture_amplitude) +
        1.0 * (1 - evaporation_fraction) +
        0.8 * vegetation_density / np.max(vegetation_density)
    )
    
    return np.clip(competition_factor, 0.1, 5.0)  # Reasonable bounds



def extract_smap_competition_metrics(smap_file_path, modis_lai_data):
    """
    Extract competition-relevant metrics from SMAP HDF5 files
    """
    with h5py.File(smap_file_path, 'r') as f:
        # Soil moisture data
        sm_data = f['Soil_Moisture_Retrieval_Data']['soil_moisture'][:]
        sm_quality = f['Soil_Moisture_Retrieval_Data']['retrieval_qual_flag'][:]
        
        # Mask low quality data
        sm_data[sm_quality > 0] = np.nan
        
        # Calculate diurnal amplitude (proxy for competition intensity)
        if len(sm_data) >= 8:  # Assuming 3-hourly data
            diurnal_amplitude = np.nanmax(sm_data) - np.nanmin(sm_data)
        else:
            diurnal_amplitude = np.nan
            
        # Get ET data if available  
        try:
            et_data = f['ET_Data']['evapotranspiration'][:]
            pet_data = f['ET_Data']['potential_et'][:]
            et_ratio = np.nanmean(et_data) / np.nanmean(pet_data) if np.nanmean(pet_data) > 0 else 1.0
        except:
            et_ratio = 1.0  # Default if ET data unavailable
    
    return {
        'diurnal_amplitude': diurnal_amplitude,
        'et_ratio': et_ratio,
        'vegetation_density': modis_lai_data  # From separate MODIS data
    }

def smap_based_competition_factor(smap_metrics):
    """
    Calculate competition factor from SMAP metrics
    """
    # Normalize metrics
    amp_norm = smap_metrics['diurnal_amplitude'] / 0.2  # Normalize to typical range
    et_norm = 1 - smap_metrics['et_ratio']  # Water stress index
    veg_norm = smap_metrics['vegetation_density'] / 6.0  # Normalize LAI
    
    # Weighted combination (weights can be calibrated)
    alpha = 0.7 * amp_norm + 1.2 * et_norm + 0.9 * veg_norm
    
    return np.clip(alpha, 0.3, 4.0)

# Calibration using flux tower data
def calibrate_competition_factor(flux_tower_data, smap_metrics):
    """
    Calibrate competition factor using eddy covariance data
    """
    # Use observed water use efficiency from flux towers
    observed_wue = flux_tower_data['GPP'] / flux_tower_data['ET']
    
    # Find alpha that best matches observed patterns
    best_alpha = None
    best_error = float('inf')
    
    for alpha_candidate in np.linspace(0.1, 5.0, 50):
        predicted_wue = run_game_theory_model(alpha_candidate, smap_metrics)
        error = np.mean((predicted_wue - observed_wue)**2)
        
        if error < best_error:
            best_error = error
            best_alpha = alpha_candidate
    
    return best_alpha


# Ecosystem Type	Expected α Range	SMAP/MODIS/met Indicators
# Desert	        0.3 - 1.0	        Low LAI, high diurnal amplitude
# Grassland	        1.0 - 2.0	        Moderate LAI, seasonal ET
# Forest	        2.0 - 3.5	        High LAI, stable soil moisture
# Dense Tropical	3.0 - 4.0	        Very high LAI, high ET
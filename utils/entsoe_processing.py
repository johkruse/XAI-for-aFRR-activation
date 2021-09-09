import pandas as pd
from matplotlib import pyplot as plt 

def drop_non_existent(data):
    
    non_existent = ((data.isnull()) | (data==0)).all()
    data = data.loc[:,~non_existent]
    
    return data

def resample_time_series_on_index(data, new_index, interpol_method):
        
    time_resol = new_index[1] - new_index[0]
    resampled_data = pd.DataFrame(columns=data.columns, index=new_index)
    data = data.sort_index()

    for col in data.columns:
    
        # Get existing values for specific column
        col_data = data.loc[:,col].dropna()
        if not col_data.empty:
            
            # Determine (most coarse) time resolution of column data 
            # (larger distances between time stamps can occur do to missing data,
            # but we try to filter these by assuming minimum resolution of 1 hour.
            # Note that the time series can have time-varying resolution due to market changes.)
            index_dts = (col_data.index[1:]-col_data.index[:-1]).unique()
            index_dts = index_dts[index_dts<='1H']
            data_time_resol = index_dts.max()
            
            # Upsample or downsample to obtain "time_resol" resolution
            if time_resol < data_time_resol:
                fill_limit = data_time_resol // time_resol -1
                col_data = col_data.resample(time_resol).interpolate(method=interpol_method,
                                                                    limit=fill_limit)
                print('Upsample', col, 'from resolution', data_time_resol)
            if time_resol > data_time_resol:
                col_data = col_data.resample(time_resol).mean()
                print('Downsample', col, 'from resolution', data_time_resol)
            
            resampled_data.loc[:,col] = col_data
        
    return resampled_data

def save_region_variable_contrib(contribution, region, path_to_hdf_file, path_to_doc_folder):
    
    contrib_info = pd.DataFrame(columns=['region','variable','nan_ratio','number_neg_vals', 'mean'])
    
    # Apply a final correction to the GB data 
    # (gen_other and gen_biomass were apparently split into two data series at some point)
    if region=='GB BZN' and ('gen_other' in contribution.columns):
        corrected_gen_other = contribution.gen_other.add(contribution.gen_biomass, fill_value=0)
        contribution.loc[:, 'gen_other'] = corrected_gen_other
        contribution.loc[:, 'gen_biomass'] = 0
    
    # Drop non-existent columns
    contribution = drop_non_existent(contribution)
    
    # Save (region, variable)-contributions to seperate files and document download results
    if not contribution.empty:
        
        contribution.sort_index(inplace=True)
        
        for var in contribution.columns:
            
            contrib_name = '{}_{}'.format(region.replace('-', '_').replace(' ','_').replace('(','_').replace(')','_'),
                                        var)
            contribution.loc[:,var].to_hdf(path_to_hdf_file, key=contrib_name, mode='a', complevel=9)
            
            # Extract infos, e.g. nan-ratio
            contrib_info.loc[contrib_name,'nan_ratio'] = contribution.loc[:,var].isnull().sum()/contribution.shape[0]
            contrib_info.loc[contrib_name, 'variable'] = var
            contrib_info.loc[contrib_name, 'region'] = region
            contrib_info.loc[contrib_name, 'number_neg_vals'] = (contribution.loc[:,var]<0).sum()
            contrib_info.loc[contrib_name, 'mean'] = contribution.loc[:,var].mean()

            # Document the download results
            contribution.loc[:,var].plot()
            plt.savefig(path_to_doc_folder+contrib_name+'.png', dpi=200,  bbox_inches='tight')
            plt.close()
        
    return contrib_info


def extract_region_variable_contrib(entsoe_data_folder, file_type, region, variable_name, pivot_column_name,
                                    column_rename_dict, area_neighbors, start_time, end_time,
                                    time_resolution='1H', interpol_method='pad'):
     
    print('--------- ',region, ' ----------')            
    contribution = pd.DataFrame()

    # Iterate over files containing different months and years
    for date_index in pd.date_range(start_time,end_time,freq='M'):

        file = entsoe_data_folder + '{}_{:02d}_{}.csv'.format(date_index.year,
                                                                    date_index.month,
                                                                    file_type)
        
        print('\r'+file, end="\r", flush=True)
        
        # Read data
        data = pd.read_csv(file, sep='\t', encoding = "utf-8", header=0,
                            index_col=False)

        # Select rows with correct region name
        if 'InAreaName' in data.columns:
            # Treat the data as flows
            into_area = (data.InAreaName==region) & (data.OutAreaName.isin(area_neighbors))
            out_of_area = (data.OutAreaName==region) & (data.InAreaName.isin(area_neighbors))
            data = data[into_area | out_of_area]
        else:
            # Treat the data as nodal variable
            data = data[(data.AreaName==region)]

        # Extract data and setup datetime index
        if pivot_column_name:
            # Data contains multiple features
            data = data.pivot(index='DateTime', columns=pivot_column_name,
                              values=variable_name)
            data.index= pd.to_datetime(data.index)
        else:
            # Data contains only one feature
            data.index= pd.to_datetime(data.DateTime)
            data = data.loc[:,[variable_name]]

        # Append time steps to contribution
        contribution = contribution.append(data, verify_integrity=True)
 
    print('\n')
    
    # Resample to target resolution and align with full index
    full_time_index = pd.date_range(start_time,end_time,freq=time_resolution)
    contribution = resample_time_series_on_index(contribution, full_time_index,
                                                 interpol_method=interpol_method)
    
    # Additionally for flow data: Extract net flow balance
    # (Skip NaNs since NaNs in the flows mostly indicate
    # that a certain line was still under construction.)
    # TODO: Separate NaNs due to "missing data point" and "line not yet constructed"
    if 'InAreaName' in contribution.columns.names:
        in_region_flow = contribution.xs(region, level='InAreaName', axis=1)
        out_of_region_flow = contribution.xs(region, level='OutAreaName', axis=1)
        contribution = in_region_flow - out_of_region_flow
        contribution = contribution.sum(1, skipna=True).to_frame()

    # Additionally for generation data: Extract pumped hydro consumption 
    if 'ActualGenerationOutput' in contribution.columns:
        if ('ActualConsumption','Hydro Pumped Storage') in contribution.columns:
            pumped_hydro_consumption = contribution['ActualConsumption',
                                                    'Hydro Pumped Storage']
            pumped_hydro_consumption = pumped_hydro_consumption.to_frame('pumped_hydro_consumption')
            contribution = contribution['ActualGenerationOutput'].join(pumped_hydro_consumption) 
        else:
            contribution = contribution['ActualGenerationOutput']
        
    # Rename columns
    contribution = contribution.rename(columns=column_rename_dict)    
        
    print('\n')
    return contribution


def calc_mean_bzn_load(entsoe_data_folder, region, start_time, end_time):
        
    print('--------- ',region, ' ----------') 
    
    data_points = 0
    load_sum =0
        
    # Iterate over files containing different months and years 
    for date_index in pd.date_range(start_time,end_time,freq='M'):


        file = entsoe_data_folder + '{}_{:02d}_ActualTotalLoad_6.1.A.csv'.format(date_index.year,
                                                                    date_index.month)
        
        print('\r'+file, end="\r", flush=True)

        # Read load data
        data = pd.read_csv(file, sep='\t', encoding = "utf-8", header=0,
                            index_col=False)
        
        # Extract load data 
        # (We use German country data for DE_AT_LU/ DE_LU bidding zones as we need a unique weight
        # for the price time series)
        if region=='DE-LU BZN':
            data = data[data.AreaName=='DE CTY']
        else:
            data = data[data.AreaName==region]

        # Add up load sum
        data = data.loc[:,'TotalLoadValue']
        if data.notnull().any():
            data_points += data.notnull().sum()
            load_sum += data.sum()
    
    print('\n')
        
    # Calculate mean load for region
    if data_points!=0:
        mean_bzn_load = load_sum / data_points
    else:
        mean_bzn_load=None
        
    return mean_bzn_load


def aggregate_external_features(region_contrib_path,mean_bzn_load, contrib_info, output_data ,ignore_list,
                                start_time, end_time, time_resolution='1H',
                                final_nan_ratio_limit = 0.3):
    
    time_index = pd.date_range(start_time,end_time,freq=time_resolution)
    length = len(time_index)
    
    # Initialize final input data and omitted contributions (for documentation)
    raw_input_data = pd.DataFrame(index=time_index)
    omitted_contribs = pd.DataFrame(columns=['region','variable','mean', 'ratio_var', 'ratio_load'] ) 
    
    # Initialize weighted averaging of prices
    total_mean_load = 0
    weight = 1
    
    # Load invalid output indices and calculate their final_nan_ratio   
    invalid_outputs = output_data.isnull().any(axis=1)
    invalid_outputs.index = invalid_outputs.index.tz_convert('UTC').tz_localize(None)
    final_nan_ratio = invalid_outputs.sum() / output_data.shape[0]
    potential_nan_ratio = 1
     
    # Add (region,variable)-contributions with increasing nan-ratio until 
    # final nan ratio exceeds threshold 
    for contrib_name,contrib in contrib_info.sort_values('nan_ratio').iterrows():
        
        if contrib.variable not in ignore_list:
        
            # If final_nan_ratio (still) low enough, load potential region contribution
            if final_nan_ratio < final_nan_ratio_limit:
                
                # Read contribution data
                data = pd.read_hdf(region_contrib_path, key=contrib_name)  
                potential_input_data = raw_input_data.copy()
                
                # Choose weights for averaging or summation
                if contrib.variable=='prices_day_ahead':
                    weight = mean_bzn_load[contrib.region]
                else:
                    weight = 1
                    
                # Add potential contribution
                if contrib.variable not in potential_input_data.columns:
                    potential_input_data.loc[:,contrib.variable] = data*weight
                else:
                    potential_input_data.loc[:,contrib.variable] += data*weight
                
                # Calculate (potential) nan-ratio in final data set
                potential_nan_ratio = (potential_input_data.isnull().any(axis=1) | invalid_outputs).sum() / length
                
            # If potential (new) nan-ratio low enough, add contribution to final data 
            if potential_nan_ratio < final_nan_ratio_limit:
                
                raw_input_data = potential_input_data.copy()  
                final_nan_ratio = potential_nan_ratio   
                
                if contrib.variable=='prices_day_ahead':
                    total_mean_load += weight  
                    
                print('Added: ', contrib_name, ' | final nu. of points: {}'.format(length-int(final_nan_ratio*length)))
            
            # If potential (new) nan-ratio is too high, omitting new contribution 
            else:
                # Save the omitted mean contributions 
                omitted_contribs.loc[contrib_name] = contrib.loc[['region', 'variable','mean']]
                
                # Save relative omitted mean contribution 
                if contrib.variable in raw_input_data:
                    raw_input_mean = raw_input_data.loc[:,contrib.variable].mean()
                    omitted_contribs.loc[contrib_name, 'ratio_var'] = contrib.loc['mean'] / raw_input_mean
                omitted_contribs.loc[contrib_name, 'ratio_load'] = contrib.loc['mean'] / raw_input_data.load_day_ahead.mean()

                print('Omitted: ', contrib_name, ' | potential final nu. of points: {}'.format(length-int(potential_nan_ratio*length)))

    # Apply weighted average to prices
    raw_input_data.loc[:,'prices_day_ahead'] = raw_input_data.prices_day_ahead / total_mean_load
    
    return raw_input_data, omitted_contribs
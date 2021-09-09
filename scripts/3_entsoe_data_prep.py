import os
import pandas as pd 
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('agg')

import sys

sys.path.append('./')

from utils.entsoe_processing import calc_mean_bzn_load,aggregate_external_features
from utils.entsoe_processing import extract_region_variable_contrib, save_region_variable_contrib

#TODO: Move flow aggregation from this pipeline step to 4_external_feature_prep 
# (may better fit to feature engineering)
#TODO: Still has to be tested for area containing multiple regions, e.g. CE

# Areas inlcuding "country" areas
areas = ['DE', 'IGCC_rest']  # 'DE' has to be first 

# Regions connected to the regions within the area
area_neighbors = {
    'DE': ['AT CTY','CH CTY', 'CZ CTY','DK CTY','FR CTY','LU CTY',
           'NL CTY','PL CTY', 'SE CTY', 'NO CTY', 'BE CTY'],
    'IGCC_rest': []
    }

# Path to ENTSO-E downloaded data
entsoe_data_folder = '../../External_data/ENTSO-E/' 

# Path for generated features and targets
folder = './data/{}/'

# Documentation of data download 
doc_folder = folder + 'documentation_of_data_download/'

# Create folders
if not os.path.exists(entsoe_data_folder):
    os.makedirs(entsoe_data_folder)
for area in areas:
    if not os.path.exists(doc_folder.format(area)):
        os.makedirs(doc_folder.format(area))
     
# Datetime parameters for input data preparation
# (entso-e timestamps are UTC)
start = pd.Timestamp('2019-07-01 00:00:00')
end = pd.Timestamp('2021-07-31 00:00:00')
time_resol = pd.Timedelta('15min') 
     
# Map codes of selected regions within each area 
# (The "area type" keys are actually not used anymore, but are kept for better readability.)
general_area_names = {
    'DE': {
        'CTY':['DE CTY']
        },
    'IGCC_rest': {}
    }

# Map codes of bidding zones within each area (for price data, including rest-of-IGCC )
# (Excluded  'PL'  as this country does not (always) have Euro-data within our time frame. 
# All other regions have price data in Euro.
# IGCC members: https://www.entsoe.eu/network_codes/eb/imbalance-netting/
bzn_area_names = {
    'DE': {
        'BZN':['DE-LU BZN']
        },
    'IGCC_rest': {
        'BZN':['AT BZN', 'BE BZN' , 'CH BZN' , 'CZ BZN',  'ES BZN',
               'FR BZN',  'HR BZN',  'HU BZN',
               'IT-North BZN','IT-Centre-North BZN',
               'IT-Centre-South BZN','IT-South BZN','IT-Sicily BZN',
               'GR BZN', 'NL BZN',  'PT BZN',  
               'SI BZN', 'SK BZN', 'DK1 BZN']
        }
    }

# Cross-border connections 
# (Here, the regions at the border of the area are defined, which is in this case only Germany)
border_regions_area_names = {
    'DE':{
        'CTY': ['DE CTY']
        },
    'IGCC_rest': {}
    }


# Map codes of selected regions within each area (including rest-of-IGCC)
general_area_names_with_CE = {
    'DE': {
        'CTY':['DE CTY']
        },
    'IGCC_rest': {
        'CTY': ['AT CTY', 'BE CTY' , 'CH CTY' , 'CZ CTY',  'ES CTY',
                'FR CTY',  'HR CTY',  'HU CTY',  'IT CTY', 'GR CTY',   'NL CTY', 'PL CTY', 'PT CTY',  
                'SI CTY', 'SK CTY', 'DK CTY' ],
        }
    }

# Old and new names for generation types
new_gen_type_cols = ['gen_biomass', 'gen_lignite', 'gen_coal_gas', 'gen_gas',
                     'gen_hard_coal', 'gen_oil', 'gen_oil_shale', 'gen_fossil_peat',
                     'gen_geothermal', 'gen_pumped_hydro', 'gen_run_off_hydro',
                     'gen_reservoir_hydro', 'gen_marine', 'gen_nuclear', 'gen_other_renew',
                     'gen_solar', 'gen_waste', 'gen_wind_off', 'gen_wind_on', 'gen_other',
                     'pumped_hydro_consumption']
old_gen_type_cols =    ['Biomass', 'Fossil Brown coal/Lignite', 'Fossil Coal-derived gas',
                        'Fossil Gas', 'Fossil Hard coal', 'Fossil Oil', 'Fossil Oil shale',
                        'Fossil Peat', 'Geothermal', 'Hydro Pumped Storage', 
                        'Hydro Run-of-river and poundage', 'Hydro Water Reservoir',
                        'Marine', 'Nuclear', 'Other renewable', 'Solar', 'Waste',
                        'Wind Offshore', 'Wind Onshore', 'Other', 'pumped_hydro_consumption']

# Old and new names for renewable forecast 
new_ren_forecast_cols = ['solar_day_ahead','wind_off_day_ahead', 'wind_on_day_ahead']
old_ren_forecast_cols = ['Solar', 'Wind Offshore', 'Wind Onshore']

# Properties of downloaded ENTSO-E data files
# (The key indicates the file. 'vals' indicates the column containing the feature values.
# 'value_cols' indicates the column to use for pivot stacked data. 'new_cols' is a dictionary
# to rename the columns. 'regions' is a dictionary indicating the regions used for 
# aggregation of this feature. Interpolation indicates the method used for upsampling
# (only for upsampling, not for missing data points)).
file_props = {
    "ActualTotalLoad_6.1.A": {
        'vals': 'TotalLoadValue',
        'value_cols':None,
        'new_cols': {'TotalLoadValue':'load'},
        'regions':general_area_names,
        'interpolation':'slinear'
        },
    "DayAheadTotalLoadForecast_6.1.B": {
        'vals':'TotalLoadValue',
        'value_cols':None,
        'new_cols': {'TotalLoadValue':'load_day_ahead'},
        'regions':general_area_names_with_CE,
        'interpolation':'slinear'
        },
    "DayAheadAggregatedGeneration_14.1.C": {
        'vals':'ScheduledGeneration',
        'value_cols':None,
        'new_cols': {'ScheduledGeneration':'scheduled_gen_total'},
        'regions':general_area_names_with_CE,
        'interpolation':'pad'
        },
    "AggregatedGenerationPerType_16.1.B_C": {
        'vals': ['ActualGenerationOutput', 'ActualConsumption'],
        'value_cols':'ProductionType',
        'new_cols': dict(zip(old_gen_type_cols, new_gen_type_cols)),
        'regions':general_area_names,
        'interpolation': 'pad' 
        },
    "DayAheadGenerationForecastForWindAndSolar_14.1.D": {
        'vals': 'AggregatedGenerationForecast',
        'value_cols':'ProductionType',
        'new_cols': dict(zip(old_ren_forecast_cols, new_ren_forecast_cols)),
        'regions':general_area_names_with_CE,
        'interpolation':'slinear'
        },
    "DayAheadPrices_12.1.D": {
        'vals':'Price',
        'value_cols':None,
        'new_cols':{'Price':'prices_day_ahead'},
        'regions':bzn_area_names,
        'interpolation':'pad'
        }, 
    "DayAheadCommercialSchedules_12.1.F": {
        'vals':'Capacity',
        'value_cols':['InAreaName', 'OutAreaName'],
        'new_cols':{0: 'import_export_day_ahead'},
        'regions':border_regions_area_names,
        'interpolation':'pad'
        },
     "TotalCommercialSchedules_12.1.F": {
         'vals':'Capacity',
         'value_cols':['InAreaName', 'OutAreaName'],
         'new_cols':{0: 'import_export_total'},
         'regions':border_regions_area_names ,
         'interpolation':'pad'       
         },
    "PhysicalFlows_12.1.G": {
        'vals':'FlowValue',
        'value_cols':['InAreaName', 'OutAreaName'],
        'new_cols':{0: 'cross_border_flow'},
        'regions': border_regions_area_names  ,
        'interpolation':'pad'      
        }
    }


#### Collect and save region-variable contributions ####

print('\nCollect region contributions\n')

# Iterate over areas
for area in areas:

    print('###################### ', area, ' ##########################')
    
    # Prepare processing for this area
    if os.path.exists(entsoe_data_folder + 'region_contributions_{}.h5'.format(area)):
        os.remove(entsoe_data_folder + 'region_contributions_{}.h5'.format(area))
    contrib_info = pd.DataFrame(columns=['region','variable','nan_ratio','number_neg_vals', 'mean'])
    
    #Iterate over file types from ENTSO-E server
    for file_type, file_type_props in file_props.items():

        print('######## ',file_type,' #########')
       
        # Iterate over region types within the area
        for region_type, region_list in file_type_props['regions'][area].items():  
            
            # Iterate over regions of this type
            for region in region_list:            
         
                
                reg_var_contribution = extract_region_variable_contrib(entsoe_data_folder = entsoe_data_folder,
                                                                       file_type = file_type, region = region,
                                                                       variable_name = file_type_props['vals'],
                                                                       pivot_column_name = file_type_props['value_cols'],
                                                                       column_rename_dict = file_type_props['new_cols'],
                                                                       area_neighbors = area_neighbors[area],
                                                                       start_time = start, end_time = end,
                                                                       time_resolution = time_resol,
                                                                       interpol_method=file_type_props['interpolation'])
                
                hdf_file = entsoe_data_folder + 'region_contributions_{}.h5'.format(area)
                
                region_contrib_info = save_region_variable_contrib(contribution = reg_var_contribution,
                                                                   region = region,
                                                                   path_to_hdf_file = hdf_file,
                                                                   path_to_doc_folder = doc_folder.format(area))
                
                contrib_info = contrib_info.append(region_contrib_info)
                
                
    contrib_info.to_csv(doc_folder.format(area)+'contribution_info.csv')
    

#### Collect mean load for bidding zones ###
# (later used for weighted aggregation)

print('\nCollect mean load for bidding zones\n')

for area in areas:
    
    print('########### ', area, ' ###########')

    mean_bzn_load = pd.Series()
    
    # Iterate over bidding zones
    for region in bzn_area_names[area]['BZN']:  
                  
        mean_bzn_load.loc[region] = calc_mean_bzn_load(entsoe_data_folder,region,start,end)


    # Save results
    mean_bzn_load.to_csv(doc_folder.format(area)+'mean_bzn_load.csv', header=False) 
     


#### Aggregate features within each area ####

print('\nAggregate data within each area\n')

for area in areas:
    
    print('############### ', area, ' ###################')

    mean_bzn_load = pd.read_csv(doc_folder.format(area) + 'mean_bzn_load.csv', header=None,
                                index_col=0, squeeze=True) 
    region_contrib_path = entsoe_data_folder + 'region_contributions_{}.h5'.format(area)
    contrib_info = pd.read_csv(doc_folder.format(area) + 'contribution_info.csv', index_col=0) 
    
    # We control for the nan share in the final data set, so we have to combine DE and outputs 
    # in the IGCC_rest aggregation. 
    if area=='DE':
        other_data = pd.read_hdf(folder.format(area) + 'outputs.h5')
    if area=='IGCC_rest':
        other_data = pd.read_hdf(folder.format('DE') + 'outputs.h5').join(pd.read_hdf(folder.format('DE') + 'raw_input_data.h5'))
    
    
    
    raw_input_data, omitted_contribs = aggregate_external_features(region_contrib_path= region_contrib_path,
                                                                   mean_bzn_load=mean_bzn_load,
                                                                   contrib_info=contrib_info,
                                                                   output_data= other_data,
                                                                   start_time=start, end_time=end,
                                                                   time_resolution=time_resol,
                                                                   ignore_list=[],
                                                                   final_nan_ratio_limit=0.37)

    # Save (final) raw input data
    raw_input_data.index = raw_input_data.index.tz_localize('UTC')
    raw_input_data.to_hdf(folder.format(area)+'raw_input_data.h5', key='df')
    omitted_contribs.to_csv(doc_folder.format(area) + 'omitted_contributions.csv')


#### Analysis of omitted contributions (during aggregation)  ####

print('\nAnalysis of omitted contributions\n')

for area in areas:
    omitted_contribs = pd.read_csv(doc_folder.format(area)+'omitted_contributions.csv')
    
    try:
        fig,ax=plt.subplots(figsize=(9,4))
        data = omitted_contribs.pivot(index='variable', columns='region', values='ratio_var')
        data = data.loc[:,data.sum()>0.001]
        data.plot.bar(stacked=True, legend=False, cmap='tab20', ax=ax)
        plt.xlabel('')
        plt.ylabel('Mean omission / variable mean')
        plt.legend(bbox_to_anchor=(0.8,0.5,0.5,0.5))
        plt.savefig(doc_folder.format(area) + 'omitted_contributions_relative_to_variable_mean.svg', bbox_inches='tight')
        plt.show()
        plt.close()

        fig,ax=plt.subplots(figsize=(9,4))
        data = omitted_contribs.pivot(index='variable', columns='region', values='ratio_load')
        data = data.loc[:,data.sum()>0.001].drop(index=['prices_day_ahead'])
        data.plot.bar(stacked=True, legend=False, cmap='tab20', ax=ax)
        plt.xlabel('')
        plt.ylabel('Mean omission / total mean load')
        plt.legend(bbox_to_anchor=(0.8,0.5,0.5,0.5))
        plt.savefig(doc_folder.format(area) + 'omitted_contributions_relative_to_mean_load.svg', bbox_inches='tight')

    except:
        pass


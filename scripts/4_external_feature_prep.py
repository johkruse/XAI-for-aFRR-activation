import pandas as pd 
import matplotlib
matplotlib.use('agg')

# Define features for specific models
# (The code includes more models than actually presented in the paper)
full_model_igcc_cols = ['gen_other',  'gen_waste', 'gen_nuclear',
                'gen_biomass', 'gen_gas', 'gen_run_off_hydro', 'gen_oil',
                'gen_pumped_hydro', 'gen_other_renew', 'gen_reservoir_hydro',
                'gen_hard_coal',  'gen_geothermal', 'gen_lignite',
                'total_gen', 'synchronous_gen',
                'total_gen_ramp', 'other_ramp', 
                'waste_ramp', 'nuclear_ramp', 'biomass_ramp', 'gas_ramp',
                'run_off_hydro_ramp', 'oil_ramp', 'pumped_hydro_ramp',
                'other_renew_ramp', 'reservoir_hydro_ramp', 'hard_coal_ramp',
                'geothermal_ramp', 'lignite_ramp',
                'forecast_error_total_gen',
                'forecast_error_total_gen_ramp', 
                'gen_solar', 'gen_wind_on', 'gen_wind_off', 
                'load', 'load_ramp', 'solar_ramp', 'wind_on_ramp',
                'wind_off_ramp', 'forecast_error_wind_on', 'forecast_error_wind_off',
                'forecast_error_solar', 
                'forecast_error_load', 'forecast_error_load_ramp',
                'forecast_error_wind_off_ramp',
                'forecast_error_wind_on_ramp', 'forecast_error_solar_ramp',
                'pumped_hydro_consumption', 'pumped_hydro_consumption_ramp',
                'solar_day_ahead', 'wind_on_day_ahead', 'scheduled_gen_total',
                'prices_day_ahead', 'load_day_ahead', 'wind_off_day_ahead', 
                'load_ramp_day_ahead', 'total_gen_ramp_day_ahead',
                'wind_off_ramp_day_ahead', 'wind_on_ramp_day_ahead',
                'solar_ramp_day_ahead', 'price_ramp_day_ahead', 'import_export_day_ahead', 
                'import_export_ramp_day_ahead', 'cross_border_flow', 'cross_border_flow_ramp',
                'forecast_error_flow', 'forecast_error_flow_ramp', 
                'import_export_total', 'import_export_total_ramp',
                'unscheduled_flow', 'unscheduled_flow_ramp',
                'load_ramp_igcc_day_ahead', 'total_gen_ramp_igcc_day_ahead', 'wind_off_ramp_igcc_day_ahead',
                'wind_on_ramp_igcc_day_ahead', 'solar_ramp_igcc_day_ahead',  'price_ramp_igcc_day_ahead',
                'load_igcc_day_ahead', 'scheduled_gen_total_igcc', 'wind_off_igcc_day_ahead',
                'wind_on_igcc_day_ahead', 'solar_igcc_day_ahead',  'price_igcc_day_ahead',
                'weekday', 'hour']

full_model_cols = ['gen_other',  'gen_waste', 'gen_nuclear',
                'gen_biomass', 'gen_gas', 'gen_run_off_hydro', 'gen_oil',
                'gen_pumped_hydro', 'gen_other_renew', 'gen_reservoir_hydro',
                'gen_hard_coal',  'gen_geothermal', 'gen_lignite',
                'total_gen', 'synchronous_gen',
                'total_gen_ramp', 'other_ramp', 
                'waste_ramp', 'nuclear_ramp', 'biomass_ramp', 'gas_ramp',
                'run_off_hydro_ramp', 'oil_ramp', 'pumped_hydro_ramp',
                'other_renew_ramp', 'reservoir_hydro_ramp', 'hard_coal_ramp',
                'geothermal_ramp', 'lignite_ramp',
                'forecast_error_total_gen',
                'forecast_error_total_gen_ramp', 
                'gen_solar', 'gen_wind_on', 'gen_wind_off', 
                'load', 'load_ramp', 'solar_ramp', 'wind_on_ramp',
                'wind_off_ramp', 'forecast_error_wind_on', 'forecast_error_wind_off',
                'forecast_error_solar', 
                'forecast_error_load', 'forecast_error_load_ramp',
                'forecast_error_wind_off_ramp',
                'forecast_error_wind_on_ramp', 'forecast_error_solar_ramp',
                'pumped_hydro_consumption', 'pumped_hydro_consumption_ramp',
                'solar_day_ahead', 'wind_on_day_ahead', 'scheduled_gen_total',
                'prices_day_ahead', 'load_day_ahead', 'wind_off_day_ahead', 
                'load_ramp_day_ahead', 'total_gen_ramp_day_ahead',
                'wind_off_ramp_day_ahead', 'wind_on_ramp_day_ahead',
                'solar_ramp_day_ahead', 'price_ramp_day_ahead', 'import_export_day_ahead', 
                'import_export_ramp_day_ahead', 'cross_border_flow', 'cross_border_flow_ramp',
                'forecast_error_flow', 'forecast_error_flow_ramp', 
                'import_export_total', 'import_export_total_ramp',
                'unscheduled_flow', 'unscheduled_flow_ramp',
                'weekday', 'hour']


extended_model_igcc_cols = ['gen_solar', 'gen_wind_on', 'gen_wind_off', 
                    'load', 'load_ramp', 'solar_ramp', 'wind_on_ramp',
                    'wind_off_ramp', 'forecast_error_wind_on', 'forecast_error_wind_off',
                    'forecast_error_solar', 
                    'forecast_error_load', 'forecast_error_load_ramp',
                    'forecast_error_wind_off_ramp',
                    'forecast_error_wind_on_ramp', 'forecast_error_solar_ramp',
                    'solar_day_ahead', 'wind_on_day_ahead', 'scheduled_gen_total',
                    'prices_day_ahead', 'load_day_ahead', 'wind_off_day_ahead', 
                    'load_ramp_day_ahead', 'total_gen_ramp_day_ahead',
                    'wind_off_ramp_day_ahead', 'wind_on_ramp_day_ahead',
                    'solar_ramp_day_ahead', 'price_ramp_day_ahead', 'import_export_day_ahead', 
                    'import_export_ramp_day_ahead', 'cross_border_flow', 'cross_border_flow_ramp',
                    'forecast_error_flow', 'forecast_error_flow_ramp', 
                    'import_export_total', 'import_export_total_ramp',
                    'unscheduled_flow', 'unscheduled_flow_ramp',
                    'load_ramp_igcc_day_ahead', 'total_gen_ramp_igcc_day_ahead', 'wind_off_ramp_igcc_day_ahead',
                    'wind_on_ramp_igcc_day_ahead', 'solar_ramp_igcc_day_ahead',  'price_ramp_igcc_day_ahead',
                    'load_igcc_day_ahead', 'scheduled_gen_total_igcc', 'wind_off_igcc_day_ahead',
                    'wind_on_igcc_day_ahead', 'solar_igcc_day_ahead',  'price_igcc_day_ahead',
                    'weekday', 'hour']

extended_model_cols = ['gen_solar', 'gen_wind_on', 'gen_wind_off', 
                    'load', 'load_ramp', 'solar_ramp', 'wind_on_ramp',
                    'wind_off_ramp', 'forecast_error_wind_on', 'forecast_error_wind_off',
                    'forecast_error_solar', 
                    'forecast_error_load', 'forecast_error_load_ramp',
                    'forecast_error_wind_off_ramp',
                    'forecast_error_wind_on_ramp', 'forecast_error_solar_ramp',
                    'solar_day_ahead', 'wind_on_day_ahead', 'scheduled_gen_total',
                    'prices_day_ahead', 'load_day_ahead', 'wind_off_day_ahead', 
                    'load_ramp_day_ahead', 'total_gen_ramp_day_ahead',
                    'wind_off_ramp_day_ahead', 'wind_on_ramp_day_ahead',
                    'solar_ramp_day_ahead', 'price_ramp_day_ahead', 'import_export_day_ahead', 
                    'import_export_ramp_day_ahead', 'cross_border_flow', 'cross_border_flow_ramp',
                    'forecast_error_flow', 'forecast_error_flow_ramp', 
                    'import_export_total', 'import_export_total_ramp',
                    'unscheduled_flow', 'unscheduled_flow_ramp',
                    'weekday', 'hour']

day_ahead_model_igcc_cols = ['load_day_ahead', 'scheduled_gen_total','prices_day_ahead',
                            'solar_day_ahead','wind_off_day_ahead', 'wind_on_day_ahead',
                            'import_export_day_ahead',
                            'load_ramp_day_ahead', 'total_gen_ramp_day_ahead',
                            'wind_off_ramp_day_ahead', 'wind_on_ramp_day_ahead',
                            'solar_ramp_day_ahead', 'price_ramp_day_ahead', 'import_export_ramp_day_ahead',
                            'load_ramp_igcc_day_ahead', 'total_gen_ramp_igcc_day_ahead', 'wind_off_ramp_igcc_day_ahead',
                            'wind_on_ramp_igcc_day_ahead', 'solar_ramp_igcc_day_ahead',  'price_ramp_igcc_day_ahead',
                            'load_igcc_day_ahead', 'scheduled_gen_total_igcc', 'wind_off_igcc_day_ahead',
                            'wind_on_igcc_day_ahead', 'solar_igcc_day_ahead',  'price_igcc_day_ahead', 
                            'weekday', 'hour']
day_ahead_model_cols = ['load_day_ahead', 'scheduled_gen_total','prices_day_ahead',
                        'solar_day_ahead','wind_off_day_ahead', 'wind_on_day_ahead',
                        'import_export_day_ahead',
                        'load_ramp_day_ahead', 'total_gen_ramp_day_ahead',
                        'wind_off_ramp_day_ahead', 'wind_on_ramp_day_ahead',
                        'solar_ramp_day_ahead', 'price_ramp_day_ahead', 'import_export_ramp_day_ahead',
                        'weekday', 'hour']


# Time zones of aFRR data
tz = 'CET'

# Setup folder paths to raw input data 
folder = './data/{}/'

# Load the pre-processed external features for DE
data = pd.read_hdf(folder.format('DE') + 'raw_input_data.h5')

# Convert index to local timezone of frequency data
data.index = data.index.tz_convert(tz)

####  Additional engineered features ###

# Time
#data['month'] = data.index.month
data['weekday'] =  data.index.weekday
data['hour'] =  data.index.hour + data.index.minute/60.
#data['trend'] = (data.index.year - data.index.year[0])*12 + data.index.month -data.index.month[0]

# Total generation
data['total_gen'] = data.filter(regex='^gen').sum(axis='columns')


# Inertia proxy - Sum of all synchronous generation
data['synchronous_gen'] = data.total_gen- data.loc[:,['gen_solar',
                                                      'gen_wind_off',
                                                      'gen_wind_on']].sum(axis=1)

# Ramps of load and total generation 
data['load_ramp_day_ahead'] = data.load_day_ahead.diff()
data['load_ramp'] = data.load.diff()
data['total_gen_ramp_day_ahead'] = data.scheduled_gen_total.diff()
data['total_gen_ramp'] = data.total_gen.diff()
data['pumped_hydro_consumption_ramp'] = data.pumped_hydro_consumption.diff()

# Ramps of generaton types
data['wind_off_ramp_day_ahead'] = data.wind_off_day_ahead.diff()
data['wind_on_ramp_day_ahead'] = data.wind_on_day_ahead.diff()
data['solar_ramp_day_ahead'] = data.solar_day_ahead.diff()
gen_ramp_cols = data.filter(regex='^gen').columns.str[4:] + '_ramp'
data[gen_ramp_cols] = data.filter(regex='^gen').diff()

# Price Ramps
data['price_ramp_day_ahead'] = data.prices_day_ahead.diff()

# Flow ramps
data['import_export_ramp_day_ahead'] = data.import_export_day_ahead.diff()
data['cross_border_flow_ramp'] = data.cross_border_flow.diff() 
data['import_export_total_ramp'] = data.import_export_total.diff()

# Forecast errors
data['forecast_error_wind_on'] = data.wind_on_day_ahead - data.gen_wind_on
data['forecast_error_wind_off'] = data.wind_off_day_ahead - data.gen_wind_off
data['forecast_error_wind_off_ramp'] = data.wind_off_ramp_day_ahead - data.wind_off_ramp
data['forecast_error_total_gen'] = data.scheduled_gen_total - data.total_gen
data['forecast_error_load'] = data.load_day_ahead - data.load
data['forecast_error_load_ramp'] = data.load_ramp_day_ahead - data.load_ramp
data['forecast_error_total_gen_ramp'] = data.total_gen_ramp_day_ahead - data.total_gen_ramp
data['forecast_error_wind_on_ramp'] = data.wind_on_ramp_day_ahead - data.wind_on_ramp
data['forecast_error_solar_ramp'] = data.solar_ramp_day_ahead - data.solar_ramp
data['forecast_error_solar'] = data.solar_day_ahead - data.gen_solar
data['forecast_error_flow'] = data.import_export_day_ahead - data.cross_border_flow
data['forecast_error_flow_ramp'] = data.import_export_ramp_day_ahead - data.cross_border_flow_ramp
data['unscheduled_flow'] = data.import_export_total - data.cross_border_flow
data['unscheduled_flow_ramp'] = data.import_export_total_ramp - data.cross_border_flow_ramp

# IGCC rest (without germany) day-ahead data
igcc_rest = pd.read_hdf(folder.format('IGCC_rest') + 'raw_input_data.h5')
igcc_rest.index = igcc_rest.index.tz_convert(tz)
data['load_igcc_day_ahead'] = igcc_rest.load_day_ahead
data['scheduled_gen_total_igcc'] = igcc_rest.scheduled_gen_total
data['wind_off_igcc_day_ahead'] = igcc_rest.wind_off_day_ahead
data['wind_on_igcc_day_ahead'] = igcc_rest.wind_on_day_ahead
data['solar_igcc_day_ahead'] = igcc_rest.solar_day_ahead
data['price_igcc_day_ahead'] = igcc_rest.prices_day_ahead
data['load_ramp_igcc_day_ahead'] = igcc_rest.load_day_ahead.diff()
data['total_gen_ramp_igcc_day_ahead'] = igcc_rest.scheduled_gen_total.diff()
data['wind_off_ramp_igcc_day_ahead'] = igcc_rest.wind_off_day_ahead.diff()
data['wind_on_ramp_igcc_day_ahead'] = igcc_rest.wind_on_day_ahead.diff()
data['solar_ramp_igcc_day_ahead'] = igcc_rest.solar_day_ahead.diff()
data['price_ramp_igcc_day_ahead'] = igcc_rest.prices_day_ahead.diff()

# Save data
data.loc[:, day_ahead_model_igcc_cols ].to_hdf(folder.format('DE')+'inputs_day_ahead_igcc.h5',key='df')
data.loc[:, extended_model_igcc_cols ].to_hdf(folder.format('DE')+'inputs_extended_igcc.h5',key='df')
data.loc[:, full_model_igcc_cols ].to_hdf(folder.format('DE')+'inputs_full_igcc.h5',key='df')
data.loc[:, day_ahead_model_cols ].to_hdf(folder.format('DE')+'inputs_day_ahead.h5',key='df')
data.loc[:, extended_model_cols ].to_hdf(folder.format('DE')+'inputs_extended.h5',key='df')
data.loc[:, full_model_cols ].to_hdf(folder.format('DE')+'inputs_full.h5',key='df')




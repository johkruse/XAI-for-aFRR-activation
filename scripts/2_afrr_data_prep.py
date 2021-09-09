import pandas as pd 
import os
import sys
import glob

sys.path.append('./')

# Time zones of frequency recordings
tzs = {'DE':'CET'}

# Datetime parameters for output data generation
start = pd.Timestamp('2019-07-01 00:00:00', tz='UTC')
end = pd.Timestamp('2021-07-31 00:00:00', tz='UTC')
time_resol = pd.Timedelta('15min')


for area in ['DE']:
    
    print(area)
    
    # prepare folder for output data
    folder = './data/{}/'.format(area)
    if not os.path.exists(folder):
        os.makedirs(folder)    

    # Load FRR data
    outputs = pd.DataFrame()
    for file in glob.glob(folder + 'frr_data/activation/*.csv'): 
        
        print('Loading activated aFRR: ', file)
        frr= pd.read_csv(file , sep=';', low_memory=False,decimal=",")
        dt_index = (frr.DATE + ' ' + frr.TIME_FROM).astype('str')
        dt_index = pd.to_datetime(dt_index, format='%d.%m.%Y %H:%M').dt.tz_localize(tzs[area],
                                                                                    ambiguous='infer')
        frr.index=dt_index
        frr = frr.iloc[:,3:].sort_index()
        outputs = outputs.append(frr)
        
    # Filter aFRR data and shorten column names
    # Use quality assured data "QA", when available, otherwise use operational data ("OP")
    outputs_operat = outputs.filter(regex='AFRR').filter(regex='OP')
    outputs_quality = outputs.filter(regex='AFRR').filter(regex='QA')
    outputs_operat.columns = outputs_operat.columns.str[8:-8]
    outputs_quality.columns = outputs_quality.columns.str[8:-8]
    outputs = outputs_quality.where(outputs_quality.notnull(), outputs_operat)

    # Add additional target variables
    outputs.loc[:,'AFRR_DE_POS'] = outputs.filter(regex='POS').sum(1,skipna=False)
    outputs.loc[:,'AFRR_DE_NEG'] = outputs.filter(regex='NEG').sum(1,skipna=False)

    # Save data reindexed with full datetime index
    full_date_index = pd.date_range(start, end, freq=time_resol, tz='UTC')
    full_date_index = full_date_index.tz_convert(tzs[area])
    outputs = outputs.reindex(index=full_date_index)
    outputs.to_hdf(folder+'outputs.h5', key='df')  
        
    # Prepare aFRR tendering data
    demand_data = pd.DataFrame()
    for file in glob.glob(folder + 'frr_data/demand/*.xlsx'):
        
        print('Loading aFRR demand: ', file)
        
        demand=pd.read_excel(file)
        demand['datetime'] = demand.DATE_FROM.dt.strftime('%Y-%m-%d') + ' ' + demand.PRODUCT.str[4:6] + ':00'
        demand['datetime'] = pd.to_datetime(demand['datetime']).dt.tz_localize(tzs[area], ambiguous='infer')
        demand['PRODUCT'] = demand.PRODUCT.str[:3]
        demand = demand.pivot(index='datetime', columns='PRODUCT', values='GERMANY_BLOCK_DEMAND_[MW]') 
        
        demand_data = demand_data.append(demand)
    
    # Save data reindexed with full datetime index
    
    demand_data = demand_data.reindex(index=full_date_index).ffill(limit=15)
    demand_data = demand_data.rename(columns={'POS':'AFRR_DE_POS', 'NEG':'AFRR_DE_NEG'})
    demand_data.to_hdf(folder+'afrr_demand.h5', key='df')  
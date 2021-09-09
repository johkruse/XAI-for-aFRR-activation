import pandas as pd 
from sklearn.model_selection import train_test_split
import os


# Choose subset of targets if required
targets = ['AFRR_DE_POS', 'AFRR_DE_NEG']


# Areas inlcuding "country" areas
areas = ['DE']


for area in areas:

   print('Processing external features from', area)
   
   # Setup folder for this specific version of train-test data
   folder = './data/{}/'.format(area)
   version_folder = folder + 'version_'+  '2021-08-20/' #pd.Timestamp("today").strftime("%Y-%m-%d") + '/'
   if not os.path.exists(version_folder):
      os.makedirs(version_folder)
   
   # Load data
   X_extended = pd.read_hdf(folder+'inputs_extended.h5').sort_index()
   X_day_ahead = pd.read_hdf(folder + 'inputs_day_ahead.h5').sort_index()
   X_full = pd.read_hdf(folder + 'inputs_full.h5').sort_index()
   X_extended_igcc = pd.read_hdf(folder+'inputs_extended_igcc.h5').sort_index()
   X_day_ahead_igcc = pd.read_hdf(folder + 'inputs_day_ahead_igcc.h5').sort_index()
   X_full_igcc = pd.read_hdf(folder + 'inputs_full_igcc.h5').sort_index()
   y = pd.read_hdf(folder + 'outputs.h5').loc[:,targets].sort_index()
      
   # Drop nan values (use X_full_igcc as it contains all possible features)
   valid_ind =  ~pd.concat([X_full_igcc, y], axis=1).isnull().any(axis=1)
   y =  y[valid_ind]
   
   # Extract a continuous sample at the end
   cont_ind = y.loc['2021-06-01 00:00':].index

   # Train-test split
   train_ind, test_ind = train_test_split(y.drop(index=cont_ind).index, test_size=0.2, random_state=42)   
   print('Using', test_ind.shape[0]//(24*4), 'days of test data.')
   print('Using', train_ind.shape[0]//(24*4), 'days of train data.')
   print('Using', cont_ind.shape[0]//(24*4), 'days of continuous data.')

   # Save data for the different models
   
   for ind_type, ind in zip(['train', 'test', 'cont'], [train_ind, test_ind, cont_ind]):
      
      X_day_ahead.loc[ind].to_hdf(version_folder+'X_{}_day_ahead.h5'.format(ind_type),key='df')
      X_day_ahead_igcc.loc[ind].to_hdf(version_folder+'X_{}_day_ahead_igcc.h5'.format(ind_type),key='df')
      X_extended.loc[ind].to_hdf(version_folder+'X_{}_extended.h5'.format(ind_type),key='df')
      X_extended_igcc.loc[ind].to_hdf(version_folder+'X_{}_extended_igcc.h5'.format(ind_type),key='df')
      X_full.loc[ind].to_hdf(version_folder+'X_{}_full.h5'.format(ind_type),key='df')
      X_full_igcc.loc[ind].to_hdf(version_folder+'X_{}_full_igcc.h5'.format(ind_type),key='df')
      
      
      y.loc[ind].to_hdf(version_folder+'y_{}.h5'.format(ind_type),key='df')
      pd.DataFrame(index=ind).to_hdf(version_folder+'y_pred_{}.h5'.format(ind_type),key='df')






import pandas as pd 
import numpy as np 
from sklearn.model_selection import GridSearchCV,  train_test_split
import os
import time
import shap
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

# Define quantile loss for quantile regression
def mqloss(y_true, y_pred, alpha):  
  if (alpha > 0) and (alpha < 1):
    residual = y_true - y_pred 
    return np.mean(residual * (alpha - (residual<0)))
  else:
    return np.nan

# Define model types by data and loss function
# (The code includes more models than actually presented in the paper)
model_type_infos = {
    'gtb_day_ahead_l2':{
        'loss':'l2',
        'data_suffix':'_day_ahead',
        'alpha':1
    },
    'gtb_day_ahead_q90':{
        'loss':'quantile',
        'data_suffix':'_day_ahead',
        'alpha':0.9
    },
    'gtb_day_ahead_igcc_l2':{
        'loss':'l2',
        'data_suffix':'_day_ahead_igcc',
        'alpha':1
    },
    'gtb_day_ahead_igcc_q90':{
        'loss':'quantile',
        'data_suffix':'_day_ahead_igcc',
        'alpha':0.9
    },
    'gtb_full_l2':{
        'loss':'l2',
        'data_suffix':'_full',
        'alpha':1,
    },
    'gtb_full_q90':{
        'loss':'quantile',
        'data_suffix':'_full',
        'alpha':0.9,
    },
    'gtb_full_igcc_l2':{
        'loss':'l2',
        'data_suffix':'_full_igcc',
        'alpha':1,
    },
    'gtb_full_igcc_q90':{
        'loss':'quantile',
        'data_suffix':'_full_igcc',
        'alpha':0.9,
    },
    'gtb_extended_l2':{
        'loss':'l2',
        'data_suffix':'_extended',
        'alpha':1
    },
    'gtb_extended_q90':{
        'loss':'quantile',
        'data_suffix':'_extended',
        'alpha':0.9
    },
    'gtb_extended_igcc_l2':{
        'loss':'l2',
        'data_suffix':'_extended_igcc',
        'alpha':1
    },
    'gtb_extended_igcc_q90':{
        'loss':'quantile',
        'data_suffix':'_extended_igcc',
        'alpha':0.9
    }
}

# Setup 
areas = ['DE']  
data_version = '2021-08-20'
targets = ['AFRR_DE_POS', 'AFRR_DE_NEG']

start_time = time.time()

for area in areas:
    
    print('\n---------------------------- ', area, ' ------------------------------------')
    
    data_folder = './data/{}/version_{}/'.format(area,data_version)

    for target in targets: 
        
        print('\n-------- ', target, ' --------')
        
        # Result folder where prediction, SHAP values and CV results are saved
        res_folder = './results/model_fit/{}/version_{}/target_{}/'.format(area,data_version, target)

        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        
        # Load target data
        y_train = pd.read_hdf(data_folder+'y_train.h5').loc[:, target]
        y_test = pd.read_hdf(data_folder+'y_test.h5').loc[:, target]
        y_cont = pd.read_hdf(data_folder+'y_cont.h5').loc[:, target]
        y_pred_train = pd.read_hdf(data_folder+'y_pred_train.h5') #contains only time index
        y_pred_test = pd.read_hdf(data_folder+'y_pred_test.h5') #contains only time index
        y_pred_cont = pd.read_hdf(data_folder+'y_pred_cont.h5') #contains only time index
        
        for model_type, model_info in model_type_infos.items():

            print('\nFitting model', model_type)
            
            # Load feature data
            X_train = pd.read_hdf(data_folder+'X_train{}.h5'.format(model_info['data_suffix']))
            X_test = pd.read_hdf(data_folder+'X_test{}.h5'.format(model_info['data_suffix'])) 
            X_cont = pd.read_hdf(data_folder+'X_cont{}.h5'.format(model_info['data_suffix'])) 
            
            # Daily profile prediction
            daily_profile = y_train.groupby(X_train.index.time).mean()
            y_pred_test['daily_profile'] = [daily_profile[time] for time in X_test.index.time]
            y_pred_cont['daily_profile'] = [daily_profile[time] for time in X_cont.index.time]
            
            # Mean predictor
            y_pred_test['mean_predictor'] = y_train.mean()
            y_pred_cont['mean_predictor'] = y_train.mean()
            
            ### Gradient boosting Regressor CV hyperparameter optimization ###

            # Parameters for hyper-parameter optimization
            params_grid = {
                'num_leaves': [10,30,200,900],
                'max_depth': [10,5],
                'subsample': [1,0.9,0.5],
                'learning_rate': [0.001, 0.05, 0.01, 0.1],
                'min_child_samples':[50,100, 500]
            }

            # Split training set into (smaller) training set and validation set
            X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train,
                                                                                      test_size=0.2)
            
            print('Using', X_train_val.shape[0]//(4*24), 'days of data for early stopping.')
            
            fit_params = {
                'eval_set':[(X_train_val, y_train_val)],
                'early_stopping_rounds':20, 
                'verbose':0
            }
            
            # Grid search for optimal hyper-parameters          
            grid_search = GridSearchCV(LGBMRegressor(n_estimators=1000,
                                                     objective= model_info['loss'],
                                                     min_child_weight=0, subsample_freq=1, n_jobs=6,
                                                     alpha=model_info['alpha']),
                                       params_grid, verbose=1, n_jobs=5, cv=5)

            grid_search.fit(X_train_train, y_train_train, **fit_params)
            
            # Save CV results
            pd.DataFrame(grid_search.cv_results_).to_csv(res_folder+'cv_results_{}.csv'.format(model_type))

            # Save best params (including n_estimators from early stopping on validation set)
            best_params = grid_search.best_estimator_.get_params()
            best_params['n_estimators'] = grid_search.best_estimator_.best_iteration_
            pd.DataFrame(best_params, index=[0]).to_csv(res_folder+'cv_best_params_{}.csv'.format(model_type))

            # Gradient boosting regression best model evaluation on test set
            best_params = pd.read_csv(res_folder+'cv_best_params_{}.csv'.format(model_type),
                                      usecols = list(params_grid.keys()) + ['n_estimators',
                                                                            'objective',
                                                                            'min_child_weight',
                                                                            'subsample_freq',
                                                                            'alpha',
                                                                            'n_jobs'])
            best_params = best_params.to_dict('records')[0]
            best_params['n_jobs'] = 30
            print('Best parameters from GridSearchCV:',
                  {p_name: best_params[p_name] for p_name in list(params_grid.keys()) + ['n_estimators']})
            
            
            # Train on whole training set (including validation set)
            model = LGBMRegressor(**best_params)
            model.fit(X_train, y_train)
                        
            # Calculate SHAP values on test set
            print('SHAP values analysis...')
            shap_vals = shap.Explainer(model).shap_values(X_test)
            np.save(res_folder + 'shap_values_test_{}.npy'.format(model_type), shap_vals)
            shap_vals = shap.Explainer(model).shap_values(X_cont)
            np.save(res_folder + 'shap_values_cont_{}.npy'.format(model_type), shap_vals)
            if model_info['loss']=='l2':
                shap_vals = shap.Explainer(model).shap_values(X_train.append(X_test).sort_index())
                np.save(res_folder + 'shap_values_train_and_test_{}.npy'.format(model_type), shap_vals)
            
            #shap_interact_vals = shap.Explainer(model).shap_interaction_values(X_test)
            #np.save(res_folder + 'shap_interaction_values_{}.npy'.format(model_type), shap_interact_vals)

            # Prediction on test set
            y_pred_train[model_type] = model.predict(X_train)
            y_pred_test[model_type] = model.predict(X_test)
            y_pred_cont[model_type] = model.predict(X_cont)
            
            
            # Print performances on test sets
            if model_info['loss']=='l2':
                print('R2 score train set: {}'.format(r2_score(y_train, y_pred_train[model_type])))  
                print('R2 score test set: {}'.format(r2_score(y_test, y_pred_test[model_type])))
                print('R2 score continuous set: {}'.format(r2_score(y_cont, y_pred_cont[model_type])))  
            if model_info['loss']=='quantile':
                print('MQ-loss train set: {}'.format(mqloss(y_train, y_pred_train[model_type],
                                                            alpha=best_params['alpha'])))   
                print('MQ-loss test set: {}'.format(mqloss(y_test, y_pred_test[model_type],
                                                           alpha=best_params['alpha'])))  
                print('MQ-loss continuous set: {}'.format(mqloss(y_cont, y_pred_cont[model_type],
                                                                 alpha=best_params['alpha'])))  
        # Save prediction
        y_pred_train.to_hdf(res_folder+'y_pred_train.h5',key='df')
        y_pred_test.to_hdf(res_folder+'y_pred_test.h5',key='df')
        y_pred_cont.to_hdf(res_folder+'y_pred_cont.h5',key='df')
        

print("Execution time [h]: {}".format((time.time() - start_time)/3600.))

# %%

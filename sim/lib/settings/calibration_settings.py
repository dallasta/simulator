import multiprocessing

'''
Default settings for model calibration
'''

settings_data = {
    'verbose' : True,
    'use_households' : True,
    'unscaled_testing_capacity' : 150, # number of tests processed every `testing_frequency` hours
}

settings_simulation = {
    'n_init_samples': 20,  # initial random evaluations
    'n_iterations': 200,  # iterations of BO
    'simulation_roll_outs': 40, # roll-outs done in parallel per parameter setting
    'cpu_count':  multiprocessing.cpu_count(), # cpus used for parallel computation
    'dynamic_tracing' : True,

}

# set initial seed count (approximately based on infection counts on March 10)
settings_initial_seeds = {
    'expo': 10,
    'ipre': 1,
    'isym': 3,
    'iasy': 3,
}

# parameter bounds
settings_param_bounds = {
    'betas': {
        'education': [0.3, 1.5],
        'social': [0.3, 2.0],
        'bus_stop': [0.3, 1.5],
        'office': [0.3, 1.5],
        'supermarket': [0.3, 1.5],
    },
    'beta_household': [0.3, 1.5],
    'mu': [0.0, 1.0]
}

# set testing parameters
settings_testing_params = {
    'testing_t_window': None,  # [set automatically in code]
    'testing_frequency': 24.0,  
    'test_reporting_lag': 48.0, 
    'tests_per_batch': None,  # [set automatically in code]
    'test_fpr': 0.0,
    'test_fnr': 0.0,
    'test_smart_delta': 24.0 * 3, 
    'test_smart_duration': 24.0 * 7, 
    'test_smart_action': 'isolate',
    'test_smart_num_contacts': 10,
    'test_targets': 'isym',
    'test_queue_policy': 'fifo',
    'smart_tracing': None,
}

# BO acquisition function optimization (Knowledge gradient)
# default settings from botorch
settings_acqf = {
    'acqf_opt_num_fantasies': 64,
    'acqf_opt_num_restarts': 10,
    'acqf_opt_raw_samples': 256,
    'acqf_opt_batch_limit': 5,
    'acqf_opt_maxiter': 20,
}

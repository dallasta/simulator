import time
import bisect
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import scipy.optimize
import scipy as sp
import os, math
import pickle
import matplotlib.pyplot as plt
#from joblib import Parallel, delayed

from lib.priorityqueue import PriorityQueue
from lib.mobilitysim_split import MobilitySimulator


TO_HOURS = 24.0


pp_legal_states = ['susc', 'expo', 'ipre', 'isym', 'iasy', 'resi', 'dead', 'hosp']


######################## functions and classes for launching the simulation ##############################

def run(mob_settings, intensity_params, distributions, t, ini_seeds):

 
    # run simulations
    summary = launch_simulation(
        mob_settings, 
        distributions, 
        intensity_params, 
        ini_seeds, 
        max_time=t
        )
    return summary


def launch_simulation(mob_settings, distributions, params, 
    initial_seeds, max_time):
    

    with open(mob_settings, 'rb') as fp:
        kwargs = pickle.load(fp)

    mob = MobilitySimulator(**kwargs)

    mob.simulate(max_time=max_time, seed=12345)
    num_people=mob.num_people, 
    num_sites=mob.num_sites, 
    site_loc=mob.site_loc, 
    home_loc=mob.home_loc
    
    sim = DiseaseModel(mob, distributions)

    
    sim.launch_epidemic(
        params=params,
        ini_seeds=initial_seeds,
        verbose=True)
    
    
    
    res = {
        'state' : sim.state,
        'state_started_at': sim.state_started_at,
        'state_ended_at': sim.state_ended_at,
        'people_age' : sim.mob.people_age,
        'children_count_iasy': sim.children_count_iasy,
        'children_count_ipre': sim.children_count_ipre,
        'children_count_isym': sim.children_count_isym,
        'initial_seeds': sim.initial_seeds
    }
    
    
    summary = Summary(max_time, num_people, num_sites, site_loc, home_loc)
    
    for code in pp_legal_states:
        summary.state[code][:] = res['state'][code]
        summary.state_started_at[code][:] = res['state_started_at'][code]
        summary.state_ended_at[code][:] = res['state_ended_at'][code]
        
 
    summary.people_age[:] = res['people_age']
        
    summary.children_count_iasy[:] = res['children_count_iasy']
    summary.children_count_ipre[:] = res['children_count_ipre']
    summary.children_count_isym[:] = res['children_count_isym']
    summary.seeds = res['initial_seeds']
    summary.mob = mob
    
    return summary



class Summary(object):
    """
    Summary class for a single evolution
    """

    def __init__(self, max_time,  n_people, n_sites, site_loc, home_loc):

        self.max_time = max_time
        self.n_people = n_people
        self.n_sites = n_sites
        self.site_loc = site_loc
        self.home_loc = home_loc
        
        self.state = {
            'susc': np.ones(n_people, dtype='bool'),
            'expo': np.zeros(n_people, dtype='bool'),
            'ipre': np.zeros(n_people, dtype='bool'),
            'isym': np.zeros(n_people, dtype='bool'),
            'iasy': np.zeros(n_people, dtype='bool'),
            'resi': np.zeros(n_people, dtype='bool'),
            'dead': np.zeros(n_people, dtype='bool'),
            'hosp': np.zeros(n_people, dtype='bool'),
        }

        self.state_started_at = {
            'susc': - np.inf * np.ones(n_people, dtype='float'),
            'expo': np.inf * np.ones(n_people, dtype='float'),
            'ipre': np.inf * np.ones(n_people, dtype='float'),
            'isym': np.inf * np.ones(n_people, dtype='float'),
            'iasy': np.inf * np.ones(n_people, dtype='float'),
            'resi': np.inf * np.ones(n_people, dtype='float'),
            'dead': np.inf * np.ones(n_people, dtype='float'),
            'hosp': np.inf * np.ones(n_people, dtype='float'),
        }
        self.state_ended_at = {
            'susc': np.inf * np.ones(n_people, dtype='float'),
            'expo': np.inf * np.ones(n_people, dtype='float'),
            'ipre': np.inf * np.ones(n_people, dtype='float'),
            'isym': np.inf * np.ones(n_people, dtype='float'),
            'iasy': np.inf * np.ones(n_people, dtype='float'),
            'resi': np.inf * np.ones(n_people, dtype='float'),
            'dead': np.inf * np.ones(n_people, dtype='float'),
            'hosp': np.inf * np.ones(n_people, dtype='float'),
        }
        
        self.mob = []
        
        self.people_age = np.zeros(n_people, dtype='int')

        self.children_count_iasy = np.zeros(n_people, dtype='int')
        self.children_count_ipre = np.zeros(n_people, dtype='int')
        self.children_count_isym = np.zeros(n_people, dtype='int')
        self.seeds = {}
  

        
        
def get_seeds(num,initial_counts):
    # init state variables with seeds
    total_seeds = sum(v for v in initial_counts.values())
    initial_people = np.random.choice(num, size=total_seeds, replace=False)

    ptr = 0
    initial_seeds = dict()
    for k, v in initial_counts.items():
        initial_seeds[k] = initial_people[ptr:ptr + v].tolist()
        ptr += v 
    return initial_seeds



######################## class for the epidemic model ##########################


class DiseaseModel(object):
    """
    Simulate continuous-time SEIR epidemics with exponentially distributed inter-event times.
    All units in the simulator are in hours for numerical stability, though disease parameters are
    assumed to be in units of days as usual in epidemiology
    """

    def __init__(self, mob, distributions):#, dynamic_tracing=False):
        """
        Init simulation object with parameters

        Arguments:
        ---------
        mob:
            object of class MobilitySimulator providing mobility data

        dynamic_tracing: bool
            If true contacts are computed on-the-fly during launch_epidemic
            instead of using the previously filled contact array

        """

        # cache settings
        self.mob = mob
        self.d = distributions
        
        # parse distributions object
        self.lambda_0 = self.d.lambda_0
        self.gamma = self.d.gamma
        self.fatality_rates_by_age = self.d.fatality_rates_by_age
        self.p_hospital_by_age = self.d.p_hospital_by_age
        self.delta = self.d.delta

        # parse mobility object
        self.n_people = mob.num_people
        self.n_sites = mob.num_sites
        self.max_time = mob.max_time
        
        # special state variables from mob object 
        self.people_age = mob.people_age
        self.num_age_groups = mob.num_age_groups
        self.site_type = mob.site_type
        self.site_dict = mob.site_dict
        self.num_site_types = mob.num_site_types
        
        self.people_household = mob.people_household
        self.households = mob.households
            
        assert(self.num_age_groups == self.fatality_rates_by_age.shape[0])
        assert(self.num_age_groups == self.p_hospital_by_age.shape[0])

        
        
        # print
        self.last_print = time.time()
        self._PRINT_INTERVAL = 0.1
        self._PRINT_MSG = (
            't: {t:.0f} '
            '| '
            '{maxt:.0f} hrs '
            '({maxd:.0f} d)'
            )

    def __print(self, t, force=False):
        if ((time.time() - self.last_print > self._PRINT_INTERVAL) or force) and self.verbose:
            print('\r', self._PRINT_MSG.format(t=t, maxt=self.max_time, maxd=self.max_time / 24),
                  sep='', end='', flush=True)
            self.last_print = time.time()
    

    def __init_run(self):
        """
        Initialize the run of the epidemic
        """

        self.queue = PriorityQueue()
        
        '''
        State and queue codes (transition event into this state)

        'susc': susceptible
        'expo': exposed
        'ipre': infectious pre-symptomatic
        'isym': infectious symptomatic
        'iasy': infectious asymptomatic
        'resi': resistant
        'dead': dead
        'hosp': hospitalized

     
        '''
        self.legal_states = ['susc', 'expo', 'ipre', 'isym', 'iasy', 'resi', 'dead', 'hosp']
        self.legal_preceeding_state = {
            'expo' : ['susc',],
            'ipre' : ['expo',],
            'isym' : ['ipre',],
            'iasy' : ['expo',],
            'resi' : ['isym', 'iasy'],
            'dead' : ['isym',],
            'hosp' : ['isym',],
        }

        self.state = {
            'susc': np.ones(self.n_people, dtype='bool'),
            'expo': np.zeros(self.n_people, dtype='bool'),
            'ipre': np.zeros(self.n_people, dtype='bool'),
            'isym': np.zeros(self.n_people, dtype='bool'),
            'iasy': np.zeros(self.n_people, dtype='bool'),
            'resi': np.zeros(self.n_people, dtype='bool'),
            'dead': np.zeros(self.n_people, dtype='bool'),
            'hosp': np.zeros(self.n_people, dtype='bool'),
        }

        self.state_started_at = {
            'susc': - np.inf * np.ones(self.n_people, dtype='float'),
            'expo': np.inf * np.ones(self.n_people, dtype='float'),
            'ipre': np.inf * np.ones(self.n_people, dtype='float'),
            'isym': np.inf * np.ones(self.n_people, dtype='float'),
            'iasy': np.inf * np.ones(self.n_people, dtype='float'),
            'resi': np.inf * np.ones(self.n_people, dtype='float'),
            'dead': np.inf * np.ones(self.n_people, dtype='float'),
            'hosp': np.inf * np.ones(self.n_people, dtype='float'),
        }
        self.state_ended_at = {
            'susc': np.inf * np.ones(self.n_people, dtype='float'),
            'expo': np.inf * np.ones(self.n_people, dtype='float'),
            'ipre': np.inf * np.ones(self.n_people, dtype='float'),
            'isym': np.inf * np.ones(self.n_people, dtype='float'),
            'iasy': np.inf * np.ones(self.n_people, dtype='float'),
            'resi': np.inf * np.ones(self.n_people, dtype='float'),
            'dead': np.inf * np.ones(self.n_people, dtype='float'),
            'hosp': np.inf * np.ones(self.n_people, dtype='float'),
        }   
       
        # infector of i
        self.parent = -1 * np.ones(self.n_people, dtype='int')

        # no. people i infected (given i was in a certain state)
        self.children_count_iasy = np.zeros(self.n_people, dtype='int')
        self.children_count_ipre = np.zeros(self.n_people, dtype='int')
        self.children_count_isym = np.zeros(self.n_people, dtype='int')
        
       
        
        self.initial_seeds = dict()
 
 
    

    def __process_exposure_event(self, t, i, parent):
        """
        Mark person `i` as exposed at time `t`
        Push asymptomatic or presymptomatic queue event
        """

        # track flags
        assert(self.state['susc'][i])
        self.state['susc'][i] = False
        self.state['expo'][i] = True
        self.state_ended_at['susc'][i] = t
        self.state_started_at['expo'][i] = t
        if parent is not None:
            self.parent[i] = parent
            if self.state['iasy'][parent]:
                self.children_count_iasy[parent] += 1
            elif self.state['ipre'][parent]:
                self.children_count_ipre[parent] += 1
            elif self.state['isym'][parent]:
                self.children_count_isym[parent] += 1
            else:
                assert False, 'only infectious parents can expose person i'


        # decide whether asymptomatic or (pre-)symptomatic
        if self.bernoulli_is_iasy[i]:
            self.queue.push(
                (t + self.delta_expo_to_iasy[i], 'iasy', i, None, None),
                priority=t + self.delta_expo_to_iasy[i])
        else:
            self.queue.push(
                (t + self.delta_expo_to_ipre[i], 'ipre', i, None, None),
                priority=t + self.delta_expo_to_ipre[i])

    def __process_presymptomatic_event(self, t, i):
        """
        Mark person `i` as presymptomatic at time `t`
        Push symptomatic queue event
        """

        # track flags
        assert(self.state['expo'][i])
        self.state['ipre'][i] = True
        self.state['expo'][i] = False
        self.state_ended_at['expo'][i] = t
        self.state_started_at['ipre'][i] = t

        # resistant event
        self.queue.push(
            (t + self.delta_ipre_to_isym[i], 'isym', i, None, None),
            priority=t + self.delta_ipre_to_isym[i])

        # contact exposure of others
        self.__push_contact_exposure_events(t, i, 1.0)
        
        # household exposures
        if self.households is not None and self.beta_household > 0:
            self.__push_household_exposure_events(t, i, 1.0)

    def __process_symptomatic_event(self, t, i, apply_for_test=True):
        """
        Mark person `i` as symptomatic at time `t`
        Push resistant queue event
        """

        # track flags
        assert(self.state['ipre'][i])
        self.state['isym'][i] = True
        self.state['ipre'][i] = False
        self.state_ended_at['ipre'][i] = t
        self.state_started_at['isym'][i] = t

      
        # hospitalized?
        if self.bernoulli_is_hospi[i]:
            self.queue.push(
                (t + self.delta_isym_to_hosp[i], 'hosp', i, None, None),
                priority=t + self.delta_isym_to_hosp[i])

        # resistant event vs fatality event
        if self.bernoulli_is_fatal[i]:
            self.queue.push(
                (t + self.delta_isym_to_dead[i], 'dead', i, None, None),
                priority=t + self.delta_isym_to_dead[i])
        else:
            self.queue.push(
                (t + self.delta_isym_to_resi[i], 'resi', i, None, None),
                priority=t + self.delta_isym_to_resi[i])

    def __process_asymptomatic_event(self, t, i, add_exposures=True):
        """
        Mark person `i` as asymptomatic at time `t`
        Push resistant queue event
        """

        # track flags
        assert(self.state['expo'][i])
        self.state['iasy'][i] = True
        self.state['expo'][i] = False
        self.state_ended_at['expo'][i] = t
        self.state_started_at['iasy'][i] = t

        # resistant event
        self.queue.push(
            (t + self.delta_iasy_to_resi[i], 'resi', i, None, None),
            priority=t + self.delta_iasy_to_resi[i])

        if add_exposures:
            # contact exposure of others
            self.__push_contact_exposure_events(t, i, self.mu)
            
            # household exposures
            if self.households is not None and self.beta_household > 0:
                self.__push_household_exposure_events(t, i, self.mu)

    def __process_resistant_event(self, t, i):
        """
        Mark person `i` as resistant at time `t`
        """

        # track flags
        assert(self.state['iasy'][i] != self.state['isym'][i]) # XOR
        self.state['resi'][i] = True
        self.state_started_at['resi'][i] = t
        
        # infection type
        if self.state['iasy'][i]:
            self.state['iasy'][i] = False
            self.state_ended_at['iasy'][i] = t

        elif self.state['isym'][i]:
            self.state['isym'][i] = False
            self.state_ended_at['isym'][i] = t
        else:
            assert False, 'Resistant only possible after asymptomatic or symptomatic.'

        # hospitalization ends
        if self.state['hosp'][i]:
            self.state['hosp'][i] = False
            self.state_ended_at['hosp'][i] = t

    def __process_fatal_event(self, t, i):
        """
        Mark person `i` as fatality at time `t`
        """

        # track flags
        assert(self.state['isym'][i])
        self.state['dead'][i] = True
        self.state_started_at['dead'][i] = t

        self.state['isym'][i] = False
        self.state_ended_at['isym'][i] = t

        # hospitalization ends
        if self.state['hosp'][i]:
            self.state['hosp'][i] = False
            self.state_ended_at['hosp'][i] = t
    
    def __process_hosp_event(self, t, i):
        """
        Mark person `i` as hospitalized at time `t`
        """

        # track flags
        assert(self.state['isym'][i])
        self.state['hosp'][i] = True
        self.state_started_at['hosp'][i] = t
    

    def __kernel_term(self, a, b, T):
        '''Computes
        \int_a^b exp(self.gamma * (u - T)) du
        =  exp(- self.gamma * T) (exp(self.gamma * b) - exp(self.gamma * a)) / self.gamma
        '''
        return (np.exp(self.gamma * (b - T)) - np.exp(self.gamma * (a - T))) / self.gamma


    def __push_contact_exposure_events(self, t, infector, base_rate):
        """
        Pushes all exposure events that person `i` causes
        for other people via contacts, using `base_rate` as basic infectivity
        of person `i` (equivalent to `\mu` in model definition)
        """

        def valid_j():
            '''Generates indices j where `infector` is present
            at least `self.delta` hours before j '''
            for j in range(self.n_people):
                if self.state['susc'][j]:
                    if self.mob.will_be_in_contact(indiv_i=j, indiv_j=infector, t=t, site=None):
                        yield j

        valid_contacts = valid_j()
       
        # generate potential exposure event for `j` from contact with `infector`
        for j in valid_contacts:
            self.__push_contact_exposure_infector_to_j(t=t, infector=infector, j=j, base_rate=base_rate)


    def __push_contact_exposure_infector_to_j(self, t, infector, j, base_rate):
        """
        Pushes the next exposure event that person `infector` causes for person `j`
        using `base_rate` as basic infectivity of person `i` 
        (equivalent to `\mu` in model definition)
        """
        tau = t
        sampled_event = False
        Z = self.__kernel_term(- self.delta, 0.0, 0.0)

        # sample next arrival from non-homogeneous point process
        while self.mob.will_be_in_contact(indiv_i=j, indiv_j=infector, t=tau, site=None) and not sampled_event:
            
            # check if j could get infected from infector at current `tau`
            # i.e. there is `delta`-contact from infector to j (i.e. non-zero intensity)
            has_infectious_contact, contact = self.mob.is_in_contact(indiv_i=j, indiv_j=infector, t=tau, site=None)

            # if yes: do nothing
            if has_infectious_contact:
                pass 

            # if no:       
            else: 
                # directly jump to next contact start of a `delta`-contact (memoryless property)
                next_contact = self.mob.next_contact(indiv_i=j, indiv_j=infector, t=tau, site=None)

                assert(next_contact is not None) # (while loop invariant)
                tau = next_contact.t_from

            # sample event with maximum possible rate (in hours)
            lambda_max = max(self.betas.values()) * base_rate * Z
            assert(lambda_max > 0.0) # this lamdba_max should never happen 
            tau += TO_HOURS * np.random.exponential(scale=1.0 / lambda_max)
        
            # thinning step: compute current lambda(tau) and do rejection sampling
            sampled_at_infectious_contact, sampled_at_contact = self.mob.is_in_contact(indiv_i=j, indiv_j=infector, t=tau, site=None)

            # 1) reject w.p. 1 if there is no more infectious contact at the new time (lambda(tau) = 0)
            if not sampled_at_infectious_contact:
                continue
            
            # 2) compute infectiousness integral in lambda(tau)
            # a. query times that infector was in [tau - delta, tau] at current site `site`
            site = sampled_at_contact.site
            infector_present = self.mob.list_intervals_in_window_individual_at_site(
                indiv=infector, site=site, t0=tau - self.delta, t1=tau)

            # b. compute contributions of infector being present in [tau - delta, tau]
            intersections = [(max(tau - self.delta, interv.left), min(tau, interv.right))
                for interv in infector_present]
            beta_k = self.betas[self.site_dict[self.site_type[site]]]
            p = (beta_k * base_rate * sum([self.__kernel_term(v[0], v[1], tau) for v in intersections])) \
                / lambda_max
            
            
            
            assert(p <= 1 + 1e-8 and p >= 0)

            # accept w.prob. lambda(t) / lambda_max
            u = np.random.uniform()
            if u <= p:
                self.queue.push(
                    (tau, 'expo', j, infector, site), priority=tau)
                sampled_event = True

    def __push_household_exposure_events(self, t, infector, base_rate):
        """
        Pushes all exposure events that person `i` causes
        in the household, using `base_rate` as basic infectivity
        of person `i` (equivalent to `\mu` in model definition)
        """

        def valid_j():
            '''Generates indices j where `infector` is present
            at least `self.delta` hours before j '''
            for j in self.households[self.people_household[infector]]:
                if self.state['susc'][j]:
                    yield j

        # generate potential exposure event for `j` from contact with `infector`
        for j in valid_j():
            self.__push_household_exposure_infector_to_j(t=t, infector=infector, j=j, base_rate=base_rate)

    def __push_household_exposure_infector_to_j(self, t, infector, j, base_rate):
        """
        Pushes the next exposure event that person `infector` causes for person `j`,
        who lives in the same household, using `base_rate` as basic infectivity of 
        person `i` (equivalent to `\mu` in model definition)
        """
        tau = t
        sampled_event = False

        # FIXME: we ignore the kernel for households infections since households members
        # will overlap for long period of times at home
        # Z = self.__kernel_term(- self.delta, 0.0, 0.0)

        lambda_household = self.beta_household * base_rate

        while tau < self.max_time and not sampled_event:
            tau += TO_HOURS * np.random.exponential(scale=1.0 / lambda_household)

            # site = -1 means it is a household infection
            # at the expo time, it will be thinned if needed
            self.queue.push(
                (tau, 'expo', j, infector, -1), priority=tau)

            sampled_event = True


 
    

    
    def launch_epidemic(self, params, ini_seeds, verbose=True):
        """
        Run the epidemic, starting from initial event list.
        Events are treated in order in a priority queue. An event in the queue is a tuple
        the form
            `(time, event_type, node, infector_node, location)`

        """
        self.verbose = verbose

        # optimized params
        self.betas = params['betas']
        self.mu = self.d.mu
        self.alpha = self.d.alpha

        # household param
        if 'beta_household' in params:
            self.beta_household = params['beta_household']
        else:
            self.beta_household = 0.0

       
        

        self.__init_run()
        self.was_initial_seed = np.zeros(self.n_people, dtype='bool')
        
        # init state variables with seeds
        self.initial_seeds = dict(ini_seeds)        
        
        
        ### sample all iid events ahead of time in batch
        batch_size = (self.n_people, )
        self.delta_expo_to_ipre = self.d.sample_expo_ipre(size=batch_size)
        self.delta_ipre_to_isym = self.d.sample_ipre_isym(size=batch_size)
        self.delta_isym_to_resi = self.d.sample_isym_resi(size=batch_size)
        self.delta_isym_to_dead = self.d.sample_isym_dead(size=batch_size)
        self.delta_expo_to_iasy = self.d.sample_expo_iasy(size=batch_size)
        self.delta_iasy_to_resi = self.d.sample_iasy_resi(size=batch_size)
        self.delta_isym_to_hosp = self.d.sample_isym_hosp(size=batch_size)

        self.bernoulli_is_iasy = np.random.binomial(1, self.alpha, size=batch_size)
        self.bernoulli_is_fatal = self.d.sample_is_fatal(self.people_age, size=batch_size)
        self.bernoulli_is_hospi = self.d.sample_is_hospitalized(self.people_age, size=batch_size)

        
 
        # initial seed
        self.initialize_states_for_seeds()
   
 
        # not initially seeded
        if self.lambda_0 > 0.0:
            delta_susc_to_expo = self.d.sample_susc_baseexpo(size=self.n_people)
            for i in range(self.n_people):
                if not self.was_initial_seed[i]:
                    # sample non-contact exposure events
                    self.queue.push(
                        (delta_susc_to_expo[i], 'expo', i, None, None),
                        priority=delta_susc_to_expo[i])

        
        # MAIN EVENT LOOP
        t = 0.0
       
        while self.queue:

        
            # get next event to process
            t, event, i, infector, k = self.queue.pop()

           
            # check termination
            if t > self.max_time:
                t = self.max_time
                self.__print(t, force=True)
                if self.verbose:
                    print(f'\n[Reached max time: {int(self.max_time)}h ({int(self.max_time // 24)}d)]')
                break
            if np.sum((1 - self.state['susc']) * (self.state['resi'] + self.state['dead'])) == self.n_people:
                if self.verbose:
                    print('\n[Simulation ended]')
                break

            # process event
            if event == 'expo':
                i_susceptible = ((not self.state['expo'][i])
                                    and (self.state['susc'][i]))

                # base rate exposure
                if (infector is None) and i_susceptible:
                    self.__process_exposure_event(t, i, None)

                # household exposure
                if (infector is not None) and i_susceptible and k == -1:

                    # 1) check whether infector recovered or dead
                    infector_recovered = \
                        (self.state['resi'][infector] or 
                            self.state['dead'][infector])

                    # 2) check whether infector got hospitalized
                    infector_hospitalized = self.state['hosp'][infector]

                    # 3) check whether infector or i are not at home
                    infector_away_from_home = False
                    i_away_from_home = False

                    infector_visits = self.mob.mob_traces[infector].find((t, t))
                    i_visits = self.mob.mob_traces[i].find((t, t))

                    for interv in infector_visits:
                        infector_away_from_home = (interv.t_to > t)
                        if infector_away_from_home:
                            break

                    for interv in i_visits:
                        i_away_from_home = i_away_from_home or (interv.t_to > t)

                    away_from_home = (infector_away_from_home or i_away_from_home)
           

                    # if none of 1), 2), 3) are true, the event is valid
                    if  (not infector_recovered) and (not infector_hospitalized) and (not away_from_home):

                        self.__process_exposure_event(t, i, infector)

                    # if 2) or 3) were true, a household infection could happen at a later point, hence sample a new event
                    if (infector_hospitalized or away_from_home):

                        mu_infector = self.mu if self.state['iasy'][infector] else 1.0
                        self.__push_household_exposure_infector_to_j(
                            t=t, infector=infector, j=i, base_rate=mu_infector) 

                # contact exposure
                if (infector is not None) and i_susceptible and k >= 0:

                    is_in_contact, contact = self.mob.is_in_contact(indiv_i=i, indiv_j=infector, site=k, t=t)
                    assert(is_in_contact and (k is not None))
                    i_visit_id, infector_visit_id = contact.id_tup

                    # 1) check whether infector recovered or dead
                    infector_recovered = \
                        (self.state['resi'][infector] or 
                            self.state['dead'][infector])

                    
                    # if 1 is not true, the event is valid
                    if  (not infector_recovered):
                    
                        self.__process_exposure_event(t, i, infector)

                   
            elif event == 'ipre':
                self.__process_presymptomatic_event(t, i)

            elif event == 'iasy':
                self.__process_asymptomatic_event(t, i)

            elif event == 'isym':
                self.__process_symptomatic_event(t, i)

            elif event == 'resi':
                self.__process_resistant_event(t, i)

            elif event == 'dead':
                self.__process_fatal_event(t, i)

            elif event == 'hosp':
                # cannot get hospitalization if not ill anymore 
                valid_hospitalization = \
                    ((not self.state['resi'][i]) and 
                        (not self.state['dead'][i]))

                if valid_hospitalization:
                    self.__process_hosp_event(t, i)
            else:
                # this should only happen for invalid exposure events
                assert(event == 'expo')

            # print
            self.__print(t, force=True)

        # free memory
        del self.queue
        
   


    def initialize_states_for_seeds(self):
        """
        Sets state variables according to invariants as given by `self.initial_seeds`

        NOTE: by the seeding heuristic using the reproductive rate
        we assume that exposures already took place
        """
        assert(isinstance(self.initial_seeds, dict))
        for state, seeds_ in self.initial_seeds.items():
            for i in seeds_:
                assert(self.was_initial_seed[i] == False)
                self.was_initial_seed[i] = True
                
                # inital exposed
                if state == 'expo':
                    self.__process_exposure_event(0.0, i, None)

                # initial presymptomatic
                elif state == 'ipre':
                    self.state['susc'][i] = False
                    self.state['expo'][i] = True

                    self.state_ended_at['susc'][i] = -1.0
                    self.state_started_at['expo'][i] = -1.0

                    self.bernoulli_is_iasy[i] = 0
                    self.__process_presymptomatic_event(0.0, i)


                # initial asymptomatic
                elif state == 'iasy':

                    self.state['susc'][i] = False
                    self.state['expo'][i] = True

                    self.state_ended_at['susc'][i] = -1.0
                    self.state_started_at['expo'][i] = -1.0

                    self.bernoulli_is_iasy[i] = 1
                    self.__process_asymptomatic_event(0.0, i, add_exposures=False)

                # initial symptomatic
                elif state == 'isym':

                    self.state['susc'][i] = False
                    self.state['ipre'][i] = True

                    self.state_ended_at['susc'][i] = -1.0
                    self.state_started_at['expo'][i] = -1.0
                    self.state_ended_at['expo'][i] = -1.0
                    self.state_started_at['ipre'][i] = -1.0

                    self.bernoulli_is_iasy[i] = 0
                    self.__process_symptomatic_event(0.0, i)

                # initial resistant
                elif state == 'resi':

                    self.state['susc'][i] = False
                    self.state['isym'][i] = True

                    self.state_ended_at['susc'][i] = -1.0
                    self.state_started_at['expo'][i] = -1.0
                    self.state_ended_at['expo'][i] = -1.0
                    self.state_started_at['ipre'][i] = -1.0
                    self.state_ended_at['ipre'][i] = -1.0
                    self.state_started_at['isym'][i] = -1.0

                    self.bernoulli_is_iasy[i] = 0
                    self.__process_resistant_event(0.0, i)

                else:
                    raise ValueError('Invalid initial seed state.')    
        
         
    def seeds_to_states(self):
        
        
        status= {
            'susc': np.ones(self.n_people, dtype='bool'),
            'expo': np.zeros(self.n_people, dtype='bool'),
            'ipre': np.zeros(self.n_people, dtype='bool'),
            'isym': np.zeros(self.n_people, dtype='bool'),
            'iasy': np.zeros(self.n_people, dtype='bool'),
            'resi': np.zeros(self.n_people, dtype='bool'),
            'dead': np.zeros(self.n_people, dtype='bool'),
            'hosp': np.zeros(self.n_people, dtype='bool'),
        }
        
        for state, seeds_ in self.initial_seeds.items():
            for i in seeds_:
                status['susc'][i]=False
                status[state][i]=True
                
        return status
    

def comp_state_over_time(sim, state):
    '''
    Computes `state` variable over time [0, sim.max_time] 
    '''
    t_unit=24
    ts, val = [], []
    for t in np.linspace(0.0, sim.max_time, num=sim.max_time+1, endpoint=True):
        s = sum([np.sum(is_state_at(sim, status, t)) for status in state])
        ts.append(t/t_unit)
        val.append(s)

    return np.array(ts), np.array(val)
    
    
def is_state_at(sim, state, t):
        
    return (sim.state_started_at[state] <= t) & (sim.state_ended_at[state] > t)

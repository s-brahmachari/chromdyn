import numpy as np

class Extrusion_kinetics():
    def __init__(self, lef):
        """
        Initialize the Gillespie extrusion object.

        Args:
        - lef: Loop extruder object containing details of all the extruders

        Returns:
        - None
        """
        self.lef = lef  # Loop extruder object
        seed = np.random.randint(100000)
        self.rng = np.random.default_rng(seed)  # Random number generator
        self.time = 0  # Initialize time
        self.dt = 0 
        self.current_event = {} # initialize empty event dict
        print("Initialized extrusion kinetics with the random seed ", seed)

    def simulate(self, verbose=False,):
        """
        Perform a single simulation step.

        Returns:
        - None
        """
        # Gillespie part
        self.create_rate_vec()
        time = self.get_time_to_next_event()
        event = self.get_next_event()
        self.current_event = event
        self.dt = time
        self.time += time
        # print('georgia',time)
        self.lef.save_anchors(self.time)

        # Loop extruder part
        self.lef.update_LEs(event, time)
        
        if verbose==1:
            self.print_event(event)
            print(f"Rate vector: {self.rate_vec}")
            
            #consistency check
            self.lef.print_latest_event()
        
        elif verbose==2:
            self.print_event(event)
            
    def create_rate_vec_new(self):
        """
        Create the rate vector with each LEF leg translocation as independent events.

        Returns:
        - None
        """
        # Generate rate vector for all LEs
        self.rate_vec = np.array([
            [
                self.lef.get_off_rate(key),
                # self.lef.loop_extruder_dict[key]['k_step_left'],
                # self.lef.loop_extruder_dict[key]['k_step_right'],
                (not self.lef.loop_extruder_dict[key]['is_left_anchor_paused']) * self.lef.loop_extruder_dict[key]['k_step_left'],
                (not self.lef.loop_extruder_dict[key]['is_right_anchor_paused']) * self.lef.loop_extruder_dict[key]['k_step_right'],
                self.lef.loop_extruder_dict[key]['is_left_anchor_paused'] * self.lef.loop_extruder_dict[key]['is_right_anchor_paused'] * self.lef.loop_extruder_dict[key]["hop_left_factor"] * self.lef.loop_extruder_dict[key]['k_hop_left'],
                self.lef.loop_extruder_dict[key]['is_right_anchor_paused'] * self.lef.loop_extruder_dict[key]['is_left_anchor_paused'] * self.lef.loop_extruder_dict[key]["hop_right_factor"] * self.lef.loop_extruder_dict[key]['k_hop_right']
            ]
            for key in self.lef.loop_extruder_dict.keys()
        ]).flatten()

        # Generate event matrix for all LEs
        self.event_matrix = np.array([
            [
                {'LE_id': key, 'move_left_anchor': False, 'move_right_anchor': False, 'unbind': True, 'hop_left': False, 'hop_right': False},
                {'LE_id': key, 'move_left_anchor': True, 'move_right_anchor': False, 'unbind': False, 'hop_left': False, 'hop_right': False},
                {'LE_id': key, 'move_left_anchor': False, 'move_right_anchor': True, 'unbind': False, 'hop_left': False, 'hop_right': False},
                {'LE_id': key, 'move_left_anchor': False, 'move_right_anchor': False, 'unbind': False, 'hop_left': True, 'hop_right': False},
                {'LE_id': key, 'move_left_anchor': False, 'move_right_anchor': False, 'unbind': False, 'hop_left': False, 'hop_right': True}
            ]
            for key in self.lef.loop_extruder_dict.keys()
        ]).flatten()
        
    def create_rate_vec(self):
        """
        Create the rate vector with each LEF leg translocation as independent events.

        Returns:
        - None
        """
        # Generate rate vector for all LEs
        self.rate_vec = np.array([
            [
                self.lef.loop_extruder_dict[key]["off_rate_factor"] * self.lef.loop_extruder_dict[key]['k_off'],
                # self.lef.loop_extruder_dict[key]['k_step_left'],
                # self.lef.loop_extruder_dict[key]['k_step_right'],
                (not self.lef.loop_extruder_dict[key]['is_left_anchor_paused']) * self.lef.loop_extruder_dict[key]['k_step_left'],
                (not self.lef.loop_extruder_dict[key]['is_right_anchor_paused']) * self.lef.loop_extruder_dict[key]['k_step_right'],
                self.lef.loop_extruder_dict[key]['is_left_anchor_paused'] * self.lef.loop_extruder_dict[key]['is_right_anchor_paused'] * self.lef.loop_extruder_dict[key]["hop_left_factor"] * self.lef.loop_extruder_dict[key]['k_hop_left'],
                self.lef.loop_extruder_dict[key]['is_right_anchor_paused'] * self.lef.loop_extruder_dict[key]['is_left_anchor_paused'] * self.lef.loop_extruder_dict[key]["hop_right_factor"] * self.lef.loop_extruder_dict[key]['k_hop_right']
            ]
            for key in self.lef.loop_extruder_dict.keys()
        ]).flatten()

        # Generate event matrix for all LEs
        self.event_matrix = np.array([
            [
                {'LE_id': key, 'move_left_anchor': False, 'move_right_anchor': False, 'unbind': True, 'hop_left': False, 'hop_right': False},
                {'LE_id': key, 'move_left_anchor': True, 'move_right_anchor': False, 'unbind': False, 'hop_left': False, 'hop_right': False},
                {'LE_id': key, 'move_left_anchor': False, 'move_right_anchor': True, 'unbind': False, 'hop_left': False, 'hop_right': False},
                {'LE_id': key, 'move_left_anchor': False, 'move_right_anchor': False, 'unbind': False, 'hop_left': True, 'hop_right': False},
                {'LE_id': key, 'move_left_anchor': False, 'move_right_anchor': False, 'unbind': False, 'hop_left': False, 'hop_right': True}
            ]
            for key in self.lef.loop_extruder_dict.keys()
        ]).flatten()

    def get_time_to_next_event(self):
        """
        Return the time to simulate before the next event occurs.

        Returns:
        - time_to_next_event: Time to next event
        """
        mean_time = 1. / self.rate_vec.sum()
        time_to_next_event = self.rng.exponential(scale=mean_time)
        return time_to_next_event

    def get_next_event(self):
        """
        Return the id and anchor of the LEF to translocate.

        Returns:
        - event_vec: Next event
        """
        event_vec = self.rng.choice(self.event_matrix, size=1, p=self.rate_vec / self.rate_vec.sum())[0]
        return event_vec
    
    def print_event(self, event):
        """
        Extract the event from evenet vector and print 

        Args:
        - event: the selected event dict

        Returns:
        - None
        """
        leid = event['LE_id']
        true_events = [key for key, value in event.items() if value is True and key != 'LE_id']
        dx_left = self.lef.loop_extruder_dict[leid]['left_anchor'] - self.lef.loop_extruder_dict_old[leid]['left_anchor']
        dx_right = self.lef.loop_extruder_dict[leid]['right_anchor'] - self.lef.loop_extruder_dict_old[leid]['right_anchor']
        print(f"Time: {self.time:<10.6f} | {leid:^10s} {true_events[0]:<20s} | dx_left: {dx_left:<6.0f} | dx_right: {dx_right:<6.0f}")


import copy
import h5py
import os
from scipy.spatial import distance

class Loop_Extruders():
    def __init__(self, 
                 num_LE, 
                 chain_topology,
                 name='LE',
                 k_off=2e-3, 
                 stepsize_LE=1.0, 
                 kstep_LE=0.1,
                 k_hop=1e-2,
                 max_hop_len=5,
                 init_config = [],
                 diffusion_strengths = [],
                 temp=0.0,
                 hop3D = False,
                 sim3D = None,
                 dist_cutoff=1.5,
                 preferential_loading=0.0,
                 loading_sites=[],
                 t_mesh = 0.001,
                 tau = 1e6,
                 v_LE = 5.0,
                 k_LE = 30.0,
                 f_stall=50.0,
                 ):
        
        """
        Initialize the loop extruder class
        """
        self.name = name
        seed = np.random.randint(100000000)
        self.rng = np.random.default_rng(seed)  # Random number generator
        print("Random seed", seed)
        self.temp = temp
        self.hop3D = hop3D
        assert type(self.hop3D)==bool, "hop3D is a boolean parameter!"
        if self.hop3D: 
            assert sim3D != None, 'For 3D hopping polymer (OpenMiChroM object) needs to be specified!'
        self.sim3D = sim3D
        self.k_LE = k_LE
        self.f_stall = f_stall
        self.dist_cutoff = dist_cutoff
        
        if len(diffusion_strengths)==0:
            diffusion_strengths = [1.0 for _ in range(num_LE)] 
        
        assert len(diffusion_strengths)==num_LE, 'Diffusion strengths should be a list of dimension equal to num_LE'
        
        #maximum no of bound LEs
        self.num_LE = num_LE
        self.max_hop_len = max_hop_len

        # set the chain for extrusion
        # chains = [(initial mono, final mono, is Ring)]
        self.topology = chain_topology
        
        #LEs stall at the ends or fall off
        self.stall_at_ends = False
        self.unbind_at_end =  False
        
        #variable to be set to 1 when things go wrong
        self.implement_CTCF_orientation = False
        self.ctcf_hop_factor = 1.0
        self.ctcf_off_rate_factor = 1.0
        
        #LE kinetic parameters
        self.k_off = k_off
        self.k_hop = k_hop
        self.drag = 1.0
        self.tau = tau
        
        self.kstep_LE = kstep_LE
        self.stepsize_LE = stepsize_LE
        #velocity
        self.v_LE = v_LE #self.stepsize_LE * self.kstep_LE
        self.t_mesh = t_mesh
        
        #preferential loading sites
        if preferential_loading==True:
            preferential_loading=1.0
        elif preferential_loading==False:
            preferential_loading=0.0
            
        assert 1.0>=float(preferential_loading)>=0.0, 'Preferential loading parameter not a float between 0 and 1. This parameter encodes the degree of preferntial loading at given loading sites'
        
        if preferential_loading>0.0: 
            assert len(loading_sites)>0 , 'Preferential loading is ON but no loading sites provided. Use loading_sites list.'
        
        self.preferential_LE_loading = preferential_loading
        self.preferential_loading_sites = loading_sites

        self.current_LE = -1
        self.LE_steps = []
        

        self.loop_extuder_3Dneighbors = {f'LE_{le_i}':[] for le_i in range(self.num_LE)}

        self.loop_extruder_dict = {f'LE_{le_i}':{'left_anchor': 0, 
                                                 'right_anchor': 1,
                                                 'bond_index': -1, 
                                                 'bound_time': 0.0, 
                                                 'is_left_anchor_paused': False, 
                                                 'is_right_anchor_paused': False, 
                                                 'left_anchor_paused_time': 0.0, 
                                                 'right_anchor_paused_time': 0.0,
                                                 'both_anchors_paused_time': 0.0,
                                                 'k_step_left': self.kstep_LE, 
                                                 'k_step_right': self.kstep_LE,
                                                 'v_left_anchor': self.v_LE, 
                                                 'v_right_anchor': self.v_LE,
                                                 'hop_left_factor': 1.0, 
                                                 'hop_right_factor': 1.0,
                                                 'off_rate_factor': 1.0,
                                                 'k_off': self.k_off, 
                                                 'k_hop_left': self.k_hop, 
                                                 'k_hop_right': self.k_hop, 
                                                 'left_anchor_paused_at': -1, 
                                                 'right_anchor_paused_at': -1, 
                                                 'left_anchor_paused_by': ['None', 'None'], 
                                                 'right_anchor_paused_by': ['None', 'None'],
                                                 'active_force_left': self.v_LE*self.drag,
                                                 'active_force_right': self.v_LE*self.drag,
                                                 'corr_time': self.tau,
                                                 'resisting_force': 0.0,
                                                 'strain':[0.0],
                                                 'diffusion_strength': diffusion_strengths[le_i]} 
                                   for le_i in range(self.num_LE)}

        self.mobile_blockers = [0]*self.num_LE + [1]*self.num_LE 
        self.immobile_blockers = {}
        all_chains = list(self.topology.chains())
        for chain in all_chains:
            self.immobile_blockers[int(list(chain.atoms())[0].id)]= {'type':'End','orientation':'left'}
            self.immobile_blockers[int(list(chain.atoms())[-1].id)]= {'type':'End','orientation':'right'}

        self.chain_start = int(list(all_chains[0].atoms())[0].id)
        self.chain_end = int(list(all_chains[-1].atoms())[-1].id)

        if len(init_config)==0:
            #loop over all LEs and bind them to free sites
            print("Randomly distributing the LEFs")
            
            for LE_id in self.loop_extruder_dict.keys():
                left_anchor, right_anchor = self.find_random_binding_site()
                self.set_LE_left_anchor(LE_id, left_anchor, -1)
                self.set_LE_right_anchor(LE_id, right_anchor, -1)
        else:
            assert len(init_config)==num_LE, "conflict between init_config and num_LE!"
            for jj, LE_id in enumerate(self.loop_extruder_dict.keys()):
                left_anchor, right_anchor = init_config[jj]
                self.set_LE_left_anchor(LE_id, left_anchor, -1)
                self.set_LE_right_anchor(LE_id, right_anchor, -1)
            # self.set_LE_position(LE_id=id , new_anchors=anchors)
        #initialize immobile blockers
        
        
        # back up loop extruder dict to store the previous state
        self.loop_extruder_dict_old = copy.deepcopy(self.loop_extruder_dict)
        self.save_anchors(0.0)
        
    def find_random_binding_site(self,):
        """
        Find a random binding site for a LE
        """
        keep_searching = True
        maxiter=0
        while keep_searching:
            maxiter += 1
            anchor_i = self.rng.choice(range(self.chain_start+1, self.chain_end-1))
            if self.free_for_binding(anchor_i) and self.free_for_binding(anchor_i+1): 
                keep_searching = False
                new_anchors = (anchor_i, anchor_i+1)
            if maxiter>100:
                keep_searching = False
                new_anchors = (-1,-1)
        return new_anchors

    def get_off_rate(self, leid):
        return self.loop_extruder_dict[leid]["off_rate_factor"] * self.loop_extruder_dict[leid]['k_off']
        
    def find_preferential_loading_site(self,):
        keep_searching = True
        search_distance = 5
        maxiter=0
        while keep_searching:
            maxiter += 1
            anchor_i = self.rng.choice(self.preferential_loading_sites, size=1)[0] + np.sign(self.rng.normal(loc=0.0, scale=1.0)) * self.rng.choice(range(1, search_distance))
            if self.free_for_binding(anchor_i) and self.free_for_binding(anchor_i+1): 
                keep_searching = False  
                new_anchors = (anchor_i, anchor_i+1)      
            search_distance+=1
            if maxiter>100:
                keep_searching = False
                new_anchors = (-1,-1)
        return new_anchors

    def get_anchors(self, leid):
        return (int(self.loop_extruder_dict[leid]['left_anchor']), int(self.loop_extruder_dict[leid]['right_anchor']))
    
    def is_site_within_chain(self, mono):
        """
        Checks if the monomer is within the chain. Returns True when inside the chain
        """
        return (self.chain_end>=int(mono)>=self.chain_start)
    
    def is_site_unoccupied(self, mono):
        """
        Checks if the monomer is blocked by either a mobile or immobile element
        Returns True when site is unoccupied.
        """
        return ((int(mono) not in self.mobile_blockers) and (int(mono) not in self.immobile_blockers.keys()))

    def free_for_binding(self, i):
        """
        check if a binding site is free to be bound by LEs
        Returns:
        - bool 
        """
        return self.is_site_within_chain(i) and self.is_site_unoccupied(i)
    
    def set_LE_left_anchor(self, LE_id, new_left_anchor, roadblock):
        """
        Set the left anchor location
        this function does not necessarily check if the anchors are legal
        make sure the anchor location is legal before calling this function
        """
        old_left_anchor = self.loop_extruder_dict[LE_id]['left_anchor']
        self.loop_extruder_dict[LE_id]['left_anchor'] = new_left_anchor
        
        
        # check if occupied monos list needs updating
        if int(old_left_anchor)!=int(new_left_anchor):
            self.update_mobile_blockers(new_anchor=int(new_left_anchor), old_anchor=int(old_left_anchor))

    def set_LE_right_anchor(self, LE_id, new_right_anchor, roadblock):
        """
        Set the right anchor location
        this function does not necessarily check if the anchors are legal
        make sure the anchor location is legal before calling this function
        """
        old_right_anchor = self.loop_extruder_dict[LE_id]['right_anchor']        
        self.loop_extruder_dict[LE_id]['right_anchor'] = new_right_anchor
        

        #check if the force field needs updating
        if int(old_right_anchor)!=int(new_right_anchor):
            self.update_mobile_blockers(new_anchor=int(new_right_anchor), old_anchor=int(old_right_anchor))
    
    def update_mobile_blockers(self, new_anchor, old_anchor):
        """
        Updtaes the occupied monomer list. This contains all mobile blockers
        """
        self.mobile_blockers.remove(old_anchor)
        self.mobile_blockers.append(new_anchor)

    def update_LE_3Dneighbors(self, cutoff):
        xyz = self.sim3D.getPositions()
        distance_matrix = np.triu(distance.cdist(xyz, xyz, 'euclidean'))
        for leid in self.loop_extruder_dict.keys():
            anchors = self.get_anchors(leid)
            # print(anchors)
            indices = list(set(np.where(((distance_matrix[anchors[0]] < cutoff) & (distance_matrix[anchors[0]] > 0.0)) | 
                               ((distance_matrix[anchors[1]] < cutoff) & (distance_matrix[anchors[1]] > 0.0)))[0]))
            
            # #exclude immediate neighbors
            # for xx,val in enumerate(indices):
            #     if abs(val-anchors[0]) < 2.0 or abs(val-anchors[1]) < 2.0:
            #         indices.pop(xx)
                    
            self.loop_extuder_3Dneighbors[leid] = indices
    
    # def update_LE_resisting_forces(self,):
    #     coords = self.sim3D.state.getPositions(asNumpy=True)
    #     for leid in self.loop_extruder_dict.keys():
    #         left_anchor_pos = coords[int(self.loop_extruder_dict[leid]['left_anchor']),:]
    #         right_anchor_pos = coords[int(self.loop_extruder_dict[leid]['right_anchor']),:]
    #         strain=(np.linalg.norm(left_anchor_pos-right_anchor_pos)-1)
    #         force = self.k_LE * strain
    #         self.loop_extruder_dict[leid]['resisting_force']=force
    #         self.loop_extruder_dict[leid]['strain'].append(strain)
    #         # print('tunisia', leid, force, strain, int(self.loop_extruder_dict[leid]['left_anchor']), int(self.loop_extruder_dict[leid]['right_anchor']))
    
    def reset_LE_resisting_forces(self,):
        for leid in self.loop_extruder_dict.keys():
            # self.loop_extruder_dict[leid]['resisting_force']=force
            self.loop_extruder_dict[leid]['strain']=[0.0]
        
    def update_LEs(self, event, dt):
        """
        Update the loop extruder dict (old and current) with the event

        Args:
        - event: event dict
        - dt: time to next event

        Returns:
        - None
        """
        #obtain the LE in event and set it to the current LE (to be used by get_latest_event)
        LE_id = event["LE_id"]
        self.current_LE = LE_id

        #backup
        self.loop_extruder_dict_old = copy.deepcopy(self.loop_extruder_dict)

        #check for the legality of the event dict -- there must be only one event
        assert sum([event[key] for key in event.keys() if key!="LE_id"])==1 , " More than one event is True in the event dict!"

        if event['unbind']:
            
            self.reset_LE(LE_id) #reset bound time
            
            #decide if loading at preferntial sites
            if self.rng.uniform(low=0.0, high=1.0) < self.preferential_LE_loading:
                left_anchor, right_anchor = self.find_preferential_loading_site()
            else:
                left_anchor, right_anchor = self.find_random_binding_site()
            
            assert left_anchor!=-1, 'Could not locate new binding site.'
            
            self.set_LE_left_anchor(LE_id, left_anchor, -1)
            self.set_LE_right_anchor(LE_id, right_anchor, -1)
            
            self.update_all_LE_times(dt, exclude=LE_id)
            
        elif event['move_left_anchor']:
            x_t = self.loop_extruder_dict[LE_id]["left_anchor"]
            # Delta_x = self.loop_extruder_dict[LE_id]["v_left_anchor"]*dt
            # noise = self.rng.normal(loc=0.0, scale=self.loop_extruder_dict[LE_id]["diffusion_strength"]*self.temp, size=1)[0]
            # x_t_plus_dt = x_t - Delta_x + noise
            Delta_x, f_t = self._traverse_LE(dt, LE_id, self.loop_extruder_dict[LE_id]['active_force_left'])
            x_t_plus_dt = x_t - Delta_x
            new_left_anchor, roadblock = self.check_for_roadblocks(x_t, x_t_plus_dt)
            self.loop_extruder_dict[LE_id]['active_force_left'] = f_t
            # is left anchor moving right of the right anchor
            # if new_left_anchor>self.loop_extruder_dict[LE_id]["right_anchor"]
            
            self.set_LE_left_anchor(LE_id, new_left_anchor, roadblock)
            
            self.update_all_LE_times(dt)
            self.LE_steps.append(Delta_x)
            
        elif event['move_right_anchor']:
            x_t = self.loop_extruder_dict[LE_id]["right_anchor"]
            # Delta_x = self.loop_extruder_dict[LE_id]["v_right_anchor"]*dt
            # noise = self.rng.normal(loc=0.0, scale=self.loop_extruder_dict[LE_id]["diffusion_strength"]*self.temp, size=1)[0]
            # x_t_plus_dt = x_t + Delta_x + noise
            Delta_x, f_t = self._traverse_LE(dt, LE_id, self.loop_extruder_dict[LE_id]['active_force_right'])
            x_t_plus_dt = x_t + Delta_x
            new_right_anchor, roadblock = self.check_for_roadblocks(x_t,x_t_plus_dt)

            self.set_LE_right_anchor(LE_id, new_right_anchor, roadblock)
            self.loop_extruder_dict[LE_id]['active_force_right'] = f_t
            
            self.update_all_LE_times(dt)
            self.LE_steps.append(Delta_x)
            
        elif event['hop_left']:
            x_t = self.loop_extruder_dict[LE_id]["left_anchor"]
            hopped=0
            if self.hop3D:
                self.update_LE_3Dneighbors(cutoff=self.dist_cutoff)
                possible_anchors = [xx for xx in self.loop_extuder_3Dneighbors[LE_id]]
                n=2
                hop_loc_1D = -1
                while hopped==0 and n<self.max_hop_len:            
                    possible_new_anchor = x_t - n
                    if self.free_for_binding(possible_new_anchor):
                        possible_anchors.append(possible_new_anchor)
                        hop_loc_1D = possible_new_anchor
                        hopped=1
                    n+=1
                self.rng.shuffle(possible_anchors)
                # print(possible_anchors)
                for new_left_anchor in possible_anchors:
                    if self.free_for_binding(new_left_anchor):
                        self.set_LE_left_anchor(LE_id, new_left_anchor, -1)
                        hopped=1
                        break
                # if hopped in 3D, randomize the velocity direction
                # if abs(hop_loc_1D - new_left_anchor)>1e-3 and hopped==1:
                #     self.loop_extruder_dict[LE_id]["v_left_anchor"] = self.v_LE * np.sign(self.rng.normal(loc=0, scale=1.0, size=1)[0])
                
            else:
                n=2
                while hopped==0 and n<self.max_hop_len:
                    new_left_anchor = x_t - n
                    if self.free_for_binding(new_left_anchor):
                        self.set_LE_left_anchor(LE_id, new_left_anchor, -1)
                        hopped=1
                    n+=1
            if hopped==0:
                # could not hop - stuck!
                self.loop_extruder_dict[LE_id]['left_anchor_paused_time'] += dt 
                
            self.loop_extruder_dict[LE_id]['active_force_left'] = 0.0
            
            self.update_all_LE_times(dt)
            
        elif event['hop_right']:
            x_t = self.loop_extruder_dict[LE_id]["right_anchor"]
            hopped=0
            
            if self.hop3D:
                self.update_LE_3Dneighbors(cutoff=self.dist_cutoff)
                possible_anchors = [xx for xx in self.loop_extuder_3Dneighbors[LE_id]]
                n=2
                hop_loc_1D = -1
                while hopped==0 and n<self.max_hop_len:            
                    possible_new_anchor = x_t + n
                    if self.free_for_binding(possible_new_anchor):
                        possible_anchors.append(possible_new_anchor)
                        hop_loc_1D = possible_new_anchor
                        hopped=1
                    n+=1
                self.rng.shuffle(possible_anchors)
                # print(possible_anchors)
                for new_right_anchor in possible_anchors:
                    if self.free_for_binding(new_right_anchor):
                        self.set_LE_right_anchor(LE_id, new_right_anchor, -1)
                        hopped=1
                        break
                # if hopped in 3D, randomize the velocity direction
                # if abs(hop_loc_1D - new_right_anchor)>1e-3 and hopped==1:
                #     self.loop_extruder_dict[LE_id]["v_right_anchor"] = self.v_LE * np.sign(self.rng.normal(loc=0, scale=1.0, size=1)[0])
                # print(new_right_anchor)
            else:
                    
                n=2
                while hopped==0 and n<self.max_hop_len:            
                    new_right_anchor = x_t + n
                    if self.free_for_binding(new_right_anchor):
                        self.set_LE_right_anchor(LE_id, new_right_anchor, -1)
                        hopped=1
                    n+=1
            if hopped==0:
                # could not hop - stuck!
                self.loop_extruder_dict[LE_id]['right_anchor_paused_time'] += dt
            
            self.loop_extruder_dict[LE_id]['active_force_right'] = 0.0
            self.update_all_LE_times(dt)
            
        self.update_blockers_all_LEs()
        
    
    def check_for_roadblocks(self, xi, xf):
        """
        Checks if extrusion can happen in a segment and returns the final position. 
        If there are roadblocks along the way, final position is modified accordingly.
        """
        x_out = xi
        extrude_len = 0.0
        roadblock = -1

        #check consistency
        assert self.is_site_within_chain(xi), 'Extruder already outside the polymer!'
        
        #attempting to move within the bead
        if int(xi)==int(xf):
            x_out = xf
            extrude_len = xf-xi
        
        #attempting to move outside the bead
        else:
            dx = np.sign(xf-xi) #this takes care of stepping up or down the chain
            extrude_len = 0.0
            # check step by step if there are any roadblocks
            for xx in np.arange(xi, xf, dx):
                if self.free_for_binding(xx+dx): 
                    extrude_len += dx
                else:
                    # roadblock identified -- no need to check more 
                    roadblock = int(xx+dx)
                    
                    # if roadblock==self.chain_end or roadblock==self.chain_start:
                    # print("Venice! Stuck", self.current_LE)
                    break
                    # elif self.rng.random()>1.0: 
                    #     #95% chance of getting stuck at the blocker
                    #     print("Rome! bypassing barrier")
                    #     roadblock = -1
                    #     continue
                    
                        
        
        #check legality
        assert self.is_site_within_chain(roadblock) or roadblock==-1, "Roadblock outside the polymer!!" 

        #if no roadblocks extrude to xf
        if roadblock==-1:
            x_out = xf
        # otherwise extrude according to extrude_len
        else:
            x_out = xi + extrude_len
        # print(f"Roadblock check: in-{xi} out-{x_out} block-{roadblock}")
        return (x_out, roadblock)
    
    def update_all_LE_times(self, dt, exclude=None):
        """
        Update the bound time of all LEs
        """
        for le_id in self.loop_extruder_dict.keys():
            if le_id != exclude:
                self.loop_extruder_dict[le_id]['bound_time'] += dt
        
        # LEid_list = list(self.loop_extruder_dict.keys())
        
        # f0 = [self.loop_extruder_dict[xx]['active_force'] for xx in LEid_list]
        # tau = [self.loop_extruder_dict[xx]['corr_time'] for xx in LEid_list]
        # t,f_t = self._integrate_force(f0,dt,tau)
        
        # for leid in LEid_list:
        #     self.loop_extruder_dict[leid]['active_force']=f_t[:-1]

    # def _integrate_force(self, f0, dt, tau):
    #     t_mesh=self.t_mesh
    #     times = np.arange(t_mesh, dt, t_mesh)
    #     f=np.zeros((len(f0),times.shape[0]+1))
    #     f[:,0]=f0
    #     for ii, _ in enumerate(times):
    #         f[:, ii+1] = f[:, ii] * np.exp(-t_mesh/tau) + np.sqrt(1-np.exp(-2 * t_mesh / tau)) * self.rng.normal(loc=0, scale=1.0, size=len(f0))
    #     return times, f
    
    def update_LE_resisting_force(self,leids, anchor_locs):
        # print(leids, anchor_locs)
        coords = self.sim3D.state.getPositions(asNumpy=True)
        for xx,leid in enumerate(leids):
            left_anchor_pos = coords[int(anchor_locs[xx][0]),:]
            right_anchor_pos = coords[int(anchor_locs[xx][1]),:]
            strain=(np.linalg.norm(left_anchor_pos-right_anchor_pos)-1)
            force = self.k_LE * strain
            self.loop_extruder_dict[leid]['resisting_force']=force
            self.loop_extruder_dict[leid]['strain'].append(strain)
            # print('tunisia', leid, force, strain, int(self.loop_extruder_dict[leid]['left_anchor']), int(self.loop_extruder_dict[leid]['right_anchor']))
    
    def _traverse_LE(self, dt, LE_id, f_prev):
        t_mesh=self.t_mesh
        if t_mesh>=dt: t_mesh = dt/5.0
        
        # coords = self.sim3D.state.getPositions(asNumpy=True)
        # left_anchor_pos = coords[int(self.loop_extruder_dict[LE_id]['left_anchor']),:]
        # right_anchor_pos = coords[int(self.loop_extruder_dict[LE_id]['right_anchor']),:]
        # strain=(np.linalg.norm(left_anchor_pos-right_anchor_pos)-1)
        
        f_resist = self.loop_extruder_dict[LE_id]["resisting_force"]
        
        # f_resist = self.k_LE * np.mean(self.loop_extruder_dict[LE_id]["strain"])
        
        f_t_minus_dt = f_prev * 0.5*(1+np.tanh(0.1*(self.f_stall-f_resist)))# self.loop_extruder_dict[LE_id]['active_force'] 
        tau = self.loop_extruder_dict[LE_id]['corr_time']
        
        f0 = self.v_LE * self.drag * 0.5*(1+np.tanh(0.1*(self.f_stall-f_resist)))
        
        Delta_x = 0.0
        # print(x_t_plus_dt)
        for t in  np.arange(t_mesh, dt, t_mesh):
            f_t = f_t_minus_dt * np.exp(-t_mesh/tau) + f0 * np.sqrt(1-np.exp(-2 * t_mesh / tau)) * self.rng.normal(loc=0, scale=1.0)
            #stall force
            # f_t *= 0.5*(1+np.tanh(0.02*(150.0-f_resist)))
            Delta_x += (t_mesh * f_t/self.drag) + np.sqrt(2*self.temp*t_mesh/self.drag) * self.rng.normal(loc=0, scale=1.0)
            
            f_t_minus_dt = f_t
            
        # self.loop_extruder_dict[LE_id]['active_force'] = f_t_minus_dt
        print(f'traverse_LE: dx={Delta_x} | force={f_t} | fres={f_resist, self.loop_extruder_dict[LE_id]["resisting_force"]} | cutoff={0.5*(1+np.tanh(0.1*(self.f_stall-f_resist)))}, dt={dt}, t_mesh={t_mesh}')
        return Delta_x, f_t_minus_dt
        
    def reset_LE(self, LE_id):
        """
        Resets the times associated with the LE. 
        (Updates one key in loop extruder dict)
        """
        self.loop_extruder_dict[LE_id]['bound_time'] = 0.0
        self.loop_extruder_dict[LE_id]['left_anchor_paused_time']= 0.0 
        self.loop_extruder_dict[LE_id]['right_anchor_paused_time']= 0.0 
        self.loop_extruder_dict[LE_id]['both_anchors_paused_time'] = 0.0
        self.loop_extruder_dict[LE_id]['is_left_anchor_paused'] = False 
        self.loop_extruder_dict[LE_id]['is_right_anchor_paused'] = False
        self.loop_extruder_dict[LE_id]['left_anchor_paused_time'] = 0.0 
        self.loop_extruder_dict[LE_id]['right_anchor_paused_time'] = 0.0
        self.loop_extruder_dict[LE_id]['both_anchors_paused_time'] = 0.0
        self.loop_extruder_dict[LE_id]['hop_left_factor'] = 1.0 
        self.loop_extruder_dict[LE_id]['strain'] = [0.0] 
        self.loop_extruder_dict[LE_id]['hop_right_factor'] = 1.0
        self.loop_extruder_dict[LE_id]['off_rate_factor'] = 1.0
        self.loop_extruder_dict[LE_id]['left_anchor_paused_at']= -1 
        self.loop_extruder_dict[LE_id]['right_anchor_paused_at']= -1
        self.loop_extruder_dict[LE_id]['left_anchor_paused_by']= ['None','None'] 
        self.loop_extruder_dict[LE_id]['right_anchor_paused_by']= ['None','None']
        self.loop_extruder_dict[LE_id]["v_right_anchor"] = self.v_LE
        self.loop_extruder_dict[LE_id]["v_left_anchor"] = self.v_LE
        self.loop_extruder_dict[LE_id]["active_force_left"] = self.v_LE * self.drag
        self.loop_extruder_dict[LE_id]["active_force_right"] = self.v_LE * self.drag
        self.loop_extruder_dict[LE_id]["resisting_force"] = 0.0
    
    def update_blockers_all_LEs(self,):
        """
        Update the blocking status of all LEs in the loop extruder dict
        """
        for le_id, le_dict in self.loop_extruder_dict.items():
            left_anchor = le_dict['left_anchor']
            right_anchor = le_dict['right_anchor']
            print(left_anchor, right_anchor)
            left_blocked_on_left_by = self.what_occupies_the_mono(int(left_anchor)-1)
            left_blocked_on_right_by = self.what_occupies_the_mono(int(left_anchor)+1)
            
            right_blocked_on_left_by = self.what_occupies_the_mono(int(right_anchor)-1)
            right_blocked_on_right_by = self.what_occupies_the_mono(int(right_anchor)+1)

            le_dict['left_anchor_paused_by'] = [left_blocked_on_left_by['name'], left_blocked_on_right_by['name']]
            le_dict['is_left_anchor_paused'] = (left_blocked_on_left_by['name']!='None' and left_blocked_on_right_by['name']!='None')
            
            le_dict['right_anchor_paused_by'] = [right_blocked_on_left_by['name'], right_blocked_on_right_by['name']]
            le_dict['is_right_anchor_paused'] = (right_blocked_on_left_by['name']!='None' and right_blocked_on_right_by['name']!='None')
            
            le_dict['hop_right_factor'] = 1.0
            le_dict['hop_left_factor'] = 1.0
            le_dict['off_rate_factor'] = 1.0
            
            #unbinding at the ends
            if left_blocked_on_left_by['name']=='End' or right_blocked_on_right_by['name']=='End':
                if self.unbind_at_end==True:
                    le_dict['off_rate_factor'] = 100.0

            #CTCF orientation dependence
            if left_blocked_on_left_by['name']=='ctcf':
                if self.implement_CTCF_orientation==True and left_blocked_on_left_by['orientation']=='-':
                    le_dict['hop_left_factor'] = self.ctcf_hop_factor
                    le_dict['off_rate_factor'] = self.ctcf_off_rate_factor
                    le_dict['is_left_anchor_paused'] = True
                    
            if right_blocked_on_right_by['name']=='ctcf':
                if self.implement_CTCF_orientation==True and right_blocked_on_right_by['orientation']=='+':
                    le_dict['hop_right_factor'] = self.ctcf_hop_factor
                    le_dict['off_rate_factor'] = self.ctcf_off_rate_factor
                    le_dict['is_right_anchor_paused'] = True

            #nucleosome hopping
            # if left_blocked_by['name']=='nucl':
            #     le_dict['hop_left_factor']=5.0

            # if right_blocked_by['name']=='nucl':
            #     le_dict['hop_right_factor']=5.0
            
    def what_occupies_the_mono(self, mono):
        """
        Return which element occupies the mono
        """
        ret = {}
        #check immobile blockers
        if int(mono) in self.immobile_blockers.keys():
            ret['name']=self.immobile_blockers[mono]['type']
            ret['orientation']=self.immobile_blockers[mono]['orientation']

        #check mobile blockers
        elif int(mono) in self.mobile_blockers:
            for le_id, le_dict in self.loop_extruder_dict.items():
                if int(le_dict["left_anchor"])==mono:
                    ret['name']=le_id
                    ret['orientation']='left'
                elif int(le_dict["right_anchor"])==mono:
                    ret['name']=le_id
                    ret['orientation']='right'
        
        #site must be free
        else:
            assert self.free_for_binding(mono), f'Consistency check failed: Monomer {mono} should be free for binding!'
            ret['name']='None'
            ret['orientation']='None'
        
        return ret
    
    def print_extruder_current_info(self, le_id):
        """
        Print the current information of the loop extruder

        Args:
        - le_id: which extruder to print info for

        Returns: 
        - None
        """
        print_string = "{0:7s} currently anchored at  ({1:6.2f},{2:6.2f}) | left blocked by ({3:7s},{4:7s}) | right blocked by ({5:7s},{6:7s})"
        print(print_string.format(le_id,
                                  self.loop_extruder_dict[le_id]['left_anchor'], 
                                  self.loop_extruder_dict[le_id]['right_anchor'], 
                                  self.loop_extruder_dict[le_id]['left_anchor_paused_by'][0], self.loop_extruder_dict[le_id]['left_anchor_paused_by'][1], 
                                  self.loop_extruder_dict[le_id]['right_anchor_paused_by'][0], self.loop_extruder_dict[le_id]['right_anchor_paused_by'][1],
                                  ))

    def print_extruder_old_info(self, le_id):
        """
        Print the previously stored information of the loop extruder

        Args:
        - le_id: which extruder to print info for

        Returns: 
        - None
        """
        print_string = "{0:7s} previously anchored at ({1:6.2f},{2:6.2f}) | left blocked by ({3:7s},{4:7s}) | right blocked by ({5:7s},{6:7s})"
        print(print_string.format(le_id,
                                  self.loop_extruder_dict_old[le_id]['left_anchor'], 
                                  self.loop_extruder_dict_old[le_id]['right_anchor'], 
                                  self.loop_extruder_dict_old[le_id]['left_anchor_paused_by'][0], self.loop_extruder_dict_old[le_id]['left_anchor_paused_by'][1],
                                  self.loop_extruder_dict_old[le_id]['right_anchor_paused_by'][0], self.loop_extruder_dict_old[le_id]['right_anchor_paused_by'][1],
                                  ))
        
    def print_latest_event(self,):
        """
        Find out the possible event using the loop extruder dict
        this is a consistencycheck that the event has been correctly implemented in Loop extruder dict
        Prints the event and the positions of the loop extrude involved in the event

        Args:
        - None

        Returns: 
        - None
        """
        le_id = self.current_LE
        le_dict = self.loop_extruder_dict[le_id]
        le_dict_old = self.loop_extruder_dict_old[le_id]
        
        # unbinding event -- both anchors moved
        if (le_dict['left_anchor'] != le_dict_old['left_anchor'] and
            le_dict['right_anchor'] != le_dict_old['right_anchor']):

            event = f"{le_id} Un/rebinding"

        #move left -- only left anchor has moved
        elif (le_dict['left_anchor'] != le_dict_old['left_anchor'] and
            le_dict['right_anchor'] == le_dict_old['right_anchor']):
            
            #check the pausing status to distinguish between a step and a hop event

            # pasuing status did not change -- it was a stepping event without encountering a boundary
            if le_dict['is_left_anchor_paused'] == le_dict_old['is_left_anchor_paused']:
                
                event = f"{le_id} Stepped left"
            
            # pausing status changed
            else:
                
                # encountered a boundary while moving left -- left anchor blocked
                if le_dict['is_left_anchor_paused']:
                    blocker = self.what_occupies_the_mono(int(le_dict['left_anchor']-1.0))
                    event = f"{le_id} Stepped left -- encountered a blocker"
                
                #left anchor got unpaused -- hopping event
                else:
                    event = f"{le_id} Hopped left"
            
        #move right -- only right anchor has moved
        elif (le_dict['left_anchor'] == le_dict_old['left_anchor'] and
            le_dict['right_anchor'] != le_dict_old['right_anchor']):

            #check the pausing status to distinguish between a step and a hop event

            # pasuing status did not change -- it was a stepping event without encountering a boundary
            if le_dict['is_right_anchor_paused'] == le_dict_old['is_right_anchor_paused']:

                event = f"{le_id} Stepped right"
            
            #pausing status changed
            else:
                
                # encountered a boundary while moving right -- right anchor blocked
                if le_dict['is_right_anchor_paused']:
                    blocker = self.what_occupies_the_mono(int(le_dict['right_anchor']+1.0))
                    event = f"{le_id} Stepped right -- encountered a blocker"
                
                #right anchor got unpaused -- hopping event
                else:
                    event = f"{le_id} Hopped right"
                
        #stuck -- anchors did not move and pausing status did not change
        elif (le_dict['left_anchor'] == le_dict_old['left_anchor'] and
            le_dict['right_anchor'] == le_dict_old['right_anchor'] and 
            le_dict['is_left_anchor_paused'] and le_dict['is_right_anchor_paused']):

            blocker_left = self.what_occupies_the_mono(int(le_dict['left_anchor']-1.0))
            blocker_right = self.what_occupies_the_mono(int(le_dict['right_anchor']+1.0))
    
            event = f"{le_id} tried Hopping but is stuck!"
            
        #stuck at zero loop state -- anchors did not move but not paused
        elif (le_dict['left_anchor'] == le_dict_old['left_anchor'] and
            le_dict['right_anchor'] == le_dict_old['right_anchor'] ):

            blocker_left = self.what_occupies_the_mono(int(le_dict['left_anchor']-1.0))
            blocker_right = self.what_occupies_the_mono(int(le_dict['right_anchor']+1.0))
    
            event = f"{le_id} Zero loop state!"
            
        else: 
            print('new',le_dict)
            print('old', le_dict_old)
            raise AssertionError ("Cannot identify the event!")
        
        print(event)
        self.print_extruder_current_info(le_id)
        self.print_extruder_old_info(le_id)
        print("\n")

    def update_immobile_blockers(self, blockers_dict):
        """
        Update the immobile blockers dictionary
        """
        #check legality
        for key in blockers_dict.keys(): 
            assert 'type' in blockers_dict[key] and 'orientation' in blockers_dict[key], 'blockers dict is not right!'
            assert self.is_site_within_chain(key), 'some blockers outside the chain!'

        self.immobile_blockers.update(blockers_dict)

    def save_anchors(self, time):
        """
        Save the anchor locations to anchor_loc_dict
        """
        if not hasattr(self, 'anchor_loc_dict'):
            self.anchor_loc_dict = {}

        self.anchor_loc_dict[time] = [[(le_dict['left_anchor'], le_dict['right_anchor']) for le_id, le_dict in self.loop_extruder_dict.items()],
                                      [le_id for le_id, le_dict in self.loop_extruder_dict.items()]]
        

    def output_anchor_locations(self, folder):
        """
        write the anchor locations to a file
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        # with h5py.File(os.path.join(folder, self.name+'_anchors.h5'),'w') as fout:
        #     for time, anchors in self.anchor_loc_dict.items():
        #         fout.create_dataset(name=str(time), data=anchors)
        with open(os.path.join(folder, self.name+'_anchors.txt'), 'w+') as fout:
            for time,anchors in self.anchor_loc_dict.items():
                fout.write("{0:10.5f}".format(time))
                for left,right in anchors[0]:
                    fout.write("\t{0:10.4f}\t{1:10.4f}".format(left, right))
                fout.write("\n")

    def set_CTCF_orientation_rule(self, bool_value):
        self.implement_CTCF_orientation = bool_value
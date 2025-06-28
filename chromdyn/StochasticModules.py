import numpy as np
from typing import Any, Dict, Union, List, Tuple, Optional
import types
from pathlib import Path
import copy
import os
from scipy.spatial import distance
from Utilities import LogManager

class Loop_Extruders:
    
    def __init__(
        self,
        num_LE: int,
        topology: Union[List[Tuple[int, int]], object],
        name: str = 'SMC',
        output_dir: str = "output",
        k_off: float = 2e-3,
        kstep_LE: float = 0.1,
        k_hop: float = 1e-2,
        max_hop_len: int = 5,
        temp: float = 0.0,
        hop3D: bool = False,
        sim3D: Optional[object] = None,
        dist3D_cutoff: float = 1.5,
        t_mesh: float = 0.001,
        tau: float = 1e6,
        v_LE: float = 5.0,
        k_LE: float = 30.0,
        f_stall: float = 50.0,
        logger: Optional[object] = None,
        console_stream: bool = True, 
        ) -> None:
        
        self.name = name
        self.num_LE = num_LE
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or LogManager(log_file=self.output_dir / f"{self.name}.log").get_logger(__name__, console=console_stream)
        self.logger.info("Initializing the Loop Extruder class ... ")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Number of extruders: {self.num_LE}")
        self._initialize_rng()
        self.preferential_LE_loading = 0.0
        self.hop3D = hop3D
        self.sim3D = sim3D
        self.dist3D_cutoff = dist3D_cutoff
        self._initialize_LE_params(k_LE, f_stall, k_off, kstep_LE, k_hop, tau, v_LE, t_mesh, max_hop_len, temp)
        self._initialize_topology(topology)
        self._initialize_LE_dict()
        self._initialize_blockers()
        self.valid_pausing_rules = ('both_blocked', 'forward_blocked')
        self.valid_hopping_rules = ('independent', 'both_paused')
        
        self.stats = LoopExtruderStats(self.num_LE)
        
        self.time: float = 0  # Initialize time
        self.dt: float = 0
        self.current_event: Dict[str, Any] = {}
        
    def _initialize_rng(self) -> None:
        seed = np.random.randint(100000000)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.logger.info(f"Random seed: {seed}")

    def _initialize_LE_params(self, k_LE, f_stall, k_off, kstep_LE,
                              k_hop, tau, v_LE, t_mesh, max_hop_len, temp) -> None:
        self.max_hop_len = max_hop_len
        self.k_LE = k_LE
        self.f_stall = f_stall
        self.k_off = k_off
        self.k_hop = k_hop
        self.kstep_LE = kstep_LE
        self.kstep_LE_factor = 1.0
        self.tau = tau
        self.v_LE = v_LE
        self.t_mesh = t_mesh
        self.drag = 1.0
        self.temp = temp
        self.unbind_at_end = True
        self.end_unbinding_factor = 1e5
        self.CTCF_rule = None
        self.ctcf_hop_factor = None
        self.ctcf_off_rate_factor = None
        
        self.logger.info(f"k_step: {self.kstep_LE} | k_off: {self.k_off} | k_hop : {self.k_hop}")
        self.logger.info(f"v_LE: {self.v_LE} | temp: {self.temp} | persistent time: {self.tau} | max hop length: {self.max_hop_len}")
        self.logger.info(f"Unbind at ends: {self.unbind_at_end}")
        
        if self.sim3D is not None:
            self.logger.info(f"3D simulation activated")
            self._validate_hop3D()
            self.logger.info(f"LE bonds[k : {self.k_LE}, r : 1.0] | F_stall: {self.f_stall} ")
    
    def _validate_hop3D(self) -> None:
        self.logger.info(f"3D hopping of Extruders: {self.hop3D}")
        if not isinstance(self.hop3D, bool):
            raise TypeError("hop3D must be a boolean.")
        if self.hop3D and self.sim3D is None:
            raise ValueError("For 3D hopping, polymer object (sim3D) must be provided.")

    def _initialize_topology(self, topology: Union[List[Tuple[int, int]], object],) -> None:
        if type(topology)==List:
            # assume topology=[(chain1_start, chain1_end), ... ]
            self.chains = topology
        else:
            #assume openmm topology object
            self.chains = [(int(list(xx.atoms())[0].id) ,int(list(xx.atoms())[-1].id)) 
                           for xx in list(topology.chains())]
        self.logger.info(f"Chain topology: {self.chains}")

    def _initialize_LE_dict(self) -> None:
        self.current_LE = -1
        self.LE_steps = []
        self.loop_extuder_3Dneighbors = {i: [] for i in range(self.num_LE)}
        self.loop_extruder_dict = {f"LE{i}": {'bond_index': -1, 'left_anchor': 0, 'right_anchor': 1} 
                                   for i in range(self.num_LE)}
        
        for leid, ledict in self.loop_extruder_dict.items():
            ledict.update(self.get_default_extruder_params())
    
    def get_leids(self) -> List:
        return list(self.loop_extruder_dict.keys())
    
    def get_default_extruder_params(self) -> Dict[str, Any]:
        deafult_params_dict = {
                'bound_time': 0.0,
                'is_left_anchor_paused': False, 
                'is_right_anchor_paused': False,
                'pausing_rule': 'forward_blocked',
                'left_anchor_paused_time': 0.0, 
                'right_anchor_paused_time': 0.0, 
                'both_anchors_paused_time': 0.0,
                'k_step_left': self.kstep_LE,
                'k_step_left_factor': self.kstep_LE_factor, 
                'k_step_right': self.kstep_LE,
                'k_step_right_factor': self.kstep_LE_factor,
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
                'active_force_left': self.v_LE * self.drag, 
                'active_force_right': self.v_LE * self.drag,
                'corr_time': self.tau, 
                'resisting_force': 0.0,
                '3D_neighbors':[],
            }
        return deafult_params_dict
    
    def _initialize_blockers(self) -> None:
        self.mobile_blockers = [0] * self.num_LE + [1] * self.num_LE
        self.immobile_blockers = {}
        for chain in self.chains:
            start = int(chain[0])
            end = int(chain[1])
            self.immobile_blockers[start]= {'type':'End','orientation':'left'}
            self.immobile_blockers[end]= {'type':'End','orientation':'right'}
        self.first_mono = int(self.chains[0][0])
        self.last_mono = int(self.chains[-1][1])
    
    def update_immobile_blockers(self, blockers_dict: Dict[int, Dict[str, str]]):
        """
        Update the immobile blockers dictionary
        """
        #check legality
        for key in blockers_dict.keys(): 
            assert 'type' in blockers_dict[key] and 'orientation' in blockers_dict[key], 'blockers dict is not right!'
            assert self.is_site_within_chain(key), 'some blockers outside the chain!'

        self.immobile_blockers.update(blockers_dict)

    def set_up(self, init_config: Optional[List[Tuple[int, int]]] = None) -> None:
        self._place_LEs(init_config)
        self._backup_LE_state()
        self.save_anchors(time=0.0)
        
        self.logger.info("Set up complete.")
    
    def _place_LEs(self, init_config: Optional[List[Tuple[int, int]]] = None) -> None:
        if init_config is None:
            self.logger.info("Randomly distributing the LEFs ... ")
            for LE_id in self.loop_extruder_dict:
                left, right = self.find_random_binding_site()
                self.set_LE_left_anchor(LE_id, left, -1)
                self.set_LE_right_anchor(LE_id, right, -1)
        else:
            if len(init_config) != self.num_LE:
                raise ValueError("init_config length must match num_LE.")
            for i, LE_id in enumerate(self.loop_extruder_dict):
                left, right = init_config[i]
                self.set_LE_left_anchor(LE_id, left, -1)
                self.set_LE_right_anchor(LE_id, right, -1)

    def _backup_LE_state(self) -> None:
        self.loop_extruder_dict_old = copy.deepcopy(self.loop_extruder_dict)
    
    def save_anchors(self, time):
        """
        Save the anchor locations to anchor_loc_dict
        """
        if not hasattr(self, 'anchor_loc_dict'):
            self.anchor_loc_dict = {}

        self.anchor_loc_dict[time] = [[(le_dict['left_anchor'], le_dict['right_anchor']) for le_id, le_dict in self.loop_extruder_dict.items()],
                                      [le_id for le_id, le_dict in self.loop_extruder_dict.items()]]
    
    def set_prefential_loading(self, preferential_loading: float, loading_sites: List[int]) -> None:
        preferential_loading = float(preferential_loading)
        if not (0.0 <= preferential_loading <= 1.0):
            raise ValueError("Preferential loading must be between 0.0 and 1.0.")
        if preferential_loading > 0.0:
            assert loading_sites is not None and len(loading_sites)>0 , 'Preferential loading is ON but no loading sites provided. Use loading_sites list.'
        self.preferential_LE_loading = preferential_loading
        self.preferential_loading_sites = loading_sites or []
        self.logger.info(f"Preferential loading: {bool(self.preferential_loading)} | Number of loading sites: {len(self.preferential_loading_sites)}")
        
    def get_preferential_loading(self) -> float:
        return self.preferential_LE_loading
    
    def get_preferential_loading_sites(self) -> List[int]:
        return self.preferential_loading_sites
    
    def set_CTCF_params(self, hop_factor: float, off_factor: float):
        self.CTCF_rule = True
        self.logger.int(f"CTCF rule: {self.CTCF_rule}")
        self.ctcf_hop_factor = hop_factor
        self.ctcf_off_rate_factor = off_factor
        self.logger.info(f"LE off-rate factor upon oriented CTCF encounter: {self.ctcf_off_rate_factor}")
        self.logger.info(f"LE hop-rate factor upon oriented CTCF encounter: {self.ctcf_hop_factor}")
    
    def find_random_binding_site(self,):
        """
        Find a random binding site for a LE
        """
        keep_searching = True
        maxiter=0
        while keep_searching:
            maxiter += 1
            anchor_i = self.rng.choice(range(self.first_mono+1, self.last_mono-1))
            if self.free_for_binding(anchor_i) and self.free_for_binding(anchor_i+1): 
                keep_searching = False
                new_anchors = (anchor_i, anchor_i+1)
            if maxiter>1000:
                raise ValueError(f"Cannot find random binding site. Exceeded maxiter={maxiter}")
        return new_anchors

    def get_off_rate(self, leid):
        return self.loop_extruder_dict[leid]["off_rate_factor"] * self.loop_extruder_dict[leid]['k_off']
    
    def get_step_left_rate(self, leid):
        is_paused = self.loop_extruder_dict[leid]['is_left_anchor_paused']
        return (not is_paused) * self.loop_extruder_dict[leid]['k_step_left_factor'] * self.loop_extruder_dict[leid]['k_step_left']
    
    def get_step_right_rate(self, leid):
        is_paused = self.loop_extruder_dict[leid]['is_right_anchor_paused']
        return (not is_paused) * self.loop_extruder_dict[leid]['k_step_right_factor'] * self.loop_extruder_dict[leid]['k_step_right']
    
    def get_hop_left_rate(self, leid):
        factor = self.loop_extruder_dict[leid]["hop_left_factor"] 
        return self.is_LE_paused(leid) * factor * self.loop_extruder_dict[leid]['k_hop_left']
    
    def is_LE_paused(self, leid):
        is_paused = self.loop_extruder_dict[leid]['is_left_anchor_paused'] * self.loop_extruder_dict[leid]['is_right_anchor_paused']
        return is_paused
    
    def get_hop_right_rate(self, leid):
        is_paused = self.loop_extruder_dict[leid]['is_left_anchor_paused'] * self.loop_extruder_dict[leid]['is_right_anchor_paused']
        factor = self.loop_extruder_dict[leid]["hop_right_factor"] 
        return is_paused * factor * self.loop_extruder_dict[leid]['k_hop_right']
    
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
        return (self.last_mono>=int(mono)>=self.first_mono)
    
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
    
    def simulate_step(self, verbose = True) -> None:
        """
        Perform a single simulation step.

        Returns:
        - None
        """
        # Gillespie part
        self.create_rate_vec()
        time: float = self.get_time_to_next_event()
        event: Dict[str, Any] = self.get_next_event()
        self.current_event = event
        self.dt = time
        self.time += time
        if bool(verbose) == True: 
            self.print_event(event)
    
        # Loop extruder part
        self.update_LEs(event, time)
        self.save_anchors(self.time)    
        # self.logger.info(f"Rate vector: {self.rate_vec}")
        if bool(verbose) == True: 
            self.print_latest_event()
        
        self.stats.record_state(extruder_id= int(self.current_LE.strip("LE")), dt=self.dt, time=self.time, step_size=self.current_LE_step, extruder_dict=self.loop_extruder_dict)
        
    def create_rate_vec(self) -> None:
        """
        Create the rate vector with each LEF leg translocation as independent events.

        Returns:
        - None
        """
        self.rate_vec: np.ndarray = np.array([
            [
                self.get_off_rate(key),
                self.get_step_left_rate(key),
                self.get_step_right_rate(key),
                self.get_hop_left_rate(key),
                self.get_hop_right_rate(key)
            ]
            for key in self.loop_extruder_dict.keys()
        ]).flatten()

        self.event_matrix: np.ndarray = np.array([
            [
                {'LE_id': key, 'move_left_anchor': False, 'move_right_anchor': False, 'unbind': True, 'hop_left': False, 'hop_right': False},
                {'LE_id': key, 'move_left_anchor': True, 'move_right_anchor': False, 'unbind': False, 'hop_left': False, 'hop_right': False},
                {'LE_id': key, 'move_left_anchor': False, 'move_right_anchor': True, 'unbind': False, 'hop_left': False, 'hop_right': False},
                {'LE_id': key, 'move_left_anchor': False, 'move_right_anchor': False, 'unbind': False, 'hop_left': True, 'hop_right': False},
                {'LE_id': key, 'move_left_anchor': False, 'move_right_anchor': False, 'unbind': False, 'hop_left': False, 'hop_right': True}
            ]
            for key in self.loop_extruder_dict.keys()
        ]).flatten()
            
    def get_time_to_next_event(self) -> float:
        """
        Return the time to simulate before the next event occurs.

        Returns:
        - time_to_next_event: Time to next event
        """
        mean_time: float = 1. / self.rate_vec.sum()
        time_to_next_event: float = self.rng.exponential(scale=mean_time)
        return time_to_next_event
    
    def get_next_event(self) -> Dict[str, Any]:
        """
        Return the id and anchor of the LEF to translocate.

        Returns:
        - event_vec: Next event
        """
        event_vec: Dict[str, Any] = self.rng.choice(self.event_matrix, size=1, p=self.rate_vec / self.rate_vec.sum())[0]
        return event_vec

    def print_event(self, event: Dict[str, Any]) -> None:
        """
        Extract the event from evenet vector and print 

        Args:
        - event: the selected event dict

        Returns:
        - None
        """
        leid: str = event['LE_id']
        true_events: list[str] = [key for key, value in event.items() if value is True and key != 'LE_id']
        # dx_left: float = self.lef.loop_extruder_dict[leid]['left_anchor'] - self.lef.loop_extruder_dict_old[leid]['left_anchor']
        # dx_right: float = self.lef.loop_extruder_dict[leid]['right_anchor'] - self.lef.loop_extruder_dict_old[leid]['right_anchor']
        self.logger.info(f"Time: {self.time:<10.5f} | dt: {self.dt:^10.5f} | {leid:^6s} {true_events[0]:<20s}")

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

        self._backup_LE_state()

        #check for the legality of the event dict -- there must be only one event
        assert sum([event[key] for key in event.keys() if key!="LE_id"])==1 , " More than one event is True in the event dict!"

        if event['unbind']:
            
            self.reset_LE(LE_id) #reset bound time
            
            #decide if loading at preferntial sites
            if self.rng.uniform(low=0.0, high=1.0) < self.preferential_LE_loading:
                left_anchor, right_anchor = self.find_preferential_loading_site()
            else:
                left_anchor, right_anchor = self.find_random_binding_site()
            
            self.set_LE_left_anchor(LE_id, left_anchor, -1)
            self.set_LE_right_anchor(LE_id, right_anchor, -1)
            self.update_all_LE_times(dt, exclude=LE_id)
            self.current_LE_step = None
            
        elif event['move_left_anchor']:
            x_t = self.loop_extruder_dict[LE_id]["left_anchor"]
            Delta_x, f_t = self._traverse_LE(dt, LE_id, self.loop_extruder_dict[LE_id]['active_force_left'])
            x_t_plus_dt = x_t - Delta_x
            new_left_anchor, roadblock = self.check_for_roadblocks(x_t, x_t_plus_dt)
            self.loop_extruder_dict[LE_id]['active_force_left'] = f_t
            self.set_LE_left_anchor(LE_id, new_left_anchor, roadblock)
            self.update_all_LE_times(dt)
            self.current_LE_step = Delta_x
            
        elif event['move_right_anchor']:
            x_t = self.loop_extruder_dict[LE_id]["right_anchor"]
            Delta_x, f_t = self._traverse_LE(dt, LE_id, self.loop_extruder_dict[LE_id]['active_force_right'])
            x_t_plus_dt = x_t + Delta_x
            new_right_anchor, roadblock = self.check_for_roadblocks(x_t,x_t_plus_dt)
            self.loop_extruder_dict[LE_id]['active_force_right'] = f_t
            self.set_LE_right_anchor(LE_id, new_right_anchor, roadblock)
            self.update_all_LE_times(dt)
            self.current_LE_step = Delta_x
            
        elif event['hop_left']:
            x_t = self.loop_extruder_dict[LE_id]["left_anchor"]
            self.current_LE_step = None
            hopped=0
            if self.hop3D:
                self.update_LE_3Dneighbors(cutoff=self.dist3D_cutoff)
                possible_anchors = self.loop_extruder_dict[LE_id]['3D_neighbors'] #[xx for xx in self.loop_extuder_3Dneighbors[LE_id]]
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
            self.current_LE_step = None
            hopped=0
            
            if self.hop3D:
                self.update_LE_3Dneighbors(cutoff=self.dist3D_cutoff)
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
    
    def update_all_LE_times(self, dt, exclude=None):
        """
        Update the bound time of all LEs
        """
        for le_id in self.loop_extruder_dict.keys():
            if le_id != exclude:
                self.loop_extruder_dict[le_id]['bound_time'] += dt

    def _traverse_LE(self, dt, LE_id, f_t_minus_dt):
        t_mesh=self.t_mesh
        if t_mesh>dt/5.0: t_mesh = dt/5.0
        tau = self.loop_extruder_dict[LE_id]['corr_time']
        f0 = self.v_LE * self.drag
        polymer_feedback = self.get_polymer_feedback_factor(LE_id)
        Delta_x = 0.0
        for t in  np.arange(t_mesh, dt, t_mesh):
            f_t = f_t_minus_dt * np.exp(-t_mesh/tau) + f0 * np.sqrt(1-np.exp(-2 * t_mesh / tau)) * self.rng.normal(loc=0, scale=1.0)
            f_t *= polymer_feedback
            Delta_x += (t_mesh * f_t/self.drag) + np.sqrt(2*self.temp*t_mesh/self.drag) * self.rng.normal(loc=0, scale=1.0)
            f_t_minus_dt = f_t
        # self.loop_extruder_dict[LE_id]['active_force'] = f_t_minus_dt
        # print(f'traverse_LE: dx={Delta_x} | force={f_t} | fres={f_resist, self.loop_extruder_dict[LE_id]["resisting_force"]} | cutoff={0.5*(1+np.tanh(0.1*(self.f_stall-f_resist)))}, dt={dt}, t_mesh={t_mesh}')
        return Delta_x, f_t
    
    def get_polymer_feedback_factor(self, leid, mu=0.1):
        f_polymer_restore = self.loop_extruder_dict[leid]["resisting_force"]
        factor = 0.5*(1.0 + np.tanh(mu*(self.f_stall-f_polymer_restore)))
        return factor
 
    def check_for_roadblocks(self, xi, xf):
        """
        Checks if extrusion can happen in a segment and returns the final position. 
        If there are roadblocks along the way, final position is modified accordingly.
        """
        x_out = xi
        extrude_len = 0.0
        roadblock = -1
        
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
                    #check legality
                    assert self.is_site_within_chain(roadblock), "Roadblock outside the polymer!!"         
                    break
        
        #if no roadblocks extrude to xf
        if roadblock==-1:
            x_out = xf
        # otherwise extrude according to extrude_len
        else:
            x_out = xi + extrude_len
        return (x_out, roadblock)

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
 
    def update_LE_resisting_force(self,leids, anchor_locs):
        # print(leids, anchor_locs)
        state = self.sim3D.simulation.context.getState(getPositions=True)
        coords = state.getPositions(asNumpy=True)
        for xx,leid in enumerate(leids):
            left_anchor_pos = coords[int(anchor_locs[xx][0]),:]
            right_anchor_pos = coords[int(anchor_locs[xx][1]),:]
            strain=(np.linalg.norm(left_anchor_pos-right_anchor_pos)-1)
            force = self.k_LE * strain
            self.loop_extruder_dict[leid]['resisting_force']=force
            self.loop_extruder_dict[leid]['strain'].append(strain)
            # print('tunisia', leid, force, strain, int(self.loop_extruder_dict[leid]['left_anchor']), int(self.loop_extruder_dict[leid]['right_anchor']))
        
    def reset_LE(self, LE_id):
        """
        Resets the times associated with the LE. 
        (Updates one key in loop extruder dict)
        """
        self.loop_extruder_dict[LE_id].update(self.get_default_extruder_params())
    
    def set_pausing_rule(self, leid, rule):
        if rule not in self.valid_pausing_rules:
            raise ValueError(f"Invalid rule. Valid rules are {self.valid_pausing_rules}")
        self.loop_extruder_dict[leid]['pausing_rule'] = rule
    
    def get_pausing_rule(self, leid):
        return self.loop_extruder_dict[leid].get('pausing_rule')
    
    def is_left_anchor_paused(self, leid):
        left_anchor, right_anchor = self.get_anchors(leid) 
        left_blocked_on_left_by = self.what_occupies_the_mono(int(left_anchor)-1)
        left_blocked_on_right_by = self.what_occupies_the_mono(int(left_anchor)+1)
        if self.get_pausing_rule(leid) == 'both_blocked':
            is_paused = (left_blocked_on_left_by['name']!='None' and left_blocked_on_right_by['name']!='None')
        elif self.get_pausing_rule(leid) == 'forward_blocked':
            is_paused = (left_blocked_on_left_by['name']!='None')
        else:
            raise ValueError("Invalid pausing rule!")
        return is_paused, (left_blocked_on_left_by, left_blocked_on_right_by)
    
    def is_right_anchor_paused(self, leid):
        left_anchor, right_anchor = self.get_anchors(leid) 
        right_blocked_on_left_by = self.what_occupies_the_mono(int(right_anchor)-1)
        right_blocked_on_right_by = self.what_occupies_the_mono(int(right_anchor)+1)
        
        if self.get_pausing_rule(leid) == 'both_blocked':
            is_paused = (right_blocked_on_left_by['name']!='None' and right_blocked_on_right_by['name']!='None')
        elif self.get_pausing_rule(leid) == 'forward_blocked':
            is_paused = (right_blocked_on_right_by['name']!='None')
        else:
            raise ValueError("Invalid pausing rule!")
        return is_paused, (right_blocked_on_left_by, right_blocked_on_right_by)
    
    def update_blockers_all_LEs(self,):
        """
        Update the blocking status of all LEs in the loop extruder dict
        """
        for le_id, le_dict in self.loop_extruder_dict.items():
            
            is_right_paused, (right_blocked_on_left_by, right_blocked_on_right_by) = self.is_right_anchor_paused(le_id)
            is_left_paused, (left_blocked_on_left_by, left_blocked_on_right_by) = self.is_left_anchor_paused(le_id)
            # left_anchor, right_anchor = self.get_anchors(le_id) 
            # left_blocked_on_left_by = self.what_occupies_the_mono(int(left_anchor)-1)
            # left_blocked_on_right_by = self.what_occupies_the_mono(int(left_anchor)+1)
            
            # right_blocked_on_left_by = self.what_occupies_the_mono(int(right_anchor)-1)
            # right_blocked_on_right_by = self.what_occupies_the_mono(int(right_anchor)+1)

            le_dict['left_anchor_paused_by'] = [left_blocked_on_left_by['name'], left_blocked_on_right_by['name']]
            le_dict['is_left_anchor_paused'] = is_left_paused
            
            le_dict['right_anchor_paused_by'] = [right_blocked_on_left_by['name'], right_blocked_on_right_by['name']]
            le_dict['is_right_anchor_paused'] = is_right_paused
            
            le_dict['hop_right_factor'] = 1.0
            le_dict['hop_left_factor'] = 1.0
            le_dict['off_rate_factor'] = 1.0
            
            #unbinding at the ends
            if str(left_blocked_on_left_by['name']).lower()=='end' or str(right_blocked_on_right_by['name']).lower()=='end':
                if self.unbind_at_end==True:
                    le_dict['off_rate_factor'] = self.end_unbinding_factor

            #CTCF orientation dependence
            if str(left_blocked_on_left_by['name']).lower()=='ctcf':
                if self.CTCF_rule is None: 
                    self.logger.warning("CTCF encountered an as unoriented blocker!")
                if self.CTCF_rule==True and left_blocked_on_left_by['orientation']=='-':
                    le_dict['hop_left_factor'] = self.ctcf_hop_factor
                    le_dict['off_rate_factor'] = self.ctcf_off_rate_factor
                    le_dict['is_left_anchor_paused'] = True
                    
            if str(right_blocked_on_right_by['name']).lower()=='ctcf':
                if self.CTCF_rule is None: 
                    self.logger.warning("CTCF encountered an as unoriented blocker!")
                if self.CTCF_rule==True and right_blocked_on_right_by['orientation']=='+':
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

    def print_latest_event(self, tol: float = 1e-8):
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
        
        dx_left = - le_dict['left_anchor'] + le_dict_old['left_anchor']
        dx_right = le_dict['right_anchor'] - le_dict_old['right_anchor']

        # both anchors moved --> unbinding event
        if abs(dx_left) > tol and abs(dx_right) > tol:
            event = f"{le_id:8^s} Un/rebinding | dx_left: {dx_left:8.2f} | dx_right: {dx_right:8.2f}"

        #only left anchor has moved --> move/hop left
        elif abs(dx_left) > tol and abs(dx_right) < tol:
            #check the pausing status to distinguish between a step and a hop event
            if le_dict['is_left_anchor_paused'] == le_dict_old['is_left_anchor_paused']: 
                # pasuing status did not change --> stepping event WITHOUT encountering a boundary
                event = f"{le_id:8^s} Stepped left | dx_left: {dx_left:8.2f} | dx_right: {dx_right:8.2f}"
            else: 
                # pausing status changed --> stepped to encounter barrier or hopped
                if le_dict['is_left_anchor_paused']:
                    # left anchor paused --> encountered a boundary while moving left 
                    blocker = self.what_occupies_the_mono(int(le_dict['left_anchor']-1.0))
                    event = f"{le_id:8^s} Stepped left - hit blocker {blocker['name']:^8s}| dx_left: {dx_left:8.2f} | dx_right: {dx_right:8.2f}"
                else:
                    #left anchor got unpaused --> hopping event
                    event = f"{le_id:8^s} Hopped left | dx_left: {dx_left:8.2f} | dx_right: {dx_right:8.2f}"
            
        # only right anchor has moved --> move/hop right
        elif abs(dx_left) < tol and abs(dx_right) > tol:
            #check the pausing status to distinguish between a step and a hop event
            if le_dict['is_right_anchor_paused'] == le_dict_old['is_right_anchor_paused']:
                # pasuing status did not change --> stepping event WITHOUT encountering a boundary
                event = f"{le_id:8^s} Stepped right | dx_left: {dx_left:8.2f} | dx_right: {dx_right:8.2f}"
            else:
                # pausing status changed --> stepped to encounter barrier or hopped
                if le_dict['is_right_anchor_paused']:
                    # right anchor paused --> encountered a boundary while moving right
                    blocker = self.what_occupies_the_mono(int(le_dict['right_anchor']+1.0))
                    event = f"{le_id:8^s} Stepped right - hit blocker {blocker['name']:^8s}| dx_left: {dx_left:8.2f} | dx_right: {dx_right:8.2f}"
                #right anchor got unpaused -- >hopping event
                else:
                    event = f"{le_id:8^s} Hopped right | dx_left: {dx_left:8.2f} | dx_right: {dx_right:8.2f}"
                
        # both anchors did not move 
        elif abs(dx_left) < tol and abs(dx_right) < tol:
            blocker_left = self.what_occupies_the_mono(int(le_dict['left_anchor']-1.0))
            blocker_right = self.what_occupies_the_mono(int(le_dict['right_anchor']+1.0))
            
            if self.is_LE_paused(le_id):
                # both still paused --> Tried hopping but stuck
                event = f"{le_id:8^s} failed to Hop - stuck between {blocker_left['name']:^8s} and {blocker_right['name']:^8s}| dx_left: {dx_left:8.2f} | dx_right: {dx_right:8.2f}"
        
            #stuck at zero loop state -- anchors did not move but not paused
            else:
                event = f"{le_id:8^s} Zero loop state????| dx_left: {dx_left:8.2f} | dx_right: {dx_right:8.2f}"
            
        else: 
            self.logger.info('new',le_dict)
            self.logger.info('old', le_dict_old)
            raise AssertionError ("Cannot identify the event!")
        
        self.logger.info(event)
        self.print_extruder_current_info(le_id)
        self.print_extruder_old_info(le_id)
        self.logger.info("--"*10)
        
    def print_extruder_current_info(self, le_id):
        """
        Print the current information of the loop extruder

        Args:
        - le_id: which extruder to print info for

        Returns: 
        - None
        """
        
        print_string = "{0:^8s} IS at  ({1:8.2f},{2:8.2f}) | left blocked by ({3:^7s},{4:^7s}) | right blocked by ({5:^7s},{6:^7s})"
        self.logger.info(print_string.format(le_id,
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
        print_string = "{0:^8s} WAS at ({1:8.2f},{2:8.2f}) | left blocked by ({3:^7s},{4:^7s}) | right blocked by ({5:^7s},{6:^7s})"
        self.logger.info(print_string.format(le_id,
                                  self.loop_extruder_dict_old[le_id]['left_anchor'], 
                                  self.loop_extruder_dict_old[le_id]['right_anchor'], 
                                  self.loop_extruder_dict_old[le_id]['left_anchor_paused_by'][0], self.loop_extruder_dict_old[le_id]['left_anchor_paused_by'][1],
                                  self.loop_extruder_dict_old[le_id]['right_anchor_paused_by'][0], self.loop_extruder_dict_old[le_id]['right_anchor_paused_by'][1],
                                  ))
        
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

    def simulate(self, tstop:float, verbose:bool = True) -> None:
        t=0
        while t<tstop:
            self.simulate_step(verbose)
            t += self.dt

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
    
    # def reset_LE_resisting_forces(self,):
    #     for leid in self.loop_extruder_dict.keys():
    #         # self.loop_extruder_dict[leid]['resisting_force']=force
    #         self.loop_extruder_dict[leid]['strain']=[0.0]

    # def _integrate_force(self, f0, dt, tau):
    #     t_mesh=self.t_mesh
    #     times = np.arange(t_mesh, dt, t_mesh)
    #     f=np.zeros((len(f0),times.shape[0]+1))
    #     f[:,0]=f0
    #     for ii, _ in enumerate(times):
    #         f[:, ii+1] = f[:, ii] * np.exp(-t_mesh/tau) + np.sqrt(1-np.exp(-2 * t_mesh / tau)) * self.rng.normal(loc=0, scale=1.0, size=len(f0))
    #     return times, f
    
class LoopExtruderStats:
    """
    A helper class to record—and later retrieve—time‐series statistics
    (e.g. step_size, force, etc.) for each loop extruder.
    """

    def __init__(self, num_extruders: int) -> None:
        """
        Parameters
        ----------
        num_extruders : int
            The total number of extruders (same as num_LE in Loop_Extruders).
        """
        self.num_extruders = num_extruders

        # Initialize a dict mapping extruder ID (0..num_extruders-1) to lists of entries
        # Each entry will be a tuple (time, step_size). You can expand this to hold other fields.
        self._data: Dict[int, Dict[str, List[Any]]] = {
            leid: {
                "bound_times": [],      # list of float
                "step_sizes": []  # list of float
                # If you want more variables (e.g. force, position), add more lists here:
                # "forces": [],
                # "positions": [],
            }
            for leid in range(num_extruders)
        }
        
        self._traj: Dict[str, Any] = {'time_stamp': [], 
                                     'anchors' : []}

    def record_state(self, extruder_id: int, time: float, dt: float, step_size: Union[float, Any], extruder_dict: Dict[str, Any]) -> None:
        """
        Append one new (time, step_size) record for the given extruder.

        Parameters
        ----------
        extruder_id : int
            Index of the extruder (0 <= extruder_id < num_extruders).
        time : float
            Simulation time (or whatever time unit) at which this step occurred.
        step_size : float
            The step size taken by this extruder at 'time'.
        """
        if not (0 <= extruder_id < self.num_extruders):
            raise IndexError(f"extruder_id {extruder_id} is out of range [0, {self.num_extruders})")
        if step_size is not None:
            self._data[extruder_id]["step_sizes"].append(step_size)
        else:
            bound_time = extruder_dict[f"LE{extruder_id}"]['bound_time']
            self._data[extruder_id]["bound_times"].append(bound_time)
        
        anchors = [(extruder_dict[idx]['left_anchor'], extruder_dict[idx]['right_anchor']) for idx in extruder_dict.keys()]
        self._traj['time_stamp'].append(time)
        self._traj['anchors'].append(anchors)
    
    def get_LEsteps(self, extruder_id: int) -> np.array:
        """
        Retrieve the full list of (time, step_size) for the given extruder.

        Returns
        -------
        List[Tuple[float, float]]
            A list of (time, step_size) tuples in the order they were recorded.
        """
        if not (0 <= extruder_id < self.num_extruders):
            raise IndexError(f"extruder_id {extruder_id} is out of range [0, {self.num_extruders})")

        return np.array(self._data[extruder_id]["step_sizes"])
    
    def get_all_steps(self,) -> List:
        """
        Retrieve the full list of (time, step_size) for the given extruder.

        Returns
        -------
        List[Tuple[float, float]]
            A list of (time, step_size) tuples in the order they were recorded.
        """
        ret = np.array([])
        for id in range(self.num_extruders):
            ret = np.concatenate((ret, self._data[id]["step_sizes"]), axis=0)
            
        return ret
    
    def get_all_bound_times(self,) -> List:
        """
        Retrieve the full list of (time, step_size) for the given extruder.

        Returns
        -------
        List[Tuple[float, float]]
            A list of (time, step_size) tuples in the order they were recorded.
        """
        ret = np.array([])
        for id in range(self.num_extruders):
            ret = np.concatenate((ret, self._data[id]["bound_times"]), axis=0)
            
        return ret

    def get_extrusion_history(self) -> Dict[str, Any]:
        """
        Retrieve the full list of (time, step_size) for the given extruder.

        Returns
        -------
        List[Tuple[float, float]]
            A list of (time, step_size) tuples in the order they were recorded.
        """
        
        return self._traj

    def get_extruder_history(self, extruder_id: int) -> List[Tuple[float, float]]:
        """
        Retrieve the full list of (time, step_size) for the given extruder.

        Returns
        -------
        List[Tuple[float, float]]
            A list of (time, step_size) tuples in the order they were recorded.
        """
        if not (0 <= extruder_id < self.num_extruders):
            raise IndexError(f"extruder_id {extruder_id} is out of range [0, {self.num_extruders})")

        times = self._data[extruder_id]["times"]
        steps = self._data[extruder_id]["step_sizes"]
        return list(zip(times, steps))

    def get_all_histories(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Retrieve data for all extruders at once.

        Returns
        -------
        Dict[int, List[Tuple[float, float]]]
            Mapping extruder_id → list of (time, step_size).
        """
        return {eid: self.get_extruder_history(eid) for eid in range(self.num_extruders)}

    def clear(self) -> None:
        """
        Clear all stored histories for every extruder.
        """
        for eid in range(self.num_extruders):
            self._data[eid]["times"].clear()
            self._data[eid]["step_sizes"].clear()
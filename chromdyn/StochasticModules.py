import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Callable, Union
from pathlib import Path
from scipy.spatial import distance
from Utilities import LogManager

def default_rng(seed):
    if seed == -1:
        seed = np.random.randint(100_000_000)
    return np.random.default_rng(seed), seed


class LoopExtruder:
    """
    Represents a single loop extruder with its own state and rate calculations.
    """
    valid_pausing_rules = ('both_blocked', 'forward_blocked')

    def __init__(
        self,
        id: int,
        lattice: Any,
        rng: np.random.Generator,
        k_off: float,
        k_hop: float,
        tau: float,
        v: float = 3.0,
        k_step: float = 1.0,
        t_mesh: float = 0.001,
        temp: float = 0.0,
        unbind_at_end: bool = True,
        end_unbinding_factor: float = 1e5,
        preferential_loading: float = 0.0,
        max_hop_len: int = 4,
        drag: float = 1.0,
        one_sided: bool = True,
        f_stall: float = 100.0,
    ):
        self.id = id
        self.k_off = k_off
        self.k_step = k_step
        self.k_hop = k_hop
        self.v = v
        self.tau = tau
        self.t_mesh = t_mesh
        self.temp = temp
        self.drag = drag
        self.unbind_at_end = unbind_at_end
        self.end_unbinding_factor = end_unbinding_factor
        self.lattice = lattice  # reference to simulation for blockers
        self.preferential_loading = preferential_loading
        self.max_hop_len = max_hop_len
        self.f_stall = f_stall
        self.polymer_feedback_force = 0.0
        f_extrude = self.v * self.drag
        self.extrusion_force = [f_extrude, f_extrude]
        self.rng = rng
        self.active_anchor = [True, True]
        if one_sided==True:
            inactive_anchor = self.rng.choice([0,1])
            self.active_anchor[inactive_anchor] = False
            
        # CTCF
        self.CTCF_rule: Optional[bool] = True
        self.ctcf_hop_factor: Optional[float] = 0.01
        self.ctcf_off_rate_factor: Optional[float] = 0.01

        # dynamic state
        self.left_anchor: float = -1
        self.right_anchor: float = -1
        self.bound_time: float = 0.0
        self.pausing_rule: str = 'forward_blocked'
        self.left_anchor_blockers: tuple = ({},{})
        self.right_anchor_blockers: tuple = ({},{})
        self.is_left_paused: bool = False
        self.is_right_paused: bool = False
        self.off_rate_factor: float = 1.0
        self.hop_left_factor: float = 1.0
        self.hop_right_factor: float = 1.0
        self.step_left_factor: float = 1.0
        self.step_right_factor: float = 1.0
        
        self.data: Dict[str, List[Any]] = {
            "bound_times":[],
            "step_sizes": []
        }
    
    def get_param_names(self) -> Tuple[str]:
        return ('id', 'k_off', 'k_step', 'k_hop', 'v', 'tau', 't_mesh',
        'temp', 'drag', 'unbind_at_end', 'end_unbinding_factor', 'preferential_loading', 'max_hop_len',
        'f_stall', 'polymer_feedback_force', 'extrusion_force', 'active_anchor', 
        'CTCF_rule', 'ctcf_hop_factor', 'ctcf_off_rate_factor',
        'left_anchor', 'right_anchor', 'bound_time',
        'pausing_rule', 'left_anchor_blockers', 'right_anchor_blockers',
        'is_left_paused', 'is_right_paused', 'off_rate_factor', 'hop_left_factor', 'hop_right_factor',
        'step_left_factor', 'step_right_factor')
        
    def get_params(self) -> Dict[str, Any]:
        ret_dict = {}
        for name in self.get_param_names():
            ret_dict[name] = getattr(self, name)
        return ret_dict
        
    def set_anchors(self, left: float, right: float):
        self.left_anchor = left
        self.right_anchor = right
    
    def get_anchors(self) -> Tuple[float, float]:
        return (self.left_anchor, self.right_anchor)

    def set_pausing_rule(self, rule: str):
        if rule not in self.valid_pausing_rules:
            raise ValueError(f"Invalid pausing rule: {rule}")
        self.pausing_rule = rule

    def set_CTCF_params(self, hop_factor: float, off_factor: float):
        self.CTCF_rule = True
        self.ctcf_hop_factor = hop_factor
        self.ctcf_off_rate_factor = off_factor

    def get_event_rates(self) -> List[Tuple[str, float]]:
        """
        Return list of (event_type, rate) tuples for this extruder,
        applying factors from blockers and CTCF rules.
        """
        self._update_blocking_factors()
        rates = []
        # unbinding
        rates.append(("unbind", self.off_rate_factor * self.k_off))
        # step left
        r_sl = (not self.is_left_paused) * self.step_left_factor * self.k_step * self.active_anchor[0]
        rates.append(("step_left", r_sl))
        # step right
        r_sr = (not self.is_right_paused) * self.step_right_factor * self.k_step * self.active_anchor[1]
        rates.append(("step_right", r_sr))
        # hop left
        r_hl = self.is_left_paused * self.hop_left_factor * self.k_hop * self.active_anchor[0]
        rates.append(("hop_left", r_hl))
        # hop right
        r_hr = self.is_right_paused * self.hop_right_factor * self.k_hop * self.active_anchor[1]
        rates.append(("hop_right", r_hr))
        return rates

    def _update_blocking_factors(self):
        """
        Check simulation blockers at anchor neighbors,
        update pause flags and rate factors including CTCF.
        """
        left_anc_blocked_on_left = self.lattice.what_occupies(np.ceil(self.left_anchor) - 1)
        left_anc_blocked_on_right = self.lattice.what_occupies(np.ceil(self.left_anchor) + 1)
        right_anc_blocked_on_left = self.lattice.what_occupies(np.ceil(self.right_anchor) - 1)
        right_anc_blocked_on_right = self.lattice.what_occupies(np.ceil(self.right_anchor) + 1)

        setattr(self, "left_anchor_blockers" , (left_anc_blocked_on_left, left_anc_blocked_on_right))
        setattr(self, "right_anchor_blockers" , (right_anc_blocked_on_left, right_anc_blocked_on_right))
        
        # pausing logic
        if self.pausing_rule == 'both_blocked':
            self.is_left_paused = (left_anc_blocked_on_left['name'] != 'None' and left_anc_blocked_on_right['name'] != 'None')
            self.is_right_paused = (right_anc_blocked_on_left['name'] != 'None' and right_anc_blocked_on_right['name'] != 'None')
        elif self.pausing_rule == 'forward_blocked':  # forward_blocked
            self.is_left_paused = (left_anc_blocked_on_left['name'] != 'None')
            self.is_right_paused = (right_anc_blocked_on_right['name'] != 'None')

        # reset factors
        self.off_rate_factor = 1.0
        self.hop_left_factor = 1.0
        self.hop_right_factor = 1.0

        # unbind at ends
        if self.unbind_at_end:
            if left_anc_blocked_on_left['name'].lower() == 'end' or right_anc_blocked_on_right['name'].lower() == 'end':
                self.off_rate_factor = self.end_unbinding_factor

        # CTCF orientation rules
        if self.CTCF_rule:
            if left_anc_blocked_on_left['name'].lower() == 'ctcf' and left_anc_blocked_on_left['orientation'] == '-':
                self.hop_left_factor = self.ctcf_hop_factor
                self.off_rate_factor = self.ctcf_off_rate_factor
                self.is_left_paused = True
            if right_anc_blocked_on_right['name'].lower() == 'ctcf' and right_anc_blocked_on_right['orientation'] == '+':
                self.hop_right_factor = self.ctcf_hop_factor
                self.off_rate_factor = self.ctcf_off_rate_factor
                self.is_right_paused = True

    def perform_event(self, event_type: str, dt: float):
        """
        Update state based on event type.
        """
        if event_type == "unbind":
            
            # preferential or random
            if self.lattice.rng.uniform() < self.preferential_loading:
                left, right = self.lattice.find_preferential_loading_site()
                if left == -1 or right == -1:
                    left, right = self.lattice.find_random_binding_site()
            else:
                left, right = self.lattice.find_random_binding_site()
            old_left, old_right = self.get_anchors()
            
            self.set_anchors(left, right)
            self.lattice.update_blockers(new=int(np.ceil(left)), old=int(np.ceil(old_left)))
            self.lattice.update_blockers(new=int(np.ceil(right)), old=int(np.ceil(old_right)))
            self.data['bound_times'].append(self.bound_time)
            self.bound_time = 0.0
            
        else:
            # move anchors with roadblock checks
            if event_type == "step_left":
                dx , f_t = self._get_traversal_distance(dt, self.extrusion_force[0])
                self._move_anchor('left', -dx)
                self.extrusion_force[0] = f_t
                self.data['step_sizes'].append(-dx)
                
            elif event_type == "step_right":
                dx , f_t = self._get_traversal_distance(dt, self.extrusion_force[1])
                self._move_anchor('right', dx)
                self.extrusion_force[1] = f_t
                self.data['step_sizes'].append(dx)
                
            elif event_type == "hop_left":
                self._hop_anchor('left')
            elif event_type == "hop_right":
                self._hop_anchor('right')
            self.bound_time += dt
    
    def _get_traversal_distance(self, dt: float, f_t_minus_dt:float) -> Tuple[float, float]:
        t_mesh=self.t_mesh
        if t_mesh>dt/5.0: t_mesh = dt/5.0
        f0 = self.v * self.drag
        polymer_feedback = self.get_polymer_feedback_factor()
        Delta_x = 0.0
        for t in  np.arange(t_mesh, dt, t_mesh):
            f_t = f_t_minus_dt * np.exp(-t_mesh/self.tau) + f0 * np.sqrt(1-np.exp(-2 * t_mesh / self.tau)) * self.rng.normal(loc=0, scale=1.0)
            f_t *= polymer_feedback
            Delta_x += (t_mesh * f_t/self.drag) + np.sqrt(2*self.temp*t_mesh/self.drag) * self.rng.normal(loc=0, scale=1.0)
            f_t_minus_dt = f_t
            # print(f'traverse_LE: dx={Delta_x} | force={f_t} | fres={polymer_feedback}, dt={dt}, t_mesh={t_mesh}')
        return (Delta_x, f_t)
    
    def get_polymer_feedback_factor(self, mu=0.1):
        factor = 0.5 * (1.0 + np.tanh(mu * (self.f_stall - self.polymer_feedback_force)))
        return factor
 
    def _move_anchor(self, side: str, delta: float):
        pos = getattr(self, f"{side}_anchor")
        target = pos + delta
        new_pos, roadblock = self.lattice.check_for_roadblocks(pos, target)
        setattr(self, f"{side}_anchor", new_pos)
        self.lattice.update_blockers(new=int(np.ceil(new_pos)), old=int(np.ceil(pos)))
        

    def _hop_anchor(self, side: str):
        pos = getattr(self, f"{side}_anchor")
        hopped = False
        for n in range(2, self.max_hop_len+1):
            candidate = pos + (n if side=='right' else -n)
            if self.lattice.free_for_binding(candidate):
                setattr(self, f"{side}_anchor", candidate)
                hopped = True
                break
        if hopped:
            self.lattice.update_blockers(new=int(np.ceil(candidate)), old=int(np.ceil(pos)))

    def update_bound_time(self, dt: float):
        self.bound_time += dt
        

class Polymer1D:
    def __init__(
        self,
        topology: List[Tuple[int,int]],
        rng:  np.random.Generator,
        sim: Any,
    ):
        self.topology = topology
        self.rng = rng
        self.sim = sim
        self.first_mono, self.last_mono = topology[0][0], topology[-1][1]
        self.len = self.last_mono
        # blockers
        self.mobile_blockers: List[int] = []
        self.immobile_blockers: Dict[int, Dict[str,str]] = {}
        for start,end in topology:
            self.immobile_blockers[start] = {'type':'End','orientation':'left'}
            self.immobile_blockers[end] = {'type':'End','orientation':'right'}

    def initialize_blocker(self, left:float, right:float):
            self.mobile_blockers.extend([int(np.ceil(left)),int(np.ceil(right))])
    
    def update_blockers(self, new: float, old: float):
        self.mobile_blockers.remove(int(np.ceil(old)))
        self.mobile_blockers.append(int(np.ceil(new)))
        
    def remove_blockers(self, left: float, right:float):
        self.mobile_blockers.remove(int(np.ceil(left)))
        self.mobile_blockers.remove(int(np.ceil(right)))
            
    def find_random_binding_site(self) -> Tuple[int,int]:
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
    
    def find_preferential_loading_site(self) -> Tuple[int,int]:
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
    
    def free_for_binding(self, i: float) -> bool:
        i = int(np.ceil(i))
        return self.first_mono < i < self.last_mono and i not in self.mobile_blockers and i not in self.immobile_blockers

    def what_occupies(self, mono: int) -> Dict[str,str]:
        if mono in self.immobile_blockers:
            blocker = self.immobile_blockers[mono]
            return {'name':blocker['type'],'orientation':blocker['orientation']}
        if mono in self.mobile_blockers:
            for le in self.sim.extruders:
                if le.left_anchor == mono:
                    return {'name':f'LE{le.id}','orientation':'left'}
                if le.right_anchor == mono:
                    return {'name':f'LE{le.id}','orientation':'right'}
        return {'name':'None','orientation':'None'}

    def check_for_roadblocks(self, xi: float, xf: float) -> Tuple[float, int]:
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
                    roadblock = int(np.ceil(xx+dx))
                    #check legality
                    # assert self.is_site_within_chain(roadblock), "Roadblock outside the polymer!!"         
                    break
        
        #if no roadblocks extrude to xf
        if roadblock==-1:
            x_out = xf
        # otherwise extrude according to extrude_len
        else:
            x_out = xi + extrude_len
        return (x_out, roadblock)


class StochasticExtrusion:
    def __init__(
        self,
        topology: List[Tuple[int,int]],
        name: str = 'SMC',
        console_stream: bool = True,
        rng_seed: int = -1,
    ):
        self.name = name
        self.output_dir = Path('output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = LogManager(log_file=self.output_dir / f"{self.name}.log").get_logger(__name__, console=console_stream)
        self.rng, seed = default_rng(rng_seed)
        self.logger.info(f"Random seed: {seed}")
        self.traj: Dict[str, Any] = {'time': [], 
                                     'anchors' : []}
        
        self.lattice = Polymer1D(topology=topology, rng=self.rng, sim=self)
        self.extruders: List[LoopExtruder] = []
        self.time = 0.0
        self.dt = 0.0
    
    def add_extruder(self, id: int, k_off: float = 2e-3, k_hop: float = 1e-2, tau: float = 1e4, **kwargs):
        assert id not in [le.id for le in self.extruders], f"Reuested ID {id} already exists!"
        # init extruders
        le = LoopExtruder(id=id, k_off=k_off, k_hop=k_hop, tau=tau, lattice=self.lattice, rng=self.rng, **kwargs)
        left,right = self.lattice.find_random_binding_site()
        le.set_anchors(left,right)
        self.extruders.append(le)
        self.lattice.initialize_blocker(left, right)
        
    def remove_extruder(self, id: int):
        assert id in [le.id for le in self.extruders], f"Requested ID {id} does not exist!"
        for le in self.extruders:
            if le.id == id:
                left, right = le.get_anchors()
                self.extruders.remove(le)
                self.lattice.remove_blockers(left, right)

    def add_ctcf(self, loc: List[int], orientations: List[str]):
        assert len(loc)==len(orientations), 'shape mismatch between ctcf loc and orientation lists'
        
        for site, orient in zip(loc, orientations):
            assert (orient=='+' or orient=='-'), 'Orientations can only be + or - !'
            self.lattice.immobile_blockers[int(site)] = {'type':'ctcf','orientation':f'{orient}'}
        
    def create_rate_vector(self) -> Tuple[List[Tuple[LoopExtruder,str]], np.ndarray]:
        events,rates = [],[]
        for le in self.extruders:
            for ev,rate in le.get_event_rates():
                events.append((le,ev))
                rates.append(rate)
        return events,np.array(rates)

    def simulate_step(self) -> bool:
        events,rates = self.create_rate_vector()
        total = rates.sum()
        if total <= 0:
            self.logger.warning("No events possible; ending simulation.")
            return (None, None, None)
        dt = self.rng.exponential(1.0/total)
        # print('dt', dt)
        idx = self.rng.choice(len(events), p=rates/total)
        le,ev = events[idx]
        le.perform_event(ev, dt)
        self.time += dt
        for other in self.extruders:
            if other is not le:
                other.update_bound_time(dt)
        self.record_state()
        # self.stats.record_state(extruder_id = le.id, dt=dt, time=self.time, step_size=self.current_LE_step, extruder_dict=self.loop_extruder_dict)
        return (le.id, ev, dt)

    def run(self, tstop: float):
        while self.time < tstop and self.simulate_step():
            pass
    
    def record_state(self):
        anchors = [le.get_anchors() for le in self.extruders]
        self.traj['time'].append(self.time)
        self.traj['anchors'].append(anchors)

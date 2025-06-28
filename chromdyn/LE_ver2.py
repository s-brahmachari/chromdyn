import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Callable
from pathlib import Path
from scipy.spatial import distance
from Utilities import LogManager


def default_rng():
    seed = np.random.randint(100_000_000)
    return np.random.default_rng(seed), seed

class LoopExtruder:
    """
    Represents a single loop extruder with its own state and rate calculations.
    """
    valid_pausing_rules = ('both_blocked', 'forward_blocked')

    def __init__(
        self,
        extruder_id: int,
        k_off: float,
        k_step: float,
        k_hop: float,
        v: float,
        tau: float,
        t_mesh: float,
        temp: float = 0.0,
        unbind_at_end: bool = True,
        end_unbinding_factor: float = 1e5,
        sim: Any = None,
    ):
        self.id = extruder_id
        self.k_off = k_off
        self.k_step = k_step
        self.k_hop = k_hop
        self.v = v
        self.tau = tau
        self.t_mesh = t_mesh
        self.temp = temp
        self.unbind_at_end = unbind_at_end
        self.end_unbinding_factor = end_unbinding_factor
        self.sim = sim  # reference to simulation for blockers

        # CTCF
        self.CTCF_rule: Optional[bool] = None
        self.ctcf_hop_factor: Optional[float] = None
        self.ctcf_off_rate_factor: Optional[float] = None

        # dynamic state
        self.left_anchor: int = -1
        self.right_anchor: int = -1
        self.bound_time: float = 0.0
        self.pausing_rule: str = 'forward_blocked'
        self.is_left_paused: bool = False
        self.is_right_paused: bool = False
        self.off_rate_factor: float = 1.0
        self.hop_left_factor: float = 1.0
        self.hop_right_factor: float = 1.0
        self.step_left_factor: float = 1.0
        self.step_right_factor: float = 1.0

    def set_anchors(self, left: int, right: int):
        self.left_anchor = left
        self.right_anchor = right

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
        r_sl = (not self.is_left_paused) * self.step_left_factor * self.k_step
        rates.append(("step_left", r_sl))
        # step right
        r_sr = (not self.is_right_paused) * self.step_right_factor * self.k_step
        rates.append(("step_right", r_sr))
        # hop left
        r_hl = self.is_left_paused * self.hop_left_factor * self.k_hop
        rates.append(("hop_left", r_hl))
        # hop right
        r_hr = self.is_right_paused * self.hop_right_factor * self.k_hop
        rates.append(("hop_right", r_hr))
        return rates

    def _update_blocking_factors(self):
        """
        Check simulation blockers at anchor neighbors,
        update pause flags and rate factors including CTCF.
        """
        left_b_left = self.sim.what_occupies(self.left_anchor - 1)
        left_b_right = self.sim.what_occupies(self.left_anchor + 1)
        right_b_left = self.sim.what_occupies(self.right_anchor - 1)
        right_b_right = self.sim.what_occupies(self.right_anchor + 1)

        # pausing logic
        if self.pausing_rule == 'both_blocked':
            self.is_left_paused = (left_b_left['name'] != 'None' and left_b_right['name'] != 'None')
            self.is_right_paused = (right_b_left['name'] != 'None' and right_b_right['name'] != 'None')
        else:  # forward_blocked
            self.is_left_paused = (left_b_left['name'] != 'None')
            self.is_right_paused = (right_b_right['name'] != 'None')

        # reset factors
        self.off_rate_factor = 1.0
        self.hop_left_factor = 1.0
        self.hop_right_factor = 1.0

        # unbind at ends
        if self.unbind_at_end:
            if left_b_left['name'].lower() == 'end' or right_b_right['name'].lower() == 'end':
                self.off_rate_factor = self.end_unbinding_factor

        # CTCF orientation rules
        if self.CTCF_rule:
            if left_b_left['name'].lower() == 'ctcf' and left_b_left['orientation'] == '-':
                self.hop_left_factor = self.ctcf_hop_factor
                self.off_rate_factor = self.ctcf_off_rate_factor
                self.is_left_paused = True
            if right_b_right['name'].lower() == 'ctcf' and right_b_right['orientation'] == '+':
                self.hop_right_factor = self.ctcf_hop_factor
                self.off_rate_factor = self.ctcf_off_rate_factor
                self.is_right_paused = True

    def perform_event(self, event_type: str, dt: float, rng: np.random.Generator,
                      free_for_binding: Callable[[int], bool], find_preferential: Callable[[], Tuple[int,int]],
                      find_random: Callable[[], Tuple[int,int]]):
        """
        Update state based on event type.
        """
        if event_type == "unbind":
            self.bound_time = 0.0
            # preferential or random
            if rng.uniform() < self.sim.preferential_loading:
                left, right = find_preferential()
            else:
                left, right = find_random()
            self.set_anchors(left, right)
        else:
            # move anchors with roadblock checks
            if event_type == "step_left":
                self._move_anchor('left', -1)
            elif event_type == "step_right":
                self._move_anchor('right', +1)
            elif event_type == "hop_left":
                self._hop_anchor('left')
            elif event_type == "hop_right":
                self._hop_anchor('right')
            self.bound_time += dt

    def _move_anchor(self, side: str, delta: int):
        pos = getattr(self, f"{side}_anchor")
        target = pos + delta
        new_pos, road = self.sim.check_for_roadblocks(pos, target)
        setattr(self, f"{side}_anchor", new_pos)

    def _hop_anchor(self, side: str):
        pos = getattr(self, f"{side}_anchor")
        for n in range(2, self.sim.max_hop_len+1):
            candidate = pos + (n if side=='right' else -n)
            if self.sim.free_for_binding(candidate):
                setattr(self, f"{side}_anchor", candidate)
                break

    def update_bound_time(self, dt: float):
        self.bound_time += dt


class LoopExtrudersSimulation:
    def __init__(
        self,
        num_LE: int,
        topology: List[Tuple[int,int]],
        name: str = 'SMC',
        k_off: float = 2e-3,
        k_step: float = 0.1,
        k_hop: float = 1e-2,
        v_LE: float = 5.0,
        tau: float = 1e6,
        t_mesh: float = 0.001,
        temp: float = 0.0,
        max_hop_len: int = 5,
        preferential_loading: float = 0.0,
        loading_sites: Optional[List[int]] = None,
        console_stream: bool = True,
    ):
        self.num_LE = num_LE
        self.topology = topology
        self.name = name
        self.output_dir = Path('output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = LogManager(log_file=self.output_dir / f"{self.name}.log").get_logger(__name__, console=console_stream)
        self.rng, seed = default_rng()
        self.logger.info(f"Random seed: {seed}")

        self.first_mono, self.last_mono = topology[0][0], topology[-1][1]
        self.max_hop_len = max_hop_len

        # loading
        self.preferential_loading = preferential_loading
        self.loading_sites = loading_sites or []

        # blockers
        self.mobile_blockers: List[int] = []
        self.immobile_blockers: Dict[int, Dict[str,str]] = {}
        for start,end in topology:
            self.immobile_blockers[start] = {'type':'End','orientation':'left'}
            self.immobile_blockers[end] = {'type':'End','orientation':'right'}

        # init extruders
        self.extruders: List[LoopExtruder] = []
        for i in range(num_LE):
            le = LoopExtruder(i, k_off, k_step, k_hop, v_LE, tau, t_mesh, temp, sim=self)
            self.extruders.append(le)
        self.initialize_blockers()

        self.time = 0.0
        self.dt = 0.0
        self.stats = []

    def initialize_blockers(self):
        for le in self.extruders:
            left,right = self.find_random_binding_site()
            le.set_anchors(left,right)
            self.mobile_blockers.extend([left,right])

    def free_for_binding(self, i: int) -> bool:
        return self.first_mono < i < self.last_mono and i not in self.mobile_blockers and i not in self.immobile_blockers

    def what_occupies(self, mono: int) -> Dict[str,str]:
        if mono in self.immobile_blockers:
            d = self.immobile_blockers[mono]
            return {'name':d['type'],'orientation':d['orientation']}
        if mono in self.mobile_blockers:
            for le in self.extruders:
                if le.left_anchor == mono:
                    return {'name':f'LE{le.id}','orientation':'left'}
                if le.right_anchor == mono:
                    return {'name':f'LE{le.id}','orientation':'right'}
        return {'name':'None','orientation':'None'}

    def check_for_roadblocks(self, xi: int, xf: int) -> Tuple[int,int]:
        dx = np.sign(xf-xi)
        road = -1
        for pos in range(xi, xf, dx):
            if not self.free_for_binding(pos+dx):
                road = pos+dx
                return xi + (pos-xi), road
        return xf, road

    def find_random_binding_site(self) -> Tuple[int,int]:
        while True:
            anchor = self.rng.integers(self.first_mono+1, self.last_mono)
            if self.free_for_binding(anchor) and self.free_for_binding(anchor+1):
                return anchor, anchor+1

    def find_preferential_loading_site(self) -> Tuple[int,int]:
        search_dist = 5
        for _ in range(100):
            base = self.rng.choice(self.loading_sites)
            offset = int(np.sign(self.rng.normal()) * self.rng.integers(1,search_dist))
            pos = base + offset
            if self.free_for_binding(pos) and self.free_for_binding(pos+1):
                return pos,pos+1
            search_dist += 1
        return -1,-1

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
            return False
        dt = self.rng.exponential(1.0/total)
        idx = self.rng.choice(len(events), p=rates/total)
        le,ev = events[idx]
        le.perform_event(ev, dt, self.rng,
                         free_for_binding=self.free_for_binding,
                         find_preferential=self.find_preferential_loading_site,
                         find_random=self.find_random_binding_site)
        self.time += dt
        for other in self.extruders:
            if other is not le:
                other.update_bound_time(dt)
        self.stats.append((self.time,le.id,ev))
        return True

    def run(self, tstop: float):
        while self.time < tstop and self.simulate_step():
            pass

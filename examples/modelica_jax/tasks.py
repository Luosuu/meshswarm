"""Three-tier Phase 1 Modelica-to-JAX benchmark suite.

Every task starts from an acausal/hybrid Modelica (.mo) model that compiles and
simulates cleanly in OpenModelica (`omc`, DASSL tolerance=1e-10), but FAILS
direct `Rumoca.to_jax()` export due to:
  - Tier 1 (`tier1_blanket_dae`): event-inducing check-valve / orifice branches
    (`if/else`) AND an implicit nonlinear Sieverts-permeation algebraic equation
    (`0 = g(z, x, p)`).
  - Tier 2 (`tier2_transform_hx`): multi-package inheritance from Modelica 4.0.0
    and ORNL `TRANSFORM-Library` (`Simple_HX` counter-current PbLi/FLiBe heat
    exchanger with `cv != cp` internal-energy vs enthalpy semantics, coupled to
    Arrhenius wall tritium permeation).
  - Tier 3 (`tier3_hdt_column`): 16-stage closed-recycle cryogenic H2/HD/T2
    isotope distillation MESH column (`dim = 32` differential composition states +
    16 implicit nonlinear Antoine bubble-point equations + flooding transition),
    where exploiting block-tridiagonal structure beats dense per-thread LU on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

DELIVERABLE = "solver_jax.py"
PROTECT = ["solver_jax.py"]
KIND = (
    "refactoring engineering Modelica (.mo) differential-algebraic models into "
    "conservation-preserving, differentiable JAX GPU ensemble solvers"
)

# Combined wall-time rungs (seconds) for 1,000-trajectory forward GPU solve + 64-trajectoryjax.grad pass.
THRESHOLDS_S = [30.0, 15.0, 8.0, 4.0, 2.0, 1.0, 0.50, 0.25, 0.12, 0.06]


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    title: str
    about: str
    model_file: str
    model_name: str
    model_source: str
    needs_transform: bool
    t_end: float
    n_states: int
    param_names: list[str]
    obs_names: list[str]
    y0_default: list[float]
    p_default: list[float]
    dev_cases: list[list[float]]
    test_cases: list[list[float]]
    ensemble_bounds: tuple[list[float], list[float]]
    traj_rtol: float
    traj_atol: float
    conservation_kind: str  # "sum_last3" | "hdt_column16"


# ---------------------------------------------------------------------------
# Tier 1: Stiff Blanket Coolant Check-Valve + Implicit Sieverts DAE
# ---------------------------------------------------------------------------
TIER1_MO = """\
model BlanketValveDAE
  parameter Real u_pump = 1.0 "Normalized primary pump head command";
  parameter Real u_ext = 0.70 "Normalized vacuum permeator extraction drive";
  parameter Real P_pulse = 1.0 "Fusion neutron heating modulation amplitude";
  parameter Real dp_back = 0.35 "Back-pressure disturbance amplitude";

  constant Real tau_hyd = 0.02 "Fast hydraulic inertia time scale (s)";
  constant Real k_fwd = 0.24 "Forward turbulent loop resistance";
  constant Real k_rev = 14.0 "Check-valve reverse-flow resistance";
  constant Real k_byp = 0.08 "Orifice bypass conductance";
  constant Real C_blanket = 26.0 "Blanket heat capacity (MJ/K)";
  constant Real C_hx = 11.0 "Secondary HX heat capacity (MJ/K)";
  constant Real cp_cool = 1.85 "Effective coolant heat capacity (MJ/(kg_eq*K))";
  constant Real h_sink = 0.90 "Power-conversion sink conductance (MW/K)";
  constant Real T_sink = 570.0 "Power-conversion sink temperature (K)";
  constant Real P_base = 115.0 "Nominal fusion blanket nuclear heating (MW)";
  constant Real k_perm = 0.042 "Interfacial tritium transfer coefficient (1/s)";
  constant Real alpha_nl = 0.65 "Quadratic Sieverts transport resistance";
  constant Real beta_nl = 0.30 "Cubic high-flux permeation saturation";
  constant Real k_regen = 0.050 "Getter regeneration transfer rate (1/s)";

  Real q_flow(start = 1.6, fixed = true);
  Real T_blanket(start = 755.0, fixed = true);
  Real T_hx(start = 635.0, fixed = true);
  Real N_blanket(start = 4.2, fixed = true);
  Real N_extractor(start = 0.4, fixed = true);
  Real N_storage(start = 0.4, fixed = true);

  Real P_fus;
  Real dp_drive;
  Real drag_valve;
  Real q_byp;
  Real q_eff;
  Real z_flux(start = 0.12);
  Real j_store;

  Real obs_q_flow;
  Real obs_T_blanket;
  Real obs_T_hx;
  Real obs_N_blanket;
  Real obs_N_extractor;
  Real obs_N_storage;
equation
  P_fus = P_base * (1.0 + 0.40 * P_pulse * sin(0.20 * time));
  dp_drive = 2.0 * u_pump - dp_back * (1.0 + 0.25 * cos(0.35 * time));

  drag_valve = if q_flow >= 0.0 then k_fwd * q_flow * q_flow else -k_rev * q_flow * q_flow;
  q_byp = if dp_drive >= 0.0 then k_byp * sqrt(dp_drive + 1.0e-8) else -0.15 * k_byp * sqrt(-dp_drive + 1.0e-8);
  der(q_flow) = (dp_drive - drag_valve - 0.30 * q_flow) / tau_hyd;

  q_eff = (if q_flow >= 0.0 then q_flow else -0.10 * q_flow) + (if q_byp >= 0.0 then q_byp else 0.0);
  der(T_blanket) = (P_fus - q_eff * cp_cool * (T_blanket - T_hx)) / C_blanket;
  der(T_hx) = (q_eff * cp_cool * (T_blanket - T_hx) - h_sink * (T_hx - T_sink)) / C_hx;

  0 = z_flux + alpha_nl * z_flux * abs(z_flux) + beta_nl * z_flux * z_flux * z_flux - u_ext * k_perm * N_blanket;
  j_store = k_regen * N_extractor;

  der(N_blanket) = -z_flux;
  der(N_extractor) = z_flux - j_store;
  der(N_storage) = j_store;

  obs_q_flow = q_flow;
  obs_T_blanket = T_blanket;
  obs_T_hx = T_hx;
  obs_N_blanket = N_blanket;
  obs_N_extractor = N_extractor;
  obs_N_storage = N_storage;
end BlanketValveDAE;
"""


# ---------------------------------------------------------------------------
# Tier 2: ORNL TRANSFORM Counter-Current PbLi-FLiBe HX + Arrhenius Permeation
# ---------------------------------------------------------------------------
TIER2_MO = """\
model TransformPbLiFLiBeHX
  inner TRANSFORM.Fluid.SystemTF systemTF;
  package Medium_PbLi = TRANSFORM.Media.Fluids.PbLi.ConstantPropertyLiquidPbLi;
  package Medium_FLiBe = TRANSFORM.Media.Fluids.FLiBe.ConstantPropertyLiquidFLiBe;

  parameter Real m_flow_pbli = 90.0 "Primary liquid PbLi mass flow command (kg/s)";
  parameter Real m_flow_flibe = 12.0 "Secondary molten FLiBe mass flow command (kg/s)";
  parameter Real T_in_pbli = 955.0 "Blanket outlet PbLi temperature baseline (K)";
  parameter Real u_vac = 0.80 "Vacuum permeator extraction command";

  TRANSFORM.Fluid.BoundaryConditions.MassFlowSource_T src_pbli(
    redeclare package Medium = Medium_PbLi, use_m_flow_in=true, use_T_in=true, nPorts=1);
  TRANSFORM.Fluid.BoundaryConditions.Boundary_pT snk_pbli(
    redeclare package Medium = Medium_PbLi, p=2.0e5, T=900.0, nPorts=1);
  TRANSFORM.Fluid.BoundaryConditions.MassFlowSource_T src_flibe(
    redeclare package Medium = Medium_FLiBe, use_m_flow_in=true, T=865.0, nPorts=1);
  TRANSFORM.Fluid.BoundaryConditions.Boundary_pT snk_flibe(
    redeclare package Medium = Medium_FLiBe, p=2.0e5, T=900.0, nPorts=1);

  TRANSFORM.HeatExchangers.Simple_HX hx(
    redeclare package Medium_1 = Medium_PbLi,
    redeclare package Medium_2 = Medium_FLiBe,
    nV=6, counterCurrent=true,
    V_1=0.12, V_2=0.12, UA=48000.0,
    p_a_start_1=2.06e5, p_b_start_1=2.0e5,
    p_a_start_2=2.06e5, p_b_start_2=2.0e5,
    m_flow_start_1=90.0, m_flow_start_2=12.0,
    T_a_start_1=930.0, T_b_start_1=930.0,
    T_a_start_2=880.0, T_b_start_2=880.0,
    energyDynamics_1=Modelica.Fluid.Types.Dynamics.FixedInitial,
    energyDynamics_2=Modelica.Fluid.Types.Dynamics.FixedInitial);

  Real N_loop(start=5.0, fixed=true);
  Real N_perm(start=0.5, fixed=true);
  Real N_getter(start=0.5, fixed=true);
  Real k_arr[6];
  Real k_bar;
  Real j_trit(start=0.05);

  Real obs_Q_MW;
  Real obs_T1_out;
  Real obs_T2_out;
  Real obs_N_loop;
  Real obs_N_perm;
  Real obs_N_getter;
equation
  src_pbli.m_flow_in = if m_flow_pbli >= 15.0 then m_flow_pbli else 15.0 + 0.15 * (m_flow_pbli - 15.0);
  src_pbli.T_in = T_in_pbli + 18.0 * sin(0.20 * time);
  src_flibe.m_flow_in = if m_flow_flibe >= 3.0 then m_flow_flibe else 3.0 + 0.15 * (m_flow_flibe - 3.0);

  connect(src_pbli.ports[1], hx.port_a1);
  connect(hx.port_b1, snk_pbli.ports[1]);
  connect(src_flibe.ports[1], hx.port_a2);
  connect(hx.port_b2, snk_flibe.ports[1]);

  for i in 1:6 loop
    k_arr[i] = 0.22 * exp(-3400.0 / hx.volume_1[i].medium.T);
  end for;
  k_bar = sum(k_arr) / 6.0;
  0 = j_trit + 0.45 * j_trit * abs(j_trit) + 0.25 * j_trit ^ 3 - u_vac * k_bar * N_loop;
  der(N_loop) = -j_trit;
  der(N_perm) = j_trit - 0.06 * N_perm;
  der(N_getter) = 0.06 * N_perm;

  obs_Q_MW = hx.Q_flow * 1e-6;
  obs_T1_out = hx.sensor_T_b1.T;
  obs_T2_out = hx.sensor_T_b2.T;
  obs_N_loop = N_loop;
  obs_N_perm = N_perm;
  obs_N_getter = N_getter;
end TransformPbLiFLiBeHX;
"""


# ---------------------------------------------------------------------------
# Tier 3: 16-Stage Cryogenic H2 / HD / T2 Distillation MESH DAE Column
# ---------------------------------------------------------------------------
STAGE_HOLDUPS_16: list[float] = [4.0] + [1.0] * 14 + [6.0]

TIER3_X_H2_0: list[float] = [0.55 * math.exp(-0.14 * i) for i in range(16)]
TIER3_X_HD_0: list[float] = [0.28] * 16
TIER3_Y0: list[float] = TIER3_X_H2_0 + TIER3_X_HD_0

TIER3_MO = """\
model CryoHDTColumn16
  constant Integer N = 16;
  parameter Real V_boil = 1.85 "Reboiler molar vapor boilup rate (mol/s)";
  parameter Real d_rec_frac = 0.14 "Overhead distillate-to-feed internal recycle fraction";
  parameter Real dp_tray = 0.012 "Pressure drop per tray (bar)";
  parameter Real P_top = 1.15 "Condenser top pressure (bar)";

  constant Real V_flood = 2.15 "Hydraulic flooding onset vapor rate (mol/s)";
  constant Real M[N] = {4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0};
  constant Real x_H2_init[N] = {""" + ", ".join(f"{v:.10f}" for v in TIER3_X_H2_0) + """};
  constant Real x_HD_init[N] = {""" + ", ".join(f"{v:.10f}" for v in TIER3_X_HD_0) + """};

  Real x_H2[N](start = x_H2_init, each fixed = true);
  Real x_HD[N](start = x_HD_init, each fixed = true);
  Real x_T2[N];
  Real T_stage[N](each start = 11.5);
  Real P_stage[N];
  Real Psat_H2[N];
  Real Psat_HD[N];
  Real Psat_T2[N];
  Real y_H2[N];
  Real y_HD[N];
  Real eta_tray;
  Real V_eff;
  Real D_rec;
  Real L_rect;

  Real obs_xH2_top;
  Real obs_xT2_bot;
  Real obs_T_top;
  Real obs_T_bot;
  Real obs_tot_H2;
  Real obs_tot_HD;
  Real obs_tot_T2;
equation
  V_eff = V_boil * (1.0 + 0.08 * sin(0.25 * time));
  eta_tray = if V_eff <= V_flood then 0.92 else 0.92 / (1.0 + 1.8 * (V_eff - V_flood) ^ 2);
  D_rec = d_rec_frac * V_eff;
  L_rect = V_eff - D_rec;

  for i in 1:N loop
    x_T2[i] = 1.0 - x_H2[i] - x_HD[i];
    P_stage[i] = P_top + (i - 1) * dp_tray;
    Psat_H2[i] = exp(9.45 - 106.0 / (T_stage[i] + 1.8));
    Psat_HD[i] = exp(9.62 - 129.0 / (T_stage[i] + 1.8));
    Psat_T2[i] = exp(9.85 - 156.0 / (T_stage[i] + 1.8));

    // Implicit nonlinear cryogenic bubble-point VLE equation on every stage:
    0 = x_H2[i] * Psat_H2[i] + x_HD[i] * Psat_HD[i] + x_T2[i] * Psat_T2[i] - P_stage[i];

    y_H2[i] = (1.0 - eta_tray) * x_H2[i] + eta_tray * x_H2[i] * Psat_H2[i] / P_stage[i];
    y_HD[i] = (1.0 - eta_tray) * x_HD[i] + eta_tray * x_HD[i] * Psat_HD[i] / P_stage[i];
  end for;

  // Stage 1: Total condenser
  M[1] * der(x_H2[1]) = V_eff * y_H2[2] - V_eff * x_H2[1];
  M[1] * der(x_HD[1]) = V_eff * y_HD[2] - V_eff * x_HD[1];

  // Stages 2..7: Rectifying section
  for i in 2:7 loop
    M[i] * der(x_H2[i]) = L_rect * x_H2[i - 1] + V_eff * y_H2[i + 1] - L_rect * x_H2[i] - V_eff * y_H2[i];
    M[i] * der(x_HD[i]) = L_rect * x_HD[i - 1] + V_eff * y_HD[i + 1] - L_rect * x_HD[i] - V_eff * y_HD[i];
  end for;

  // Stage 8: Recycle feed tray
  M[8] * der(x_H2[8]) = L_rect * x_H2[7] + D_rec * x_H2[1] + V_eff * y_H2[9] - V_eff * x_H2[8] - V_eff * y_H2[8];
  M[8] * der(x_HD[8]) = L_rect * x_HD[7] + D_rec * x_HD[1] + V_eff * y_HD[9] - V_eff * x_HD[8] - V_eff * y_HD[8];

  // Stages 9..15: Stripping section
  for i in 9:15 loop
    M[i] * der(x_H2[i]) = V_eff * x_H2[i - 1] + V_eff * y_H2[i + 1] - V_eff * x_H2[i] - V_eff * y_H2[i];
    M[i] * der(x_HD[i]) = V_eff * x_HD[i - 1] + V_eff * y_HD[i + 1] - V_eff * x_HD[i] - V_eff * y_HD[i];
  end for;

  // Stage 16: Reboiler
  M[16] * der(x_H2[16]) = V_eff * x_H2[15] - V_eff * y_H2[16];
  M[16] * der(x_HD[16]) = V_eff * x_HD[15] - V_eff * y_HD[16];

  obs_xH2_top = x_H2[1];
  obs_xT2_bot = x_T2[16];
  obs_T_top = T_stage[1];
  obs_T_bot = T_stage[16];
  obs_tot_H2 = sum(M[i] * x_H2[i] for i in 1:N);
  obs_tot_HD = sum(M[i] * x_HD[i] for i in 1:N);
  obs_tot_T2 = sum(M[i] * x_T2[i] for i in 1:N);
end CryoHDTColumn16;
"""


TASKS: dict[str, BenchmarkTask] = {
    "tier1_blanket_dae": BenchmarkTask(
        name="tier1_blanket_dae",
        title="a stiff fusion blanket check-valve & implicit Sieverts permeation DAE loop",
        about=(
            "The Modelica model `BlanketValveDAE.mo` couples fast primary coolant hydraulics "
            "(`tau_hyd = 0.02 s`, piecewise forward/reverse check-valve resistance and orifice bypass), "
            "blanket and heat-exchanger thermal inertia, and a closed-loop 3-reservoir tritium inventory "
            "governed by an implicit cubic-Sieverts algebraic flux equation."
        ),
        model_file="BlanketValveDAE.mo",
        model_name="BlanketValveDAE",
        model_source=TIER1_MO,
        needs_transform=False,
        t_end=20.0,
        n_states=6,
        param_names=["u_pump", "u_ext", "P_pulse", "dp_back"],
        obs_names=["obs_q_flow", "obs_T_blanket", "obs_T_hx", "obs_N_blanket", "obs_N_extractor", "obs_N_storage"],
        y0_default=[1.6, 755.0, 635.0, 4.2, 0.4, 0.4],
        p_default=[1.0, 0.70, 1.0, 0.35],
        dev_cases=[
            [1.00, 0.70, 1.00, 0.35],
            [0.12, 0.95, 1.30, 0.55],  # dips into negative dp_drive / check-valve transition
        ],
        test_cases=[
            [1.25, 0.85, 0.80, 0.25],
            [0.75, 0.50, 1.35, 0.40],
            [0.18, 1.10, 1.15, 0.52],  # crosses dp_drive = 0
            [0.05, 0.90, 0.70, 0.65],  # backflow check-valve engaged
            [1.40, 0.35, 1.45, 0.20],
            [0.55, 1.25, 0.60, 0.45],
            [0.28, 0.75, 1.20, 0.62],  # intermittent transition
            [0.95, 1.05, 0.95, 0.30],
        ],
        ensemble_bounds=([0.04, 0.25, 0.50, 0.15], [1.45, 1.30, 1.50, 0.70]),
        traj_rtol=2.5e-4,
        traj_atol=1.0e-5,
        conservation_kind="sum_last3",
    ),
    "tier2_transform_hx": BenchmarkTask(
        name="tier2_transform_hx",
        title="an ORNL TRANSFORM counter-current PbLi/FLiBe heat exchanger with Arrhenius tritium extraction",
        about=(
            "The Modelica model `TransformPbLiFLiBeHX.mo` instantiates ORNL's `TRANSFORM.HeatExchangers.Simple_HX` "
            "(`nV=6`, `counterCurrent=true`) between liquid lead-lithium (`ConstantPropertyLiquidPbLi`) and "
            "molten salt (`ConstantPropertyLiquidFLiBe`), with piecewise low-flow limiters and a 6-zone "
            "Arrhenius wall-permeation + implicit nonlinear Sieverts tritium extraction loop."
        ),
        model_file="TransformPbLiFLiBeHX.mo",
        model_name="TransformPbLiFLiBeHX",
        model_source=TIER2_MO,
        needs_transform=True,
        t_end=30.0,
        n_states=15,
        param_names=["m_flow_pbli", "m_flow_flibe", "T_in_pbli", "u_vac"],
        obs_names=["obs_Q_MW", "obs_T1_out", "obs_T2_out", "obs_N_loop", "obs_N_perm", "obs_N_getter"],
        y0_default=[930.0] * 6 + [880.0] * 6 + [5.0, 0.5, 0.5],
        p_default=[90.0, 12.0, 955.0, 0.80],
        dev_cases=[
            [90.0, 12.0, 955.0, 0.80],
            [12.0, 2.5, 940.0, 1.10],  # exercises low-flow limiter branch on both streams
        ],
        test_cases=[
            [110.0, 15.0, 965.0, 0.95],
            [65.0, 9.5, 945.0, 0.60],
            [14.0, 11.0, 950.0, 1.15],  # near primary flow-limiter knee (15 kg/s)
            [80.0, 2.6, 935.0, 0.85],   # near secondary flow-limiter knee (3 kg/s)
            [12.5, 2.4, 960.0, 1.05],   # below both knees
            [98.0, 13.5, 970.0, 0.50],
            [50.0, 7.5, 930.0, 1.20],
            [75.0, 10.5, 952.0, 0.75],
        ],
        ensemble_bounds=([11.0, 2.2, 925.0, 0.40], [115.0, 16.0, 972.0, 1.25]),
        traj_rtol=3.0e-4,
        traj_atol=1.0e-5,
        conservation_kind="sum_last3",
    ),
    "tier3_hdt_column": BenchmarkTask(
        name="tier3_hdt_column",
        title="a 16-stage cryogenic H2/HD/T2 isotope distillation MESH DAE column",
        about=(
            "The Modelica model `CryoHDTColumn16.mo` models a 16-stage closed-recycle cryogenic hydrogen "
            "isotope distillation column (`32` differential mole-fraction states + `16` implicit Antoine "
            "bubble-point algebraic equations + hydraulic flooding efficiency drop above `V_flood = 2.15 mol/s`). "
            "All three total column species inventories (H2, HD, T2) are strictly conserved."
        ),
        model_file="CryoHDTColumn16.mo",
        model_name="CryoHDTColumn16",
        model_source=TIER3_MO,
        needs_transform=False,
        t_end=25.0,
        n_states=32,
        param_names=["V_boil", "d_rec_frac", "dp_tray", "P_top"],
        obs_names=[
            "obs_xH2_top", "obs_xT2_bot", "obs_T_top", "obs_T_bot",
            "obs_tot_H2", "obs_tot_HD", "obs_tot_T2",
        ],
        y0_default=list(TIER3_Y0),
        p_default=[1.85, 0.14, 0.012, 1.15],
        dev_cases=[
            [1.85, 0.14, 0.012, 1.15],
            [2.30, 0.18, 0.015, 1.25],  # above V_flood = 2.15 flooding knee
        ],
        test_cases=[
            [1.60, 0.10, 0.010, 1.08],
            [1.95, 0.16, 0.014, 1.20],
            [2.12, 0.12, 0.012, 1.15],  # straddles V_flood during sinusoidal modulation
            [2.40, 0.20, 0.016, 1.30],  # flooded regime
            [1.45, 0.22, 0.009, 1.05],
            [2.05, 0.08, 0.013, 1.18],  # crosses V_flood
            [2.28, 0.15, 0.011, 1.12],
            [1.75, 0.19, 0.015, 1.24],
        ],
        ensemble_bounds=([1.35, 0.06, 0.008, 1.02], [2.45, 0.24, 0.018, 1.35]),
        traj_rtol=3.5e-4,
        traj_atol=1.0e-5,
        conservation_kind="hdt_column16",
    ),
}


def skeleton_code(task: BenchmarkTask) -> str:
    return f'''"""Deliverable JAX GPU ensemble solver for {task.model_name}.

Contract required by the grader:
  1. `forward_ensemble(y0_batch, p_batch) -> jax.Array`
     - `y0_batch`: float64 array of shape `(B, {task.n_states})`
     - `p_batch`:  float64 array of shape `(B, {len(task.param_names)})` with columns {task.param_names}
     - returns float64 array of shape `(B, {len(task.obs_names)})` matching {task.obs_names} at t = {task.t_end}.
  2. `differentiable_closure(y0_batch, p_ref_batch) -> Callable[[jax.Array], jax.Array]`
     - returns a pure JAX-traceable function `f(p_jax) -> obs_final_jax` of shape `(B, {len(task.obs_names)})`
       suitable for `jax.grad` around `p_ref_batch`.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def forward_ensemble(y0_batch: np.ndarray, p_batch: np.ndarray) -> jax.Array:
    raise NotImplementedError("Implement GPU-vectorized stiff forward ensemble solver")


def differentiable_closure(y0_batch: np.ndarray, p_ref_batch: np.ndarray):
    raise NotImplementedError("Return a jax.grad-differentiable function mapping p_jax -> obs_final_jax")
'''


def starting_files(name: str) -> dict[str, str]:
    t = TASKS[name]
    return {
        t.model_file: t.model_source,
        DELIVERABLE: skeleton_code(t),
    }


def task_text(name: str) -> str:
    t = TASKS[name]
    rungs = ", ".join(f"{s:g}s" for s in THRESHOLDS_S)
    omc_hint = (
        "Set `setModelicaPath(getEnvironmentVar(\"OPENMODELICALIBRARY\"))`, `loadModel(Modelica)`, and "
        "`loadFile(getEnvironmentVar(\"TRANSFORM_LIB\"))` in `.mos` scripts to inspect or simulate with `omc`."
        if t.needs_transform
        else "Use `omc` to simulate `model.mo` (`method=\"dassl\", tolerance=1e-10`) and compare your JAX outputs."
    )
    return f"""Build the fastest verified, differentiable JAX GPU ensemble solver for {t.title}.

{t.about}

Your workspace holds:
  - `{t.model_file}`: the reference Modelica model (`t in [0, {t.t_end}]`). Direct `Rumoca.to_jax()` fails on this model; you must handle its package flattening, algebraic constraints, and non-smooth branches properly.
  - `{DELIVERABLE}`: the deliverable module you must implement.

Required functions in `{DELIVERABLE}` (FP64 enabled):
  1. `forward_ensemble(y0_batch, p_batch)` -> `(B, {len(t.obs_names)})` final observables `{t.obs_names}` at `t = {t.t_end}` for initial states `(B, {t.n_states})` and parameters `(B, {len(t.param_names)})` = `{t.param_names}`.
  2. `differentiable_closure(y0_batch, p_ref_batch)` -> `Callable[[jax.Array], jax.Array]` returning a `jax.grad`-differentiable function `p_jax -> obs_final_jax` around `p_ref_batch`.

Default initial state `y0`: `{t.y0_default}`
Example parameter sets `p`: `{t.dev_cases}`

How it is graded (strict multiplicative zero-multiplier gates on unseen parameter sets):
  1. Compilation & finiteness: runs on GPU in float64 with no NaN/Inf and no external subprocess/omc calls.
  2. Trajectory parity: matches OpenModelica DASSL (`tolerance=1e-10`) on held-out parameter sets (including regime transitions) to relative error <= {t.traj_rtol:g}.
  3. Exact conservation: closed-loop molar inventory drift across 1,000 trajectories must be < 1e-11.
  4. Exact reverse-mode sensitivities: `jax.grad` through `differentiable_closure` must match `jax.jvp` to < 1e-10, closure central finite differences to < 5e-5, AND fresh macro finite differences to < 1.5e-2.
  5. Speedup / wall-time ladder: once all four gates pass, score (0 to 10) is the share of wall-time rungs ({rungs}) beaten by `(median of 5 runs of 1,000-trajectory forward_ensemble) + (64-trajectory jax.grad pass)`.

{omc_hint}"""

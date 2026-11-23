model BlanketTritiumLoop
  parameter Real u_pump = 1.0 "Normalized coolant pump head command";
  parameter Real u_ext = 0.65 "Normalized vacuum-permeator extraction command";
  parameter Real P_pulse = 1.0 "Normalized fusion neutron heating amplitude";
  parameter Real k_valve = 18.0 "Check-valve backflow resistance gain";

  constant Real tau_hyd = 0.025 "Fast hydraulic inertia time scale (s)";
  constant Real k_fric = 0.22 "Turbulent loop friction coefficient";
  constant Real q_nom = 2.0 "Nominal mass flow rate scale";
  constant Real C_blanket = 28.0 "Blanket heat capacity (MJ/K)";
  constant Real C_hx = 12.0 "Heat exchanger secondary capacity (MJ/K)";
  constant Real cp_cool = 1.8 "Effective coolant heat capacity (MJ/(kg_eq*K))";
  constant Real h_sink = 0.85 "Power-conversion heat sink conductance (MW/K)";
  constant Real T_sink = 573.15 "Steam/sCO2 sink temperature (K)";
  constant Real P_base = 120.0 "Base blanket nuclear heating (MW)";
  constant Real k_perm = 0.035 "Sieverts-law permeation rate constant (1/s)";
  constant Real k_regen = 0.045 "Getter bed regeneration transfer rate (1/s)";
  constant Real eps_smooth = 1.0e-6 "Regularization scale";

  Real q_flow(start = 1.8, fixed = true) "Coolant flow rate";
  Real T_blanket(start = 760.0, fixed = true) "Blanket temperature (K)";
  Real T_hx(start = 640.0, fixed = true) "Heat-exchanger temperature (K)";
  Real N_blanket(start = 4.0, fixed = true) "Tritium in blanket coolant (mol)";
  Real N_extractor(start = 0.5, fixed = true) "Tritium in permeator (mol)";
  Real N_storage(start = 0.5, fixed = true) "Tritium in getter storage (mol)";

  Real P_fus;
  Real q_abs;
  Real valve_drag;
  Real j_perm;
  Real j_store;
equation
  P_fus = P_base * (1.0 + 0.35 * P_pulse * sin(0.15 * time));
  q_abs = sqrt(q_flow * q_flow + eps_smooth * eps_smooth);
  valve_drag = k_valve * 0.5 * (q_abs - q_flow);
  der(q_flow) = (u_pump * q_nom - k_fric * q_flow * q_abs - valve_drag - 0.35 * q_flow) / tau_hyd;

  der(T_blanket) = (P_fus - q_abs * cp_cool * (T_blanket - T_hx)) / C_blanket;
  der(T_hx) = (q_abs * cp_cool * (T_blanket - T_hx) - h_sink * (T_hx - T_sink)) / C_hx;

  j_perm = u_ext * k_perm * N_blanket / sqrt(N_blanket * N_blanket + 0.25);
  j_store = k_regen * N_extractor;
  der(N_blanket) = -j_perm;
  der(N_extractor) = j_perm - j_store;
  der(N_storage) = j_store;
end BlanketTritiumLoop;

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

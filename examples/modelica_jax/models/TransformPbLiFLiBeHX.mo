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

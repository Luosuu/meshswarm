model CryoHDTColumn16
  constant Integer N = 16;
  parameter Real V_boil = 1.85 "Reboiler molar vapor boilup rate (mol/s)";
  parameter Real d_rec_frac = 0.14 "Overhead distillate-to-feed internal recycle fraction";
  parameter Real dp_tray = 0.012 "Pressure drop per tray (bar)";
  parameter Real P_top = 1.15 "Condenser top pressure (bar)";

  constant Real V_flood = 2.15 "Hydraulic flooding onset vapor rate (mol/s)";
  constant Real M[N] = {4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0};
  constant Real x_H2_init[N] = {0.5500000000, 0.4781470295, 0.4156810578, 0.3613757509, 0.3141649851, 0.2731219171, 0.2374407879, 0.2064211044, 0.1794538870, 0.1560097146, 0.1356283302, 0.1179096058, 0.1025056868, 0.0891141630, 0.0774721315, 0.0673510355};
  constant Real x_HD_init[N] = {0.2800000000, 0.2800000000, 0.2800000000, 0.2800000000, 0.2800000000, 0.2800000000, 0.2800000000, 0.2800000000, 0.2800000000, 0.2800000000, 0.2800000000, 0.2800000000, 0.2800000000, 0.2800000000, 0.2800000000, 0.2800000000};

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

    0 = x_H2[i] * Psat_H2[i] + x_HD[i] * Psat_HD[i] + x_T2[i] * Psat_T2[i] - P_stage[i];

    y_H2[i] = (1.0 - eta_tray) * x_H2[i] + eta_tray * x_H2[i] * Psat_H2[i] / P_stage[i];
    y_HD[i] = (1.0 - eta_tray) * x_HD[i] + eta_tray * x_HD[i] * Psat_HD[i] / P_stage[i];
  end for;

  M[1] * der(x_H2[1]) = V_eff * y_H2[2] - V_eff * x_H2[1];
  M[1] * der(x_HD[1]) = V_eff * y_HD[2] - V_eff * x_HD[1];

  for i in 2:7 loop
    M[i] * der(x_H2[i]) = L_rect * x_H2[i - 1] + V_eff * y_H2[i + 1] - L_rect * x_H2[i] - V_eff * y_H2[i];
    M[i] * der(x_HD[i]) = L_rect * x_HD[i - 1] + V_eff * y_HD[i + 1] - L_rect * x_HD[i] - V_eff * y_HD[i];
  end for;

  M[8] * der(x_H2[8]) = L_rect * x_H2[7] + D_rec * x_H2[1] + V_eff * y_H2[9] - V_eff * x_H2[8] - V_eff * y_H2[8];
  M[8] * der(x_HD[8]) = L_rect * x_HD[7] + D_rec * x_HD[1] + V_eff * y_HD[9] - V_eff * x_HD[8] - V_eff * y_HD[8];

  for i in 9:15 loop
    M[i] * der(x_H2[i]) = V_eff * x_H2[i - 1] + V_eff * y_H2[i + 1] - V_eff * x_H2[i] - V_eff * y_H2[i];
    M[i] * der(x_HD[i]) = V_eff * x_HD[i - 1] + V_eff * y_HD[i + 1] - V_eff * x_HD[i] - V_eff * y_HD[i];
  end for;

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

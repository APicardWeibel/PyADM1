import warnings

import numpy as np
import pandas as pd
import scipy.integrate
import yaml

R = 0.083145
p_atm = 1.013


def run_ADM1(
    param: pd.Series,  ## Dict would work as well
    influent_state: pd.DataFrame,
    initial_state: pd.Series,
    V_liq: float,
    V_gas: float,
    T_ad: float,
    T_op: float = None,
    solvermethod: str = "DOP853",
    days_only: bool = True,
):
    warnings.filterwarnings("error", category=RuntimeWarning)
    if T_op is None:
        T_op = T_ad
    ################################################################
    ## SET PARAMETER VALUES

    ## Unpack values (unpacked values are slightly more efficiently accessed)
    T_base = param["T_base"]

    f_sI_xc = param["f_sI_xc"]
    f_xI_xc = param["f_xI_xc"]
    f_ch_xc = param["f_ch_xc"]
    f_pr_xc = param["f_pr_xc"]
    f_li_xc = param["f_li_xc"]
    N_xc = param["N_xc"]
    N_I = param["N_I"]
    N_aa = param["N_aa"]
    C_xc = param["C_xc"]
    C_sI = param["C_sI"]
    C_ch = param["C_ch"]
    C_pr = param["C_pr"]
    C_li = param["C_li"]
    C_xI = param["C_xI"]
    C_su = param["C_su"]
    C_aa = param["C_aa"]
    f_fa_li = param["f_fa_li"]
    C_fa = param["C_fa"]
    f_h2_su = param["f_h2_su"]
    f_bu_su = param["f_bu_su"]
    f_pro_su = param["f_pro_su"]
    f_ac_su = param["f_ac_su"]
    N_bac = param["N_bac"]
    C_bu = param["C_bu"]
    C_pro = param["C_pro"]
    C_ac = param["C_ac"]
    C_bac = param["C_bac"]
    Y_su = param["Y_su"]
    f_h2_aa = param["f_h2_aa"]
    f_va_aa = param["f_va_aa"]
    f_bu_aa = param["f_bu_aa"]
    f_pro_aa = param["f_pro_aa"]
    f_ac_aa = param["f_ac_aa"]
    C_va = param["C_va"]
    Y_aa = param["Y_aa"]
    Y_fa = param["Y_fa"]
    Y_c4 = param["Y_c4"]
    Y_pro = param["Y_pro"]
    C_ch4 = param["C_ch4"]
    Y_ac = param["Y_ac"]
    Y_h2 = param["Y_h2"]

    k_dis = param["k_dis"]
    k_hyd_ch = param["k_hyd_ch"]
    k_hyd_pr = param["k_hyd_pr"]
    k_hyd_li = param["k_hyd_li"]
    K_S_IN = param["K_S_IN"]
    k_m_su = param["k_m_su"]
    K_S_su = param["K_S_su"]
    pH_UL_aa = param["pH_UL_aa"]
    pH_LL_aa = param["pH_LL_aa"]
    k_m_aa = param["k_m_aa"]
    K_S_aa = param["K_S_aa"]
    k_m_fa = param["k_m_fa"]
    K_S_fa = param["K_S_fa"]
    K_I_h2_fa = param["K_I_h2_fa"]
    k_m_c4 = param["k_m_c4"]
    K_S_c4 = param["K_S_c4"]
    K_I_h2_c4 = param["K_I_h2_c4"]
    k_m_pro = param["k_m_pro"]
    K_S_pro = param["K_S_pro"]
    K_I_h2_pro = param["K_I_h2_pro"]
    k_m_ac = param["k_m_ac"]
    K_S_ac = param["K_S_ac"]
    K_I_nh3 = param["K_I_nh3"]
    pH_UL_ac = param["pH_UL_ac"]
    pH_LL_ac = param["pH_LL_ac"]
    k_m_h2 = param["k_m_h2"]
    K_S_h2 = param["K_S_h2"]
    pH_UL_h2 = param["pH_UL_h2"]
    pH_LL_h2 = param["pH_LL_h2"]
    k_dec_X_su = param["k_dec_X_su"]
    k_dec_X_aa = param["k_dec_X_aa"]
    k_dec_X_fa = param["k_dec_X_fa"]
    k_dec_X_c4 = param["k_dec_X_c4"]
    k_dec_X_pro = param["k_dec_X_pro"]
    k_dec_X_ac = param["k_dec_X_ac"]
    k_dec_X_h2 = param["k_dec_X_h2"]

    K_a_va = param["K_a_va"]
    K_a_bu = param["K_a_bu"]
    K_a_pro = param["K_a_pro"]
    K_a_ac = param["K_a_ac"]

    p_gas_h2o = 0.0313 * np.exp(5290 * (1 / T_base - 1 / T_ad))  # bar #0.0557
    k_p = param[
        "k_p"
    ]  # m^3.d^-1.bar^-1 #only for BSM2 AD conditions, recalibrate for other AD cases #gas outlet friction
    k_L_a = param["k_L_a"]
    K_H_co2 = 0.035 * np.exp(
        (-19410 / (100 * R)) * (1 / T_base - 1 / T_ad)
    )  # Mliq.bar^-1 #0.0271
    K_H_ch4 = 0.0014 * np.exp(
        (-14240 / (100 * R)) * (1 / T_base - 1 / T_ad)
    )  # Mliq.bar^-1 #0.00116
    K_H_h2 = (
        7.8 * 10**-4 * np.exp(-4180 / (100 * R) * (1 / T_base - 1 / T_ad))
    )  # Mliq.bar^-1 #7.38*10^-4

    # T_ad depends on time, should be influent_state
    p_gas_h2o = 0.0313 * np.exp(5290 * (1 / T_base - 1 / T_ad))
    K_w = 10**-14.0 * np.exp(
        (55900 / (100 * R)) * (1 / T_base - 1 / T_ad)
    )  # M #2.08 * 10 ^ -14
    K_H_co2 = 0.035 * np.exp(
        (-19410 / (100 * R)) * (1 / T_base - 1 / T_ad)
    )  # Mliq.bar^-1 #0.0271
    K_H_ch4 = 0.0014 * np.exp(
        (-14240 / (100 * R)) * (1 / T_base - 1 / T_ad)
    )  # Mliq.bar^-1 #0.00116
    K_H_h2 = (
        7.8 * 10**-4 * np.exp(-4180 / (100 * R) * (1 / T_base - 1 / T_ad))
    )  # Mliq.bar^-1 #7.38*10^-4

    K_pH_aa = 10 ** (-1 * (pH_LL_aa + pH_UL_aa) / 2.0)
    nn_aa = 3.0 / (
        pH_UL_aa - pH_LL_aa
    )  # we need a differece between N_aa and n_aa to avoid typos and nn_aa refers to the n_aa in BSM2 report
    K_pH_ac = 10 ** (-1 * (pH_LL_ac + pH_UL_ac) / 2.0)
    n_ac = 3.0 / (pH_UL_ac - pH_LL_ac)
    K_pH_h2 = 10 ** (-1 * (pH_LL_h2 + pH_UL_h2) / 2.0)
    n_h2 = 3.0 / (pH_UL_h2 - pH_LL_h2)

    K_a_co2 = 10**-6.35 * np.exp(
        (7646 / (100 * R)) * (1 / T_base - 1 / T_ad)
    )  # M #4.94 * 10 ^ -7
    K_a_IN = 10**-9.25 * np.exp(
        (51965 / (100 * R)) * (1 / T_base - 1 / T_ad)
    )  # M #1.11 * 10 ^ -9

    ## Add equation parameter
    s_1 = (
        -1 * C_xc
        + f_sI_xc * C_sI
        + f_ch_xc * C_ch
        + f_pr_xc * C_pr
        + f_li_xc * C_li
        + f_xI_xc * C_xI
    )
    s_2 = -1 * C_ch + C_su
    s_3 = -1 * C_pr + C_aa
    s_4 = -1 * C_li + (1 - f_fa_li) * C_su + f_fa_li * C_fa
    s_5 = (
        -1 * C_su
        + (1 - Y_su) * (f_bu_su * C_bu + f_pro_su * C_pro + f_ac_su * C_ac)
        + Y_su * C_bac
    )
    s_6 = (
        -1 * C_aa
        + (1 - Y_aa)
        * (f_va_aa * C_va + f_bu_aa * C_bu + f_pro_aa * C_pro + f_ac_aa * C_ac)
        + Y_aa * C_bac
    )
    s_7 = -1 * C_fa + (1 - Y_fa) * 0.7 * C_ac + Y_fa * C_bac
    s_8 = (
        -1 * C_va + (1 - Y_c4) * 0.54 * C_pro + (1 - Y_c4) * 0.31 * C_ac + Y_c4 * C_bac
    )
    s_9 = -1 * C_bu + (1 - Y_c4) * 0.8 * C_ac + Y_c4 * C_bac
    s_10 = -1 * C_pro + (1 - Y_pro) * 0.57 * C_ac + Y_pro * C_bac
    s_11 = -1 * C_ac + (1 - Y_ac) * C_ch4 + Y_ac * C_bac
    s_12 = (1 - Y_h2) * C_ch4 + Y_h2 * C_bac
    s_13 = -1 * C_bac + C_xc

    ## END OF PARAM UNPACKING
    ############################################
    ## Create initial values
    state_zero = initial_state.loc[
        0,
        [
            "S_su",
            "S_aa",
            "S_fa",
            "S_va",
            "S_bu",
            "S_pro",
            "S_ac",
            "S_h2",
            "S_ch4",
            "S_IC",
            "S_IN",
            "S_I",
            "X_xc",
            "X_ch",
            "X_pr",
            "X_li",
            "X_su",
            "X_aa",
            "X_fa",
            "X_c4",
            "X_pro",
            "X_ac",
            "X_h2",
            "X_I",
            "S_cation",
            "S_anion",
            "S_H_ion",
            "S_va_ion",
            "S_bu_ion",
            "S_pro_ion",
            "S_ac_ion",
            "S_hco3_ion",
            "S_nh3",
            "S_gas_h2",
            "S_gas_ch4",
            "S_gas_co2",
        ],
    ].to_numpy()

    t = influent_state["time"]

    columns = [
        "S_su",
        "S_aa",
        "S_fa",
        "S_va",
        "S_bu",
        "S_pro",
        "S_ac",
        "S_h2",
        "S_ch4",
        "S_IC",
        "S_IN",
        "S_I",
        "X_xc",
        "X_ch",
        "X_pr",
        "X_li",
        "X_su",
        "X_aa",
        "X_fa",
        "X_c4",
        "X_pro",
        "X_ac",
        "X_h2",
        "X_I",
        "S_cation",
        "S_anion",
        "pH",
        "S_va_ion",
        "S_bu_ion",
        "S_pro_ion",
        "S_ac_ion",
        "S_hco3_ion",
        "S_nh3",
        "S_gas_h2",
        "S_gas_ch4",
        "S_gas_co2",
    ]

    ## Create empty accu for results
    simulate_results = pd.DataFrame(index=range(len(t) + 1), columns=columns)
    simulate_results.iloc[0, :] = state_zero

    ########################################################################
    ## ADM1_ODE
    def ADM1_ODE(t, state_zero):
        ## Unpack state_zero
        S_su = state_zero[0]
        S_aa = state_zero[1]
        S_fa = state_zero[2]
        S_va = state_zero[3]
        S_bu = state_zero[4]
        S_pro = state_zero[5]
        S_ac = state_zero[6]
        S_h2 = state_zero[7]
        S_ch4 = state_zero[8]
        S_IC = state_zero[9]
        S_IN = state_zero[10]
        S_I = state_zero[11]
        X_xc = state_zero[12]
        X_ch = state_zero[13]
        X_pr = state_zero[14]
        X_li = state_zero[15]
        X_su = state_zero[16]
        X_aa = state_zero[17]
        X_fa = state_zero[18]
        X_c4 = state_zero[19]
        X_pro = state_zero[20]
        X_ac = state_zero[21]
        X_h2 = state_zero[22]
        X_I = state_zero[23]
        S_cation = state_zero[24]
        S_anion = state_zero[25]
        S_H_ion = state_zero[26]
        S_hco3_ion = state_zero[31]
        S_nh3 = state_zero[32]
        S_gas_h2 = state_zero[33]
        S_gas_ch4 = state_zero[34]
        S_gas_co2 = state_zero[35]

        # Main part of the function
        try:
            I_pH_aa = (K_pH_aa**nn_aa) / (S_H_ion**nn_aa + K_pH_aa**nn_aa)
        except RuntimeWarning:
            print(
                f"RuntimeWarning: K_ph_aa: {K_pH_aa}, nn_aa: {nn_aa}, S_H_ion: {S_H_ion}"
            )
            I_pH_aa = 0
        try:
            I_pH_h2 = (K_pH_h2**n_h2) / (S_H_ion**n_h2 + K_pH_h2**n_h2)
        except RuntimeWarning:
            print(
                f"RunitmeWarning: K_ph_h2: {K_pH_h2}, n_h2: {n_h2}, S_H_ion: {S_H_ion}"
            )
            I_pH_h2 = 0
        I_pH_ac = (K_pH_ac**n_ac) / (S_H_ion**n_ac + K_pH_ac**n_ac)
        I_IN_lim = 1 / (1 + (K_S_IN / S_IN))
        I_h2_fa = 1 / (1 + (S_h2 / K_I_h2_fa))
        I_h2_c4 = 1 / (1 + (S_h2 / K_I_h2_c4))
        I_h2_pro = 1 / (1 + (S_h2 / K_I_h2_pro))
        I_nh3 = 1 / (1 + (S_nh3 / K_I_nh3))

        I_5 = I_pH_aa * I_IN_lim
        I_6 = I_5
        I_7 = I_pH_aa * I_IN_lim * I_h2_fa
        I_8 = I_pH_aa * I_IN_lim * I_h2_c4
        I_9 = I_8
        I_10 = I_pH_aa * I_IN_lim * I_h2_pro
        I_11 = I_pH_ac * I_IN_lim * I_nh3
        I_12 = I_pH_h2 * I_IN_lim

        # biochemical process rates from Rosen et al (2006) BSM2 report
        Rho_1 = k_dis * X_xc  # Disintegration
        Rho_2 = k_hyd_ch * X_ch  # Hydrolysis of carbohydrates
        Rho_3 = k_hyd_pr * X_pr  # Hydrolysis of proteins
        Rho_4 = k_hyd_li * X_li  # Hydrolysis of lipids
        Rho_5 = k_m_su * (S_su / (K_S_su + S_su)) * X_su * I_5  # Uptake of sugars
        Rho_6 = k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6  # Uptake of amino-acids
        Rho_7 = (
            k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7
        )  # Uptake of LCFA (long-chain fatty acids)
        Rho_8 = (
            k_m_c4
            * (S_va / (K_S_c4 + S_va))
            * X_c4
            * (S_va / (S_bu + S_va + 1e-6))
            * I_8
        )  # Uptake of valerate
        Rho_9 = (
            k_m_c4
            * (S_bu / (K_S_c4 + S_bu))
            * X_c4
            * (S_bu / (S_bu + S_va + 1e-6))
            * I_9
        )  # Uptake of butyrate
        Rho_10 = (
            k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10
        )  # Uptake of propionate
        Rho_11 = k_m_ac * (S_ac / (K_S_ac + S_ac)) * X_ac * I_11  # Uptake of acetate
        Rho_12 = k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12  # Uptake of hydrogen
        Rho_13 = k_dec_X_su * X_su  # Decay of X_su
        Rho_14 = k_dec_X_aa * X_aa  # Decay of X_aa
        Rho_15 = k_dec_X_fa * X_fa  # Decay of X_fa
        Rho_16 = k_dec_X_c4 * X_c4  # Decay of X_c4
        Rho_17 = k_dec_X_pro * X_pro  # Decay of X_pro
        Rho_18 = k_dec_X_ac * X_ac  # Decay of X_ac
        Rho_19 = k_dec_X_h2 * X_h2  # Decay of X_h2

        # gas phase algebraic equations from Rosen et al (2006) BSM2 report
        p_gas_h2 = S_gas_h2 * R * T_op / 16
        p_gas_ch4 = S_gas_ch4 * R * T_op / 64
        p_gas_co2 = S_gas_co2 * R * T_op

        p_gas = (
            p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o
        )  ## Check whether p_gas_h2O is a parameter
        q_gas = k_p * (p_gas - p_atm)
        if q_gas < 0:
            q_gas = 0

        # gas transfer rates from Rosen et al (2006) BSM2 report
        Rho_T_8 = k_L_a * (S_h2 - 16 * K_H_h2 * p_gas_h2)
        Rho_T_9 = k_L_a * (S_ch4 - 64 * K_H_ch4 * p_gas_ch4)
        Rho_T_10 = k_L_a * (S_IC - S_hco3_ion - K_H_co2 * p_gas_co2)

        ##differential equaitons from Rosen et al (2006) BSM2 report
        # differential equations 1 to 12 (soluble matter)
        diff_S_su = (
            q_ad / V_liq * (S_su_in - S_su) + Rho_2 + (1 - f_fa_li) * Rho_4 - Rho_5
        )  # eq1

        diff_S_aa = q_ad / V_liq * (S_aa_in - S_aa) + Rho_3 - Rho_6  # eq2

        diff_S_fa = q_ad / V_liq * (S_fa_in - S_fa) + (f_fa_li * Rho_4) - Rho_7  # eq3

        diff_S_va = (
            q_ad / V_liq * (S_va_in - S_va) + (1 - Y_aa) * f_va_aa * Rho_6 - Rho_8
        )  # eq4

        diff_S_bu = (
            q_ad / V_liq * (S_bu_in - S_bu)
            + (1 - Y_su) * f_bu_su * Rho_5
            + (1 - Y_aa) * f_bu_aa * Rho_6
            - Rho_9
        )  # eq5

        diff_S_pro = (
            q_ad / V_liq * (S_pro_in - S_pro)
            + (1 - Y_su) * f_pro_su * Rho_5
            + (1 - Y_aa) * f_pro_aa * Rho_6
            + (1 - Y_c4) * 0.54 * Rho_8
            - Rho_10
        )  # eq6

        diff_S_ac = (
            q_ad / V_liq * (S_ac_in - S_ac)
            + (1 - Y_su) * f_ac_su * Rho_5
            + (1 - Y_aa) * f_ac_aa * Rho_6
            + (1 - Y_fa) * 0.7 * Rho_7
            + (1 - Y_c4) * 0.31 * Rho_8
            + (1 - Y_c4) * 0.8 * Rho_9
            + (1 - Y_pro) * 0.57 * Rho_10
            - Rho_11
        )  # eq7

        # diff_S_h2 is defined with DAE parralel equations

        diff_S_ch4 = (
            q_ad / V_liq * (S_ch4_in - S_ch4)
            + (1 - Y_ac) * Rho_11
            + (1 - Y_h2) * Rho_12
            - Rho_T_9
        )  # eq9

        ## eq10 start##
        Sigma = (
            s_1 * Rho_1
            + s_2 * Rho_2
            + s_3 * Rho_3
            + s_4 * Rho_4
            + s_5 * Rho_5
            + s_6 * Rho_6
            + s_7 * Rho_7
            + s_8 * Rho_8
            + s_9 * Rho_9
            + s_10 * Rho_10
            + s_11 * Rho_11
            + s_12 * Rho_12
            + s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        )

        diff_S_IC = q_ad / V_liq * (S_IC_in - S_IC) - Sigma - Rho_T_10
        ## eq10 end##

        diff_S_IN = (
            q_ad / V_liq * (S_IN_in - S_IN)
            + (N_xc - f_xI_xc * N_I - f_sI_xc * N_I - f_pr_xc * N_aa) * Rho_1
            - Y_su * N_bac * Rho_5
            + (N_aa - Y_aa * N_bac) * Rho_6
            - Y_fa * N_bac * Rho_7
            - Y_c4 * N_bac * Rho_8
            - Y_c4 * N_bac * Rho_9
            - Y_pro * N_bac * Rho_10
            - Y_ac * N_bac * Rho_11
            - Y_h2 * N_bac * Rho_12
            + (N_bac - N_xc)
            * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        )  # eq11

        diff_S_I = q_ad / V_liq * (S_I_in - S_I) + f_sI_xc * Rho_1  # eq12

        # Differential equations 13 to 24 (particulate matter)
        diff_X_xc = (
            q_ad / V_liq * (X_xc_in - X_xc)
            - Rho_1
            + Rho_13
            + Rho_14
            + Rho_15
            + Rho_16
            + Rho_17
            + Rho_18
            + Rho_19
        )  # eq13

        diff_X_ch = q_ad / V_liq * (X_ch_in - X_ch) + f_ch_xc * Rho_1 - Rho_2  # eq14

        diff_X_pr = q_ad / V_liq * (X_pr_in - X_pr) + f_pr_xc * Rho_1 - Rho_3  # eq15

        diff_X_li = q_ad / V_liq * (X_li_in - X_li) + f_li_xc * Rho_1 - Rho_4  # eq16

        diff_X_su = q_ad / V_liq * (X_su_in - X_su) + Y_su * Rho_5 - Rho_13  # eq17

        diff_X_aa = q_ad / V_liq * (X_aa_in - X_aa) + Y_aa * Rho_6 - Rho_14  # eq18

        diff_X_fa = q_ad / V_liq * (X_fa_in - X_fa) + Y_fa * Rho_7 - Rho_15  # eq19

        diff_X_c4 = (
            q_ad / V_liq * (X_c4_in - X_c4) + Y_c4 * Rho_8 + Y_c4 * Rho_9 - Rho_16
        )  # eq20

        diff_X_pro = q_ad / V_liq * (X_pro_in - X_pro) + Y_pro * Rho_10 - Rho_17  # eq21

        diff_X_ac = q_ad / V_liq * (X_ac_in - X_ac) + Y_ac * Rho_11 - Rho_18  # eq22

        diff_X_h2 = q_ad / V_liq * (X_h2_in - X_h2) + Y_h2 * Rho_12 - Rho_19  # eq23

        diff_X_I = q_ad / V_liq * (X_I_in - X_I) + f_xI_xc * Rho_1  # eq24

        # Differential equations 25 and 26 (cations and anions)
        diff_S_cation = q_ad / V_liq * (S_cation_in - S_cation)  # eq25

        diff_S_anion = q_ad / V_liq * (S_anion_in - S_anion)  # eq26

        diff_S_h2 = 0

        # Differential equations 27 to 32 (ion states, only for ODE implementation)
        diff_S_va_ion = 0  # eq27 ## Changed with DAESolve
        diff_S_bu_ion = 0  # eq28
        diff_S_pro_ion = 0  # eq29
        diff_S_ac_ion = 0  # eq30
        diff_S_hco3_ion = 0  # eq31
        diff_S_nh3 = 0  # eq32

        # Gas phase equations: Differential equations 33 to 35
        diff_S_gas_h2 = (q_gas / V_gas * -1 * S_gas_h2) + (
            Rho_T_8 * V_liq / V_gas
        )  # eq33

        diff_S_gas_ch4 = (q_gas / V_gas * -1 * S_gas_ch4) + (
            Rho_T_9 * V_liq / V_gas
        )  # eq34

        diff_S_gas_co2 = (q_gas / V_gas * -1 * S_gas_co2) + (
            Rho_T_10 * V_liq / V_gas
        )  # eq35

        diff_S_H_ion = 0

        return (
            diff_S_su,
            diff_S_aa,
            diff_S_fa,
            diff_S_va,
            diff_S_bu,
            diff_S_pro,
            diff_S_ac,
            diff_S_h2,
            diff_S_ch4,
            diff_S_IC,
            diff_S_IN,
            diff_S_I,
            diff_X_xc,
            diff_X_ch,
            diff_X_pr,
            diff_X_li,
            diff_X_su,
            diff_X_aa,
            diff_X_fa,
            diff_X_c4,
            diff_X_pro,
            diff_X_ac,
            diff_X_h2,
            diff_X_I,
            diff_S_cation,
            diff_S_anion,
            diff_S_H_ion,
            diff_S_va_ion,
            diff_S_bu_ion,
            diff_S_pro_ion,
            diff_S_ac_ion,
            diff_S_hco3_ion,
            diff_S_nh3,
            diff_S_gas_h2,
            diff_S_gas_ch4,
            diff_S_gas_co2,
        )

    ## Simulate
    def simulate(
        tstep,
        state_zero,
    ):
        r = scipy.integrate.solve_ivp(ADM1_ODE, tstep, state_zero, method=solvermethod)
        return r.y[:, -1]

    ################################################################
    ## Main loop
    t0 = 0
    n = 0

    for u in t:
        ## Set up influent state
        (
            S_su_in,
            S_aa_in,
            S_fa_in,
            S_va_in,
            S_bu_in,
            S_pro_in,
            S_ac_in,
            S_h2_in,
            S_ch4_in,
            S_IC_in,
            S_IN_in,
            S_I_in,
            X_xc_in,
            X_ch_in,
            X_pr_in,
            X_li_in,
            X_su_in,
            X_aa_in,
            X_fa_in,
            X_c4_in,
            X_pro_in,
            X_ac_in,
            X_h2_in,
            X_I_in,
            S_cation_in,
            S_anion_in,
            q_ad,
        ) = influent_state.loc[
            n,
            [
                "S_su",
                "S_aa",
                "S_fa",
                "S_va",
                "S_bu",
                "S_pro",
                "S_ac",
                "S_h2",
                "S_ch4",
                "S_IC",
                "S_IN",
                "S_I",
                "X_xc",
                "X_ch",
                "X_pr",
                "X_li",
                "X_su",
                "X_aa",
                "X_fa",
                "X_c4",
                "X_pro",
                "X_ac",
                "X_h2",
                "X_I",
                "S_cation",
                "S_anion",
                "Q",
            ],
        ].to_numpy()

        # Span for next time step
        tstep = [t0, u]

        # Run integration to next step
        (
            S_su,
            S_aa,
            S_fa,
            S_va,
            S_bu,
            S_pro,
            S_ac,
            S_h2,
            S_ch4,
            S_IC,
            S_IN,
            S_I,
            X_xc,
            X_ch,
            X_pr,
            X_li,
            X_su,
            X_aa,
            X_fa,
            X_c4,
            X_pro,
            X_ac,
            X_h2,
            X_I,
            S_cation,
            S_anion,
            S_H_ion,
            S_va_ion,
            S_bu_ion,
            S_pro_ion,
            S_ac_ion,
            S_hco3_ion,
            S_nh3,
            S_gas_h2,
            S_gas_ch4,
            S_gas_co2,
        ) = simulate(tstep=tstep, state_zero=state_zero)

        # Solve DAE states
        eps = 0.0000001
        prevS_H_ion = S_H_ion

        # initial values for Newton-Raphson solver parameter
        shdelta = 1.0
        shgradeq = 1.0
        S_h2delta = 1.0
        S_h2gradeq = 1.0
        tol = 10 ** (-12)  # solver accuracy tolerance
        maxIter = 1000  # maximum number of iterations for solver
        i = 1
        j = 1

        ## DAE solver for S_H_ion from Rosen et al. (2006)
        while (shdelta > tol or shdelta < -tol) and (i <= maxIter):
            S_va_ion = K_a_va * S_va / (K_a_va + S_H_ion)
            S_bu_ion = K_a_bu * S_bu / (K_a_bu + S_H_ion)
            S_pro_ion = K_a_pro * S_pro / (K_a_pro + S_H_ion)
            S_ac_ion = K_a_ac * S_ac / (K_a_ac + S_H_ion)
            S_hco3_ion = K_a_co2 * S_IC / (K_a_co2 + S_H_ion)
            S_nh3 = K_a_IN * S_IN / (K_a_IN + S_H_ion)
            shdelta = (
                S_cation
                + (S_IN - S_nh3)
                + S_H_ion
                - S_hco3_ion
                - S_ac_ion / 64.0
                - S_pro_ion / 112.0
                - S_bu_ion / 160.0
                - S_va_ion / 208.0
                - K_w / S_H_ion
                - S_anion
            )
            shgradeq = (
                1
                + K_a_IN * S_IN / ((K_a_IN + S_H_ion) * (K_a_IN + S_H_ion))
                + K_a_co2 * S_IC / ((K_a_co2 + S_H_ion) * (K_a_co2 + S_H_ion))
                + 1 / 64.0 * K_a_ac * S_ac / ((K_a_ac + S_H_ion) * (K_a_ac + S_H_ion))
                + 1
                / 112.0
                * K_a_pro
                * S_pro
                / ((K_a_pro + S_H_ion) * (K_a_pro + S_H_ion))
                + 1 / 160.0 * K_a_bu * S_bu / ((K_a_bu + S_H_ion) * (K_a_bu + S_H_ion))
                + 1 / 208.0 * K_a_va * S_va / ((K_a_va + S_H_ion) * (K_a_va + S_H_ion))
                + K_w / (S_H_ion * S_H_ion)
            )
            S_H_ion = S_H_ion - shdelta / shgradeq
            if S_H_ion <= 0:
                S_H_ion = tol
            i += 1

        ## DAE solver for S_h2 from Rosen et al. (2006)
        while (S_h2delta > tol or S_h2delta < -tol) and (j <= maxIter):
            I_pH_aa = (K_pH_aa**nn_aa) / (prevS_H_ion**nn_aa + K_pH_aa**nn_aa)

            I_pH_h2 = (K_pH_h2**n_h2) / (prevS_H_ion**n_h2 + K_pH_h2**n_h2)
            I_IN_lim = 1 / (1 + (K_S_IN / S_IN))
            I_h2_fa = 1 / (1 + (S_h2 / K_I_h2_fa))
            I_h2_c4 = 1 / (1 + (S_h2 / K_I_h2_c4))
            I_h2_pro = 1 / (1 + (S_h2 / K_I_h2_pro))

            I_5 = I_pH_aa * I_IN_lim
            I_6 = I_5
            I_7 = I_pH_aa * I_IN_lim * I_h2_fa
            I_8 = I_pH_aa * I_IN_lim * I_h2_c4
            I_9 = I_8
            I_10 = I_pH_aa * I_IN_lim * I_h2_pro

            I_12 = I_pH_h2 * I_IN_lim
            Rho_5 = k_m_su * (S_su / (K_S_su + S_su)) * X_su * I_5  # Uptake of sugars
            Rho_6 = (
                k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6
            )  # Uptake of amino-acids
            Rho_7 = (
                k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7
            )  # Uptake of LCFA (long-chain fatty acids)
            Rho_8 = (
                k_m_c4
                * (S_va / (K_S_c4 + S_va))
                * X_c4
                * (S_va / (S_bu + S_va + 1e-6))
                * I_8
            )  # Uptake of valerate
            Rho_9 = (
                k_m_c4
                * (S_bu / (K_S_c4 + S_bu))
                * X_c4
                * (S_bu / (S_bu + S_va + 1e-6))
                * I_9
            )  # Uptake of butyrate
            Rho_10 = (
                k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10
            )  # Uptake of propionate
            Rho_12 = (
                k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12
            )  # Uptake of hydrogen
            p_gas_h2 = S_gas_h2 * R * T_ad / 16
            Rho_T_8 = k_L_a * (S_h2 - 16 * K_H_h2 * p_gas_h2)
            S_h2delta = (
                q_ad / V_liq * (S_h2_in - S_h2)
                + (1 - Y_su) * f_h2_su * Rho_5
                + (1 - Y_aa) * f_h2_aa * Rho_6
                + (1 - Y_fa) * 0.3 * Rho_7
                + (1 - Y_c4) * 0.15 * Rho_8
                + (1 - Y_c4) * 0.2 * Rho_9
                + (1 - Y_pro) * 0.43 * Rho_10
                - Rho_12
                - Rho_T_8
            )
            S_h2gradeq = (
                -1.0 / V_liq * q_ad
                - 3.0
                / 10.0
                * (1 - Y_fa)
                * k_m_fa
                * S_fa
                / (K_S_fa + S_fa)
                * X_fa
                * I_pH_aa
                / (1 + K_S_IN / S_IN)
                / ((1 + S_h2 / K_I_h2_fa) * (1 + S_h2 / K_I_h2_fa))
                / K_I_h2_fa
                - 3.0
                / 20.0
                * (1 - Y_c4)
                * k_m_c4
                * S_va
                * S_va
                / (K_S_c4 + S_va)
                * X_c4
                / (S_bu + S_va + eps)
                * I_pH_aa
                / (1 + K_S_IN / S_IN)
                / ((1 + S_h2 / K_I_h2_c4) * (1 + S_h2 / K_I_h2_c4))
                / K_I_h2_c4
                - 1.0
                / 5.0
                * (1 - Y_c4)
                * k_m_c4
                * S_bu
                * S_bu
                / (K_S_c4 + S_bu)
                * X_c4
                / (S_bu + S_va + eps)
                * I_pH_aa
                / (1 + K_S_IN / S_IN)
                / ((1 + S_h2 / K_I_h2_c4) * (1 + S_h2 / K_I_h2_c4))
                / K_I_h2_c4
                - 43.0
                / 100.0
                * (1 - Y_pro)
                * k_m_pro
                * S_pro
                / (K_S_pro + S_pro)
                * X_pro
                * I_pH_aa
                / (1 + K_S_IN / S_IN)
                / ((1 + S_h2 / K_I_h2_pro) * (1 + S_h2 / K_I_h2_pro))
                / K_I_h2_pro
                - k_m_h2 / (K_S_h2 + S_h2) * X_h2 * I_pH_h2 / (1 + K_S_IN / S_IN)
                + k_m_h2
                * S_h2
                / ((K_S_h2 + S_h2) * (K_S_h2 + S_h2))
                * X_h2
                * I_pH_h2
                / (1 + K_S_IN / S_IN)
                - k_L_a
            )
            S_h2 = S_h2 - S_h2delta / S_h2gradeq
            if S_h2 <= 0:
                S_h2 = tol
            j += 1
        # DAE states solved

        # Algebraic equations
        p_gas_h2 = S_gas_h2 * R * T_op / 16
        p_gas_ch4 = S_gas_ch4 * R * T_op / 64
        p_gas_co2 = S_gas_co2 * R * T_op
        p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o
        q_gas = k_p * (p_gas - p_atm)
        if q_gas < 0:
            q_gas = 0

        q_ch4 = q_gas * (p_gas_ch4 / p_gas)  # methane flow
        if q_ch4 < 0:
            q_ch4 = 0

        # state transfer
        state_zero = [
            S_su,
            S_aa,
            S_fa,
            S_va,
            S_bu,
            S_pro,
            S_ac,
            S_h2,
            S_ch4,
            S_IC,
            S_IN,
            S_I,
            X_xc,
            X_ch,
            X_pr,
            X_li,
            X_su,
            X_aa,
            X_fa,
            X_c4,
            X_pro,
            X_ac,
            X_h2,
            X_I,
            S_cation,
            S_anion,
            S_H_ion,
            S_va_ion,
            S_bu_ion,
            S_pro_ion,
            S_ac_ion,
            S_hco3_ion,
            S_nh3,
            S_gas_h2,
            S_gas_ch4,
            S_gas_co2,
        ]

        n = n + 1
        simulate_results.iloc[n] = state_zero
        t0 = u
    ## END OF LOOP
    phlogarray = -1 * simulate_results["pH"].apply(lambda x: np.log10(x))
    simulate_results["pH"] = phlogarray
    simulate_results["S_co2"] = (
        simulate_results["S_IC"] - simulate_results["S_hco3_ion"]
    )
    simulate_results["S_nh4_ion"] = simulate_results["S_IN"] - simulate_results["S_nh3"]

    if days_only:
        ## Filter for time
        loc_time = int(t[0])
        t_to_keep = t * 0
        t_to_keep[0] = loc_time

        loc = 1
        loc_day = 1

        while loc < len(t):
            if int(t[loc]) > loc_time:
                t_to_keep[loc_day] = loc
                loc_day = loc_day + 1
                loc_time = int(t[loc])
            loc = loc + 1
        t_to_keep = t_to_keep[0:loc_day]

        simulate_results = simulate_results.loc[t_to_keep, :]

    warnings.resetwarnings()
    return simulate_results


if __name__ == "__main__":
    from time import process_time

    start_time = process_time()
    influent_state = pd.read_csv("digester_influent.csv")
    initial_state = pd.read_csv("digester_initial.csv")
    V_liq = 3400  # m^3
    V_gas = 300  # m^3
    with open("parameter.yml") as file:
        parameter = yaml.load(file, Loader=yaml.FullLoader)
    run_ADM1(
        param=parameter,
        influent_state=influent_state,
        initial_state=initial_state,
        V_liq=V_liq,
        V_gas=V_gas,
        T_ad=308.15,
        days_only=False,
    ).to_csv("dynamic_out_func.csv", index=False)
    end_time = process_time()
    print("Elapsed time during the whole program in seconds:", end_time - start_time)

"""
synthesis_analytics.py — Cross-Dataset Synthesis & Hidden Pattern Analytics
Student Success Intelligence Framework (SSIF)

Performs comprehensive empirical analyses across:
  1. Dataset A: Academic Retention (N=79,239 student-semesters, 20,000 students)
  2. Dataset B: Campus Placement (N=215 candidates)
  3. Synthesized Higher Education Pipeline: Student labor paradox, academic safety
     thresholds, and latent human capital compounding.

RULE-002: Never fabricate student identifiers.
RULE-003: Never row-merge retention and placement datasets.
RULE-016: Always report statistical effect sizes and p-values.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.data_loader import load_placement, load_retention
from src.logger import get_module_logger
from src.retention.features import compute_longitudinal_trajectories

logger = get_module_logger("cross_dataset.synthesis")


@dataclass(frozen=True)
class RetentionTippingPoints:
    gpa_brackets: dict[str, float]
    attendance_brackets: dict[str, float]
    course_load_hazard: dict[int, float]
    scholarship_q1_reduction: float
    advising_early_drop: float


@dataclass(frozen=True)
class PlacementHiddenPatterns:
    workex_odds_ratio: float
    workex_p_value: float
    rescue_rate_low_gpa_workex: float
    rescue_rate_low_gpa_no_workex: float
    mkt_fin_placement_rate: float
    mkt_hr_placement_rate: float
    sci_tech_salary_mean: float
    comm_mgmt_salary_mean: float
    gender_wage_gap_p_value: float
    board_prestige_p_value: float


@dataclass(frozen=True)
class SynthesisPipelineMetrics:
    work_hours_retention_or: float
    work_hours_retention_p_val: float
    workex_placement_or: float
    workex_placement_p_val: float
    retention_safe_gpa_p75: float
    placement_hiring_threshold_p: float


def compute_retention_deep_insights() -> RetentionTippingPoints:
    """Extract non-linear tipping points and vulnerability multipliers from Dataset A."""
    logger.info("Computing Retention deep insights...")
    df = load_retention()

    # GPA buckets
    df['gpa_bucket'] = pd.cut(
        df['Sem_GPA'],
        bins=[0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        labels=['<1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0']
    )
    gpa_rates = (df.groupby('gpa_bucket', observed=False)['Target_Dropout_Next_Sem'].mean() * 100).to_dict()

    # Attendance buckets
    df['att_bucket'] = pd.cut(
        df['Attendance'],
        bins=[0, 60, 70, 75, 80, 85, 90, 100],
        labels=['<60%', '60-70%', '70-75%', '75-80%', '80-85%', '85-90%', '90-100%']
    )
    att_rates = (df.groupby('att_bucket', observed=False)['Target_Dropout_Next_Sem'].mean() * 100).to_dict()

    # Course load hazard
    load_rates = (df.groupby('Course_Load')['Target_Dropout_Next_Sem'].mean() * 100).to_dict()

    # Scholarship in Q1 income
    df_income = df.dropna(subset=['Family_Income']).copy()
    df_income['income_quartile'] = pd.qcut(df_income['Family_Income'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    q1_noschol = df_income[(df_income['income_quartile'] == 'Q1') & (df_income['Scholarship'] == 0)]['Target_Dropout_Next_Sem'].mean() * 100
    q1_schol = df_income[(df_income['income_quartile'] == 'Q1') & (df_income['Scholarship'] == 1)]['Target_Dropout_Next_Sem'].mean() * 100
    q1_red = float(q1_noschol - q1_schol)

    # Early advising drop
    early = df[df['Semester'] <= 2]
    adv0 = early[early['Advising_Visits'] == 0]['Target_Dropout_Next_Sem'].mean() * 100
    adv2 = early[early['Advising_Visits'] >= 2]['Target_Dropout_Next_Sem'].mean() * 100
    advising_drop = float(adv0 - adv2)

    return RetentionTippingPoints(
        gpa_brackets={str(k): float(v) for k, v in gpa_rates.items()},
        attendance_brackets={str(k): float(v) for k, v in att_rates.items()},
        course_load_hazard={int(k): float(v) for k, v in load_rates.items()},
        scholarship_q1_reduction=round(q1_red, 2),
        advising_early_drop=round(advising_drop, 2),
    )


def compute_placement_deep_insights() -> PlacementHiddenPatterns:
    """Extract hidden patterns, compensatory interactions, and equity dynamics from Dataset B."""
    logger.info("Computing Placement deep insights...")
    df = load_placement()
    df_clean = df.copy()
    df_clean['is_placed'] = (df_clean['status'] == 'Placed').astype(int)

    # Workex odds ratio
    logit_work = smf.logit("is_placed ~ C(workex) + degree_p + mba_p", data=df_clean).fit(disp=False)
    workex_or = float(np.exp(logit_work.params["C(workex)[T.Yes]"]))
    workex_pval = float(logit_work.pvalues["C(workex)[T.Yes]"])

    # Low GPA (<65%) Rescue Rate
    low_deg = df_clean['degree_p'] < 65.0
    rescue_workex = float(df_clean[low_deg & (df_clean['workex'] == 'Yes')]['is_placed'].mean() * 100)
    rescue_noworkex = float(df_clean[low_deg & (df_clean['workex'] == 'No')]['is_placed'].mean() * 100)

    # Specialization rates
    mkt_fin_rate = float(df_clean[df_clean['specialisation'] == 'Mkt&Fin']['is_placed'].mean() * 100)
    mkt_hr_rate = float(df_clean[df_clean['specialisation'] == 'Mkt&HR']['is_placed'].mean() * 100)

    # Undergrad stream salaries
    placed = df_clean[df_clean['status'] == 'Placed']
    sci_sal = float(placed[placed['degree_t'] == 'Sci&Tech']['salary'].mean())
    comm_sal = float(placed[placed['degree_t'] == 'Comm&Mgmt']['salary'].mean())

    # Gender salary Mann-Whitney U
    sal_m = placed[placed['gender'] == 'M']['salary']
    sal_f = placed[placed['gender'] == 'F']['salary']
    _, u_pval = stats.mannwhitneyu(sal_m, sal_f)

    # Board prestige Chi-square
    _, p_ssc, _, _ = stats.chi2_contingency(pd.crosstab(df_clean['ssc_b'], df_clean['status']))

    return PlacementHiddenPatterns(
        workex_odds_ratio=round(workex_or, 4),
        workex_p_value=float(workex_pval),
        rescue_rate_low_gpa_workex=round(rescue_workex, 2),
        rescue_rate_low_gpa_no_workex=round(rescue_noworkex, 2),
        mkt_fin_placement_rate=round(mkt_fin_rate, 2),
        mkt_hr_placement_rate=round(mkt_hr_rate, 2),
        sci_tech_salary_mean=round(sci_sal, 2),
        comm_mgmt_salary_mean=round(comm_sal, 2),
        gender_wage_gap_p_value=float(u_pval),
        board_prestige_p_value=float(p_ssc),
    )


def compute_cross_dataset_synthesis_metrics() -> SynthesisPipelineMetrics:
    """Compute the synthesis metrics linking Retention and Placement."""
    logger.info("Computing Cross-Dataset Synthesis metrics...")
    df_ret = load_retention()
    df_place = load_placement()
    df_place_clean = df_place.copy()
    df_place_clean['is_placed'] = (df_place_clean['status'] == 'Placed').astype(int)

    # Student labor paradox
    logit_work_ret = smf.logit("Target_Dropout_Next_Sem ~ Work_Hours + Sem_GPA + Financial_Stress", data=df_ret).fit(disp=False)
    ret_or = float(np.exp(logit_work_ret.params['Work_Hours']))
    ret_pval = float(logit_work_ret.pvalues['Work_Hours'])

    logit_place_work = smf.logit("is_placed ~ C(workex) + degree_p + mba_p", data=df_place_clean).fit(disp=False)
    place_or = float(np.exp(logit_place_work.params["C(workex)[T.Yes]"]))
    place_pval = float(logit_place_work.pvalues["C(workex)[T.Yes]"])

    # Safe GPA (75th percentile in retention)
    safe_gpa = float(np.percentile(df_ret['Sem_GPA'], 75))

    # Hiring threshold (65% in placement where placement jumps to 90%)
    hiring_thresh = 65.0

    return SynthesisPipelineMetrics(
        work_hours_retention_or=round(ret_or, 4),
        work_hours_retention_p_val=float(ret_pval),
        workex_placement_or=round(place_or, 4),
        workex_placement_p_val=float(place_pval),
        retention_safe_gpa_p75=round(safe_gpa, 2),
        placement_hiring_threshold_p=round(hiring_thresh, 2),
    )

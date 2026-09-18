# Basel III Framework: Credit Risk & Capital Adequacy Guidelines

## 1. Expected Credit Loss (ECL) Calculation
Under the Basel III Foundation Internal Ratings-Based (F-IRB) framework and IFRS 9 / CECL standards, Expected Credit Loss is computed on each loan exposure as:
$$\text{ECL} = \text{PD} \times \text{LGD} \times \text{EAD}$$

Where:
- **PD (Probability of Default)**: The empirical likelihood that an obligor will default within 12 months (estimated via calibrated XGBoost default classifier).
- **LGD (Loss Given Default)**: The economic net loss incurred if default occurs, accounting for recovery costs. Standard benchmark for unsecured consumer loans is 45.0% (0.45).
- **EAD (Exposure at Default)**: The gross committed loan balance at the time of default (Loan Amount $\times$ Approved Amount Fraction).

## 2. Portfolio Risk Metrics & Capital Buffers
- **Unexpected Loss (UL)**: Standard deviation of losses reflecting credit tail risk.
- **Value-at-Risk (VaR 99.9%)**: 99.9th percentile loss over a 1-year holding period defining minimum economic capital reserves.
- **ECL Budget Utilization**:
  $$\text{Budget Usage} = \frac{\text{Cumulative Portfolio ECL}}{\text{Configured Task ECL Budget}}$$
  - Usage > 80%: Tighten approval thresholds, switch borderline cases to COUNTER.
  - Usage > 95%: Hard brake on new credit extensions, approve only FICO > 750 with low default probabilities.

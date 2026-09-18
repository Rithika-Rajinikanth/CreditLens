# Institutional Bank Credit Underwriting Policy Manual

## 1. Credit Tier Classification & Minimum Requirements
- **Super-Prime Tier (FICO 750+)**:
  - Maximum Loan Amount: $100,000 unsecured.
  - Standard Approval: Approved at requested amount fraction 1.0.
  - Maximum Allowable Debt-to-Income (DTI): 48.0%.
- **Prime Tier (FICO 680-749)**:
  - Maximum Loan Amount: $75,000 unsecured.
  - Standard Approval: Approved at requested amount fraction 0.90 to 1.0 if DTI <= 40%.
  - Moderate DTI (40.1% - 45.0%): Approved at 0.85 amount fraction.
- **Near-Prime Tier (FICO 620-679)**:
  - Maximum Loan Amount: $50,000.
  - Counteroffer Mandate: Counter-offer required if DTI > 38% or credit utilization > 50%.
  - Standard Counter Terms: Amount fraction 0.70 to 0.80, rate delta +1.00% to +1.50%.
- **Subprime Tier (FICO < 620)**:
  - FICO < 580: Strict mandatory denial. Adverse Action Code: `LOW_CREDIT_SCORE`.
  - FICO 580-619: Allowable only with mitigating factors (e.g. low DTI < 30%, 0 derogatory marks). Must issue counter-offer of <= 0.65 amount fraction with +2.00% to +3.00% rate delta.

## 2. Capacity & Debt-to-Income (DTI) Hard Limits
- **Hard Ceiling**: Absolute DTI cap is 50.0%. Any application with DTI > 50.0% must be declined under reason code `HIGH_DTI`.
- **Borderline Zone (DTI 43.1% - 50.0%)**:
  - Unconditional approvals are strictly prohibited.
  - Mitigating recourse: Issue counter-offer reducing principal exposure by 20-35% (fraction 0.65-0.80) to bring imputed DTI below 40.0%.

## 3. Macroeconomic Rate Shock Response Protocols
- **Trigger Event**: Federal Reserve rate hikes or benchmark Treasury yield spikes (+50 to +100 basis points).
- **Underwriting Action Post-Shock**:
  - Tighten standard XGBoost default probability cutoff by 15 percentage points (e.g., from 0.62 down to 0.47).
  - Target minimum 70% of post-shock decisions to be conservative actions (`REJECT` or defensive `COUNTER`).
  - Cap maximum amount fraction on all variable rate loans at 0.75.

## 4. Counteroffer Framework & Negotiated Settlements
- A counteroffer is a conditional approval offered when an applicant demonstrates repayment intent but presents elevated credit risk at the requested loan amount.
- Minimum allowable revised amount fraction: 0.25 (25% of request).
- Maximum allowable revised amount fraction: 0.90 (90% of request).
- Allowable rate adjustment: +0.25% to +5.00% annual percentage rate delta.

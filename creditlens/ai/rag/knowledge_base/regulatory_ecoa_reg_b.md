# Equal Credit Opportunity Act (ECOA) & CFPB Regulation B Compliance Handbook

## 1. Statutory Mandate & Scope (12 CFR § 1002.9)
The Equal Credit Opportunity Act (ECOA), implemented by the Consumer Financial Protection Bureau's Regulation B (12 CFR Part 1002), prohibits creditors from discriminating against any applicant on the basis of race, color, religion, national origin, sex, marital status, or age.

## 2. Adverse Action Notice Requirements
- **Notification Timing**: A creditor must notify an applicant of action taken within 30 days of receiving a completed application.
- **Statement of Specific Reasons**: When adverse action is taken (denial, cancellation, or unfavorable counteroffer rejected by applicant), the statement must be specific and indicate the principal reasons:
  - No more than the top 4 principal reasons may be disclosed to avoid confusing the applicant.
  - Vague statements like "internal credit scoring system" or "policy guidelines" are illegal under 12 CFR § 1002.9(b)(2).
  - Standard compliant reason codes:
    1. `HIGH_DTI`: Debt-to-income ratio exceeds allowable limits.
    2. `LOW_CREDIT_SCORE`: Credit score below minimum qualification threshold.
    3. `INSUFFICIENT_INCOME`: Stated or verified cash flow insufficient for debt service.
    4. `FRAUD_SUSPECTED`: Synthetic identity inconsistencies or cluster anomaly.
    5. `RECENT_DEROGATORY`: Delinquency history, charge-offs, or derogatory public records.
    6. `INCOMPLETE_APPLICATION`: Unsubmitted verifications or missing proof of income.

## 3. Disparate Impact & The Four-Fifths (80%) Rule
- **Definition**: A facially neutral credit policy or algorithm that disproportionately excludes protected demographic groups violates ECOA unless justified by demonstrable business necessity with no less discriminatory alternative.
- **Adverse Impact Ratio (AIR)**:
  $$\text{AIR} = \frac{\text{Approval Rate}_{\text{Protected Group}}}{\text{Approval Rate}_{\text{Reference Group}}}$$
- **Threshold**: An AIR below 0.80 (80%) constitutes prima facie evidence of disparate impact requiring model audit and mitigating underwriting adjustments.

## 4. Model Form C-1 (Sample Notice of Action Taken)
Every adverse decision must include:
- Creditor identification and date.
- Description of credit requested and decision taken.
- Primary credit scoring disclosure (FICO score, scoring model, key factors).
- ECOA rights notice and regulatory agency contact information (CFPB / FTC).

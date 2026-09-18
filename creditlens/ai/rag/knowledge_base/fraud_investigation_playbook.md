# Fraud Intelligence & Synthetic Identity Ring Playbook

## 1. Synthetic Identity Fraud Mechanisms
Synthetic identity theft occurs when bad actors combine real and fabricated personally identifiable information (PII) — such as compromised Social Security Numbers paired with newly created names, disposable phone numbers, and fabricated corporate employer IDs.

## 2. Graph Analytics & Connected Component Thresholds
CreditLens utilizes graph-theoretic analysis (NetworkX) where applicants form nodes and shared contact credentials (phone numbers, employer tax IDs) form weighted edges:
- **Shared Phone Credential**: Edge weight = 0.50.
- **Shared Employer ID**: Edge weight = 0.50.
- **Fraud Ring Score Calculation**:
  Combines connected component cluster size, subgraph density, and incident edge weights into a normalized probability in $[0.0, 1.0]$.

## 3. Escalation & Action Directives
- **Fraud Ring Score > 0.70**: Conclusive synthetic fraud ring indicator. Immediate mandatory denial under reason code `FRAUD_SUSPECTED`.
- **Dual Shared Credentials (Shared Phone AND Shared Employer)**: High-confidence fraud ring collusion. Immediate mandatory denial under `FRAUD_SUSPECTED`.
- **Fraud Ring Score 0.40 - 0.70**: Ambiguous cluster. Issue `REQUEST_INFO` with `identity_document` or `tax_returns` to break synthetic credit build-up before extending credit.

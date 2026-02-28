# Sealed Evaluation NDA Clause Set (Template)

## Scope
This clause set governs temporary evaluation-only access to archive subsets in sealed environments.

## Required Clauses
1. Evaluation-only purpose: access is limited to inspection, benchmarking, and reporting.
2. Training prohibition: no model training, fine-tuning, or parameter updates are permitted.
3. No residual retention: no extracted corpus copies, embeddings, or feature stores may be retained after expiry.
4. No egress: external network transfer of subset content is prohibited.
5. Read-only mounts: provided subsets must be mounted read-only.
6. Time limit: access automatically expires at 72 hours.
7. Audit evidence: provider must retain process logs and pre/post fixity artifacts.
8. Breach presumption: missing or tampered audit evidence is treated as a breach condition.
9. Return/destruction: all temporary derivatives must be destroyed at term end.
10. Verification rights: disclosing party may request verification records and integrity reports.

## Audit Attachments
- Pre-run hash manifest and verification report.
- Post-run hash manifest and verification report.
- Sealed evaluation summary report.
- PREMIS event log excerpt for the evaluation window.

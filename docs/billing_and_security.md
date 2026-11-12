# Cloud FinOps Billing Policies & Encryption Standards

## 1. Mandatory Resource Tagging & Cost Allocation
Every AWS and Kubernetes resource must carry four mandatory FinOps tags enforced by AWS Config rules:
- `cost-center`: 5-digit department ledger code (e.g., `CC-40120`).
- `environment`: One of `dev`, `staging`, or `prod`.
- `owner-team`: Slack channel or engineering squad slug (e.g., `platform-core`).
- `data-classification`: `public`, `internal`, `confidential`, or `restricted`.
Untagged resources in `dev-sandbox` are automatically terminated after 24 hours by Cloud Custodian.

## 2. Cryptographic & Secrets Management Standards
- **Data at Rest:** AES-256 encryption using customer-managed AWS KMS keys rotated automatically every 365 days.
- **Data in Transit:** TLS 1.3 minimum for all external ingress endpoints; TLS 1.2 deprecated as of Q3 2026.
- **Secrets Storage:** Hardcoded credentials in Git repositories or `.env` files are strictly prohibited. Applications must fetch runtime secrets dynamically via External Secrets Operator (ESO) backed by AWS Secrets Manager.

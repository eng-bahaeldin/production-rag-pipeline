# Production & Staging Deployment Runbook

## 1. Standard CI/CD Promotion Pipeline
Every microservice build passes through four automated gates before reaching production:
1. **Static Analysis & SBOM:** `ruff`, `mypy --strict`, `trivy` container vulnerability scan (zero `CRITICAL` CVEs allowed), and CycloneDX SBOM attestation.
2. **Ephemeral Integration Testing:** Automated deployment into an isolated staging namespace with synthetic traffic replay.
3. **Canary Rollout (Argo Rollouts):** Production traffic shifts progressively (`5% -> 25% -> 50% -> 100%`) over 30 minutes.
4. **Automated Rollback Triggers:** Argo Rollouts automatically aborts and reverts to the previous ReplicaSet if HTTP `5xx` error rate exceeds `0.5%` for 2 consecutive minutes or p99 latency exceeds `450ms`.

## 2. Infrastructure-as-Code Provisioning Workflow
To apply infrastructure changes in `staging-preprod`:
- Clone the `infra-live-staging` repository and create a feature branch (`feat/vpc-isolation-<ticket>`).
- Initialize Terraform (`1.9.5`) with the remote S3 state backend and DynamoDB lock table (`tf-state-lock-staging`).
- Run `terraform plan -var-file="staging.tfvars" -out=tfplan` and attach the plan artifact to the pull request.
- Two senior platform engineers must approve the PR before Atlantis executes `terraform apply tfplan`.

## 3. Incident Response & Break-Glass Access
Production SSH/SSM sessions are disabled by default. During a Sev-1 outage, on-call engineers request temporary 60-minute `BreakGlassAdmin` elevation via PagerDuty + AWS IAM Identity Center, which automatically notifies `#sec-ops-audit` in Slack.

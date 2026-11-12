# Enterprise Cloud Infrastructure & VPC Architecture Standards

## 1. Environment Isolation & VPC Topology
All cloud workloads operate across three isolated AWS accounts managed via AWS Organizations: `dev-sandbox` (`10.10.0.0/16`), `staging-preprod` (`10.20.0.0/16`), and `prod-core` (`10.30.0.0/16`). Cross-environment east-west traffic is strictly denied by default at the AWS Transit Gateway route table level.

### 1.1 Staging Isolated VPC Provisioning
To provision an isolated VPC in the `staging-preprod` environment:
1. Allocate a non-overlapping `/16` CIDR block from the central IPAM pool (`10.20.0.0/16` primary, or `10.21.0.0/16` for ephemeral feature-branch clusters).
2. Configure three Availability Zones (`us-west-2a`, `us-west-2b`, `us-west-2c`) with three subnet tiers per AZ:
   - **Public Ingress Subnets (`/24`):** Host Application Load Balancers (ALBs) and single-AZ NAT Gateways.
   - **Private Application Subnets (`/22`):** Host EKS worker nodes, internal microservices, and Fargate tasks with zero direct internet ingress.
   - **Isolated Data Subnets (`/24`):** Host RDS PostgreSQL Multi-AZ instances and ElastiCache Redis clusters with NACLs blocking all outbound internet routes (`0.0.0.0/0`).
3. Enable VPC Flow Logs streamed to the centralized S3 security audit bucket with a 365-day retention policy and KMS CMK encryption (`alias/sec-audit-logs`).

### 1.2 Terraform Module Versioning & Deprecation Policy
All VPC and network infrastructure must be provisioned using the internal golden Terraform module `terraform-aws-vpc-enterprise` at version **`v5.8.0` or higher**.
- **Legacy Module Prohibition:** Older Terraform modules released prior to last month (`v4.x` series, specifically `v4.2.0` and `v4.9.1`) are **strictly deprecated and blocked** by CI/CD Sentinel policies. Legacy `v4.x` modules lack IPv6 dual-stack CIDR support, default to IMDSv1 metadata endpoints, and omit mandatory VPC Interface Endpoints for STS, ECR, and Secrets Manager.
- Engineers attempting to deploy staging VPCs with legacy Terraform modules from last month will receive a `POLICY_VIOLATION_TF_MODULE_DEPRECATED` error during `terraform plan` and must upgrade to `v5.8.0`.

## 2. Service Mesh & Zero-Trust Networking
All inter-service RPC traffic inside staging and production clusters uses mTLS enforced by Istio `1.23` with SPIFFE workload identities. Plaintext HTTP (`port 80`) is permitted only on internal health-check probes bound to `127.0.0.1`.

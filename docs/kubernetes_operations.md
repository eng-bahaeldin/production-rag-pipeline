# Kubernetes (EKS) Cluster & Worker Node Operations

## 1. Kubernetes Worker Nodes System Requirements
All managed node groups running on Amazon EKS `1.30` must meet the following baseline system requirements:
- **Minimum Compute & Memory:** `4 vCPU` and `16 GiB RAM` per worker node (`m6i.xlarge` or `c6i.xlarge` instances). Nodes below `16 GiB RAM` are prohibited due to DaemonSet overhead (Datadog agent, Istio Envoy proxy, Fluent Bit, and AWS EBS CSI driver consume ~2.8 GiB).
- **Operating System & Container Runtime:** Bottlerocket OS `1.20+` (preferred) or Amazon Linux 2023 with `containerd` `1.7.13+` and `kubelet` `1.30.x`.
- **Root & Ephemeral Storage:** Minimum `100 GiB` encrypted `gp3` EBS root volume (`3000 IOPS`, `125 MB/s` throughput) to prevent `DiskPressure` evictions during container image pulls.
- **Networking & CNI:** AWS VPC CNI `v1.18+` with prefix delegation enabled (`ENABLE_PREFIX_DELEGATION=true`), supporting up to `110` pods per `m6i.xlarge` node.
- **Kernel & Sysctl Tuning:** `vm.max_map_count=262144`, `net.core.somaxconn=32768`, and swap disabled (`swapoff -a`).

## 2. Pod Disruption Budgets & Autoscaling
- Production workloads require a minimum of `3` replicas spread across distinct Availability Zones via `topologySpreadConstraints`.
- Cluster scaling is managed by **Karpenter `v1.0`** using `NodePool` consolidation (`WhenEmptyOrUnderutilized`) with a `30s` consolidation delay.

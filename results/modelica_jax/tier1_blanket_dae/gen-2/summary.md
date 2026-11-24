# Evolution summary

| gen | run | lessons | score | cost_usd | seconds | agents | spawned_peers | subagents | peer_messages | talking_pairs | hub_share | tasks_posted | handovers | delegations | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | run-1 | yes | 0.0 | 0.1756 | 3600 | 3 | 0 | 0 | 8 | 3 | 0.88 | 1 | 0 | 0 | timeout after 3600.0s |
| 1 | run-2 | yes | 0.0 | 0.2024 | 3600 | 3 | 0 | 0 | 16 | 3 | 0.69 | 0 | 0 | 0 | timeout after 3600.0s |
| 1 | control | no | 0.0 | 0.1829 | 3600 | 3 | 0 | 0 | 12 | 3 | 0.83 | 2 | 1 | 0 | timeout after 3600.0s |

| gen | mean score with lessons | mean score without | mean cost |
|---|---|---|---|
| 1 | 0.0 | 0.0 | 0.19 |

`hub_share` is the share of peer messages involving the busiest agent (1.0 = one hub).
Each generation's lessons are in `gen-NN/lessons_out.md`; the latest are in `lessons.md`.

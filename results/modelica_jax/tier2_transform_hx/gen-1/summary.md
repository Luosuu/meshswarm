# Evolution summary

| gen | run | lessons | score | cost_usd | seconds | agents | spawned_peers | subagents | peer_messages | talking_pairs | hub_share | tasks_posted | handovers | delegations | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | run-1 | yes | 4.0 | 0.6504 | 3600 | 3 | 0 | 0 | 18 | 3 | 0.67 | 2 | 0 | 0 | timeout after 3600.0s |
| 1 | run-2 | yes | 9.0 | 0.8848 | 3600 | 3 | 0 | 0 | 32 | 3 | 0.78 | 0 | 0 | 0 | timeout after 3600.0s |
| 1 | control | no | 5.0 | 0.7833 | 3600 | 3 | 0 | 0 | 23 | 3 | 0.78 | 1 | 0 | 0 | timeout after 3600.0s |

| gen | mean score with lessons | mean score without | mean cost |
|---|---|---|---|
| 1 | 6.5 | 5.0 | 0.77 |

`hub_share` is the share of peer messages involving the busiest agent (1.0 = one hub).
Each generation's lessons are in `gen-NN/lessons_out.md`; the latest are in `lessons.md`.

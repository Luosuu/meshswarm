# Evolution summary

| gen | run | lessons | score | cost_usd | seconds | agents | spawned_peers | subagents | peer_messages | talking_pairs | hub_share | tasks_posted | handovers | delegations | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | run-1 | yes | 5.0 | 0.1908 | 3600 | 3 | 0 | 0 | 9 | 3 | 0.78 | 0 | 0 | 0 | timeout after 3600.0s |
| 1 | run-2 | yes | 0.0 | 0.2119 | 3600 | 3 | 0 | 2 | 15 | 3 | 0.8 | 3 | 0 | 2 | timeout after 3600.0s |
| 1 | control | no | 0.0 | 0.2044 | 3600 | 3 | 0 | 1 | 6 | 2 | 1.0 | 4 | 0 | 1 | timeout after 3600.0s |

| gen | mean score with lessons | mean score without | mean cost |
|---|---|---|---|
| 1 | 2.5 | 0.0 | 0.2 |

`hub_share` is the share of peer messages involving the busiest agent (1.0 = one hub).
Each generation's lessons are in `gen-NN/lessons_out.md`; the latest are in `lessons.md`.

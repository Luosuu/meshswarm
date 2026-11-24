# Evolution summary

| gen | run | lessons | score | cost_usd | seconds | agents | spawned_peers | subagents | peer_messages | talking_pairs | hub_share | tasks_posted | handovers | delegations | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | run-1 | yes | 2.0 | 0.3135 | 3600 | 3 | 0 | 0 | 37 | 3 | 0.7 | 2 | 0 | 0 | timeout after 3600.0s |
| 1 | run-2 | yes | 3.0 | 0.332 | 3600 | 3 | 0 | 0 | 27 | 3 | 0.74 | 4 | 0 | 0 | timeout after 3600.0s |
| 1 | control | no | 6.0 | 0.3563 | 3600 | 3 | 0 | 0 | 15 | 3 | 0.93 | 2 | 0 | 0 | timeout after 3600.0s |

| gen | mean score with lessons | mean score without | mean cost |
|---|---|---|---|
| 1 | 2.5 | 6.0 | 0.33 |

`hub_share` is the share of peer messages involving the busiest agent (1.0 = one hub).
Each generation's lessons are in `gen-NN/lessons_out.md`; the latest are in `lessons.md`.

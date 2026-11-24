# Evolution summary

| gen | run | lessons | score | cost_usd | seconds | agents | spawned_peers | subagents | peer_messages | talking_pairs | hub_share | tasks_posted | handovers | delegations | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | run-1 | yes | 10.0 | 0.9224 | 3600 | 3 | 0 | 0 | 22 | 3 | 0.73 | 1 | 0 | 0 | timeout after 3600.0s |
| 1 | run-2 | yes | 5.0 | 0.9795 | 3600 | 3 | 0 | 0 | 17 | 3 | 0.94 | 0 | 0 | 0 | timeout after 3600.0s |
| 1 | control | no | 8.0 | 1.0183 | 3600 | 3 | 0 | 1 | 15 | 2 | 1.0 | 4 | 0 | 1 | timeout after 3600.0s |

| gen | mean score with lessons | mean score without | mean cost |
|---|---|---|---|
| 1 | 7.5 | 8.0 | 0.97 |

`hub_share` is the share of peer messages involving the busiest agent (1.0 = one hub).
Each generation's lessons are in `gen-NN/lessons_out.md`; the latest are in `lessons.md`.

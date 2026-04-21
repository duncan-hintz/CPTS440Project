# PPO-Based AI Agent for Pokémon Doubles Battles
### poke-env + PettingZoo + CleanRL/PyTorch

---

## 1. Project Statement

### Problem & Motivation
Competitive Pokémon Doubles battling is a difficult sequential decision-making task with:
- **Partial information** — hidden opponent move sets, items, and abilities
- **Large action space** — up to 107 individual orders per slot × 2 slots = 11,449 combined actions per turn
- **Long-term strategy** — setup moves, weather control, team synergy, and switching sequencing

This project builds an RL agent that learns to play Gen 9 Doubles battles through PPO, targeting measurable win-rate improvement over rule-based baselines.

### Goal
Train and evaluate a Pokémon Doubles battle agent that reliably outperforms baseline opponents and demonstrates measurable learning progress over time.

### Algorithms & Tools
| Component | Choice |
|---|---|
| Core algorithm | PPO (CleanRL / PyTorch) |
| Battle environment | poke-env (`DoublesEnv`) |
| Multi-agent interface | PettingZoo Parallel API |
| Battle server | pokemon-showdown (local, `--no-security`) |
| Action masking | Custom `get_mask()` in `CustomEnv` |
| Reward shaping | `reward_computing_helper` (fainted×20, HP×10, victory×100, status×2) |

To make learning feasible and stable we are using:
- **Action masking** to eliminate illegal moves/switches
- **Reward shaping** on KOs, HP damage, status, and win/loss
- **Curriculum strategy** (planned) — start against random, graduate to heuristic, then self-play

### Battle Format
- **Primary:** `gen9randomdoublesbattle` — simplifies team-building, reduces team-selection variance
- **Secondary / Fixed-team mode:** `gen9doublesou` with a hardcoded team (toggleable via `fixed = True/False` in `cleanRL_implementation.py`)

### Observation Encoding (current — 18-dimensional float32 vector)
| Indices | Feature |
|---|---|
| 0–3 | Base power / 100 for slot 0's 4 moves |
| 4–7 | Base power / 100 for slot 1's 4 moves *(bug: currently overlaps at index 3 — see Known Bugs)* |
| 8–11 | Type effectiveness multiplier of slot 0's moves vs. first active opponent |
| 12–15 | Type effectiveness multiplier of slot 1's moves vs. first active opponent |
| 16 | Fraction of own team fainted (count / 6) |
| 17 | Fraction of opponent team fainted (count / 6) |

> **Planned expansions:** HP%, status conditions, stat boosts, weather/terrain/hazards, revealed opponent Pokémon info, PP tracking.

### Data Collection
Online RL — training data is generated live during battle simulation as trajectories of `(observation, action, reward, done)` tuples. No static dataset.

### Deliverables
1. Trained PPO agent playable in the same poke-env environment
2. Evaluation package: win-rate curves, ablation comparisons, baseline benchmarks
3. Final write-up with qualitative battle trace analysis

### Backup Plans
- If self-play is unstable → train against fixed scripted opponents with curriculum
- If learning is slow → behavior cloning pretraining on heuristic trajectories, then PPO fine-tune
- If action space is too large → restrict to moves-only first, then add switching

---

## 2. Current Implementation Status

### Repository Structure
```
CPTS440Project/
├── README.md                         — Setup/run instructions
├── spec.md                           — This document
├── env/
│   ├── customEnv.py                  ✅ Core PettingZoo env (239 lines)
│   └── testing.py                    ⚠️  Basic test harness (incomplete loop)
├── neural-net/
│   ├── cleanRL_implementation.py     ✅ Full PPO training loop (463 lines)
│   ├── NN_AlgoPlayer.py              ⚠️  Standalone inference player (missing return)
│   └── models/
│       └── testing/
│           └── agent.pt              ✅ Saved test model weights (1.9 MB)
└── pokemon-showdown/                 ❌ Git submodule — NOT initialized (empty)
```

### `env/customEnv.py` — `CustomEnv(DoublesEnv)`
- **Action space:** `Discrete(11449)` — flat product of two 107-slot individual-order spaces
- **`get_mask(battle)`** — builds valid 11,449-dim `int8` mask via `battle.valid_orders`, meshgrid combination, plus post-filtering of illegal combos (double-pass, same-target switch, simultaneous Tera)
- **`embed_battle(battle)`** — returns `{"observations": float32[18], "action_mask": int8[11449]}`
- **`calc_reward(battle)`** — delegates to `reward_computing_helper(fainted=20, hp=10, victory=100, status=2)`
- **`step(actions)`** — decodes flat index → `(a//107, a%107)` tuple, delegates to `DoublesEnv.step()`
- **`render()`** — opens browser on first call, prints formatted HP status every turn

### `neural-net/cleanRL_implementation.py` — PPO Training
- **`Agent(nn.Module)`** — MLP: `18→32→64→128→256→36→{actor: 11449, critic: 1}`, orthogonal init
- **Training config:** 1024 episodes, max 125 steps/ep, batch=32, 3 update epochs, lr=1e-4, γ=0.99, ent_coef=0.1, vf_coef=0.1, clip_coef=0.1
- **Opponent:** Agent 2 plays masked-random (`action_space.sample(mask=p2mask)`)
- **Advantage:** GAE-style backward pass *(non-standard: uses γ² instead of γλ — see Known Bugs)*
- **Post-training eval:** 1000 battles vs. `RandomPlayer`, win rate printed via `tabulate`
- **Model save path:** `./models/setup/agent.pt`

### `neural-net/NN_AlgoPlayer.py` — Standalone Inference Player
- Loads `agent.pt` weights, wraps inference in poke-env's `Player` callback interface
- Uses a dormant `CustomEnv()` instance solely for `embed_battle()` observation encoding
- **Incomplete** — missing final `return` statement in `choose_move()`

---

## 3. Known Bugs & Missing Pieces

| # | File | Issue | Priority |
|---|---|---|---|
| 1 | `env/customEnv.py` | `embed_battle`: `moves_base_power[3]` is overwritten by both slot 0 move 3 and slot 1 move 0; slot 1 indices should be `[4:8]` | **HIGH** |
| 2 | `neural-net/NN_AlgoPlayer.py` | `choose_move()` missing final `return DoublesEnv.action_to_order(action=actions, battle=battle)` | **HIGH** |
| 3 | `neural-net/cleanRL_implementation.py` | GAE advantage uses `gamma * gamma` instead of `gamma * lambda`; no separate λ hyperparameter | **MEDIUM** |
| 4 | `neural-net/cleanRL_implementation.py` | `team2` (fixed-format opponent) is empty string — will error in `gen9doublesou` mode | **MEDIUM** |
| 5 | `neural-net/cleanRL_implementation.py` | `battle._wait` handling is commented out — forced-switch/wait turns may cause crashes | **MEDIUM** |
| 6 | `env/testing.py` | `__main__` loop is commented out — only runs one episode | **LOW** |
| 7 | `pokemon-showdown/` | Git submodule not initialized — local server cannot start without `git submodule update --init` | **BLOCKER** |

---

## 4. Remaining Work

### Environment & Observation (Abdur)
- [ ] **Fix `embed_battle` index overlap** — slot 1 move features must occupy indices `[4:8]` and `[12:15]`
- [ ] **Expand observation vector** — add HP%, per-Pokémon status conditions, stat stage boosts, weather/terrain/hazard flags, opponent revealed info; update `observation_num` in the training script accordingly
- [ ] **Handle `battle._wait`** — re-enable or properly implement wait-turn / forced-switch logic in `Agent.get_action_and_value` and `CustomEnv.step`
- [ ] **Fix `NN_AlgoPlayer.py`** — add the missing `return` statement in `choose_move()`
- [ ] **Initialize submodule** — run `git submodule update --init` to bring up the local Showdown server

### PPO Training Pipeline (Duncan)
- [ ] **Fix GAE advantage** — introduce separate λ (`gae_lambda`) and use `gamma * gae_lambda * rb_advantages[t+1]`
- [ ] **Add logging** — replace print statements with TensorBoard or WandB metric tracking (loss curves, win rate, explained variance)
- [ ] **Hyperparameter sweep** — tune lr, ent_coef, clip_coef, batch_size, num_epochs; use a config file (argparse or hydra)
- [ ] **Multi-seed reproducibility** — run training with ≥3 seeds, report mean ± std win rate
- [ ] **Fix `team2` for fixed-format** — provide a valid Gen 9 Doubles OU team string for the opponent
- [ ] **Checkpoint cadence** — save model every N episodes, not only at end; support resume from checkpoint

### Baselines & Reward Shaping (Aden)
- [ ] **Implement heuristic baseline** — e.g., always pick the highest base-power move accounting for type effectiveness (max-damage greedy)
- [ ] **Implement curriculum schedule** — start opponent as `RandomPlayer`, switch to heuristic after win rate > threshold, then snapshot-based self-play
- [ ] **Tune reward weights** — run ablations: victory-only vs. shaped; document effect on learning speed and stability
- [ ] **Implement self-play snapshot opponent** — freeze current agent periodically as opponent; manage a snapshot pool

### Evaluation & Analysis (Felix)
- [ ] **Build evaluation harness** — standalone script: load checkpoint, run N battles vs. specified opponent, report win rate + 95% CI
- [ ] **Produce training curves** — win rate vs. episode, episodic return, value/policy loss
- [ ] **Ablation matrix** — shaped vs. unshaped reward × random vs. heuristic opponent × observation variants
- [ ] **Qualitative battle traces** — record 3–5 representative battles; annotate interesting decisions (switch reads, setup plays)
- [ ] **Optional Elo/mini-league** — round-robin between agent checkpoints + baselines; compute Elo ratings

---

## 5. Team Roles

| Member | Role | Responsibilities |
|---|---|---|
| **Duncan** | RL Lead / Training Pipeline | PPO integration, training loop, rollouts/GAE/checkpointing, hyperparameter sweeps, reproducibility |
| **Abdur** | Environment + Obs/Action Engineering | PettingZoo wrapper, observation encoding, action mapping, action masking, bug fixes |
| **Felix** | Evaluation + Analysis | Evaluation harness, metrics reporting, plots, ablation comparisons, final write-up results |
| **Aden** | Baselines + Reward Shaping / Curriculum | Heuristic opponents, reward shaping terms, curriculum schedule, training stability support |

---

## 6. Timeline

| Week | Dates | Focus | Status |
|---|---|---|---|
| 1 | 3/2–3/8 | Environment bring-up: poke-env + PettingZoo wrapper, action mapping/masking, end-to-end battle runs | ✅ Done |
| 2 | 3/9–3/15 | Observation encoding finalized; random + heuristic baselines; reproducible eval script | 🔄 In Progress |
| 3 | 3/16–3/22 | PPO first working version; stable training on small runs; initial learning curves + checkpoints | 🔄 In Progress |
| 4 | 3/23–3/29 | Reward shaping + stabilization; tune hyperparameters; clear improvement over random baseline | ⬜ Not Started |
| 5 | 3/30–4/5 | Evaluation + ablations harness; shaped vs. unshaped, obs variants, opponent variants | ⬜ Not Started |
| 6 | 4/6–4/12 | Self-play / snapshot-based league training (or fixed-opponent curriculum fallback) | ⬜ Not Started |
| 7 | 4/13–4/19 | Generalization tests on unseen seeds/teams; qualitative battle trace collection | ⬜ Not Started |
| 8 | 4/20–4/26 | Final training runs (multi-seed); finalize tables/figures; draft final write-up | ⬜ Not Started |
| Buffer | 4/27–4/30 | Polish, rerun missing experiments, finalize for submission before 5/1 | ⬜ Not Started |

---

## 7. End Results

We will deliver:
1. **Trained PPO agent** — playable in poke-env + PettingZoo, `agent.pt` loadable via `NN_AlgoPlayer`
2. **Evaluation package** — win-rate curves, ablation tables, baseline comparisons with confidence intervals
3. **Final report** — methodology, results, limitations (partial observability, action complexity, opponent diversity), and future work (recurrent policies, stronger self-play leagues, imitation learning pretraining)

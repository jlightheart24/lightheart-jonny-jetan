# Designing and Evaluating Jetan Agents

**Student:** [Jonny Lighthear]<br>
**Private repository:** [URL]<br>
**Access:** [Confirm `fractal13` has read access]<br>
**Submitted commit:** [Hash]

## 1. Deterministic Evaluation Functions

### Reading-Based Design Principles

Arthur L. Samuel, “Some Studies in Machine Learning Using the Game of Checkers,” IBM Journal of Research and Development, 3(3), 1959, pp. 210-229, https://doi.org/10.1147/rd.33.0210.

For this project I will be following the minimax procedure given by Samuel in the article. To make my deterministic evaluators I will be using two principles from the article.

First, I will be using the minimax procedure from the article. This is the same minimax procedure that was discussed during our class. Basically, it is having an agent make the most positive decision for themselves and the least positive decision for the opposing agent based on a determined tree.

Second, I will also be using the linear scoring polynomial as the basis for the leaf evaluation functions. Different parts of the game will be broken up and only consider their own values. (e.g., piece value, move distance, winning and moves that threaten winning) Then I will combine the values into a single number as an evaluation for the position.

I will not be using the rote learning mechanism from the article where the machine keeps a complete catalog of previously encountered positions, since I am already simplifying the problem to a 2-ply search.

### Evaluation 1

**Definition:** Piece Value: For a board and perspective player the total value of pieces will be calculated. Then the potential moves will be evaluated based on the difference in piece value. Here is an initial estimation of how that will be calculated: 
positive_total = weight(piece_type) over board.pieces(my potential pieces)
negative_total = weight(piece_type) over board.pieces(opponent potential pieces)<br>
**Information used and intentionally ignored:** This evaluation will only consider the value of my board and the opponent's board. No other factors will be considered at this point. <br>
**Scaling/range:** The value of each of the pieces will be weighted and the percentage of the full value of the total weight will be used to convert the value to a number between -1 and 1.<br>
**Expected relationship to utility:** Using this evaluator will be a good start though it definitely has flaws. Having a higher value board than the opponent does lead to better positions but this ignores potential tactical threats and is really weak against pieces that currently cannot be captured but will soon pose a threat.<br>
**Computational cost:** O(n) where n is the number of pieces on the board. <br>
**Known weakness:** Ignores position entirely. 

### Evaluation 2 and Motivation

**Definition:** Piece Mobility: This evaluator will positively value moving a piece forward. This will assign a positive value to your pieces moving forward and a negative value to the opponent's pieces moving forward. The evaluator will look something like this:
advancement_score = (my_adv - opp_adv) with my_adv = sum advancement(y,potential board)
then this position score will be combined with the material score from Evaluation 1, weighting each one. <br>
**Information used and intentionally ignored:** Adds each non-Princess piece's distance from the other end of the board. Ignores side-to-side mobility and potential threats.<br>
**Scaling/range:** These values will again be bounded by -1 and 1 then weighted with the number from Evaluation 1.<br>
**Expected relationship to utility:** Adds a bias to pushing pieces forward<br>
**Computational cost:** O(n) where n is number of pieces. Still O(n) when combined with evaluation 1, since it's just one more linear pass over the pieces, not a nested one. <br>
**Known weakness:** Again completely ignores tactics and potential threats. <br>
**Revision evidence:** Evaluator 1 would just get stuck moving back and forth and make no progress in the game. 

### Evaluation 3 and Motivation

**Definition:** Piece Threats: This evaluator adds on to evaluation 2 by giving value to pieces that are currently under attack. For each of my pieces sitting on a square the opponent could move to, that piece counts against me weighted by its piece value, and for each opponent piece sitting on a square I could move to it counts in my favor the same way. The board already tracks this with `attacked_locations(player)` so I just check which of my pieces/opponent pieces land in that set. The evaluator looks something like this:
threat_score = (opp_attacked_value - my_attacked_value)
then this gets combined with the material score from Evaluation 1 and the advancement score from Evaluation 2, each weighted. <br>
**Information used and intentionally ignored:** Adds whether a piece is currently sitting on a square the opponent attacks (and vice versa). Still ignores whose turn it actually is, so it can't tell if I get to move the threatened piece away before anything happens, and it ignores whether a threatened piece is actually defended. <br>
**Scaling/range:** Bounded the same way as the other two, between -1 and 1, then combined with Evaluation 1 and Evaluation 2's numbers using weights that add up to 1. <br>
**Expected relationship to utility:** Should make the agent avoid leaving pieces hanging and start noticing when it can threaten the opponent's pieces, which should turn into real material gains more often. <br>
**Computational cost:** O(n) where n is number of pieces, same as before, since `attacked_locations` is already being computed/cached by the board. <br>
**Known weakness:** Still static, it just looks at the current position, so it can overreact to a "threat" that I could just move away from or already have covered/defended, since it has no idea whose move it actually is. <br>
**Revision evidence:** Evaluation 2's known weakness was that it completely ignores tactics and potential threats, so pieces would get left hanging with no way for the evaluator to notice. Evaluation 3 directly fixes that by adding the threat term, at the cost of some of the material weight (material weight goes from 0.7 down to 0.5 to make room for it).

### Hand-Checked Positions and Tests

| Position | Perspective | Eval 1 | Eval 2 | Eval 3 | Why these values are reasonable |
|---|---|---:|---:|---:|---|
| Initial board (`JetanBoard.initial()`) | Orange | 0.000 | 0.000 | 0.000 | Both armies are identical, so a material-only evaluator (Eval 1), a material-plus-advancement evaluator (Eval 2, since no piece has moved yet), and Eval 3 (since no piece is attacked yet — any threats present are mirror-symmetric between the two sides) must all return exactly 0 for either side. |
| Orange's Chief removed from the initial board (all other pieces unchanged) | Orange | −0.102 | −0.071 | −0.051 | Orange is down its Chief (weight 5 of 49), so Eval 1 correctly returns a fixed negative score of −5/49. Eval 2 returns a smaller-magnitude negative score (−0.071 = 0.7 × −0.102) because the advancement term contributes 0 here (no piece has moved) and material is only weighted 0.7 in Eval 2's formula. Eval 3 (−0.051 = 0.5 × −0.102 + 0.2 × 0 + 0.3 × 0) further dampens the material term to 0.5 to make room for the threat term, which also contributes 0 here since no piece has moved into an attacking position yet — none of this is a bug, each revision intentionally reallocates weight away from material as new terms are added. |
| After Orange's opening move b1–b2 (one Panthan advances one row; Black has not moved) | Orange | 0.000 | +0.0018 | +0.0012 | Material is still tied, so Eval 1 is indifferent between this move and any other non-capturing first move (its known weakness). Eval 2 correctly breaks the tie in favor of the position that advanced a piece, returning a small positive value; from Black's perspective the same position scores −0.0018, confirming the term is antisymmetric as designed. Eval 3 also breaks the tie in Orange's favor (+0.0012), combining a smaller-weighted advancement contribution (0.2 × 0.0018) with a small threat contribution from the moved Panthan now covering a different square; Black's mirrored perspective again scores the exact negation (−0.0012). |

## 2. LLM Cutoff Evaluator

**Exact model tag:** `Gemma-4-26B-A4B-it-oQ4e-mtp`<br>
**Endpoint category:** Course-hosted OpenAI-compatible endpoint (`http://golem:8000/v1`), not a public provider. No API key value is recorded here.<br>
**Temperature:** 0<br>
**Requested seed and observed reproducibility:** Requested seed is 0 (the client default). I ran the identical request against the live endpoint 3 times back-to-back (same board, same seed=0, same temperature=0): all 3 produced the identical final score (`{"score": 0}`), so the final answer is reproducible. The number of reasoning tokens needed to get there was not perfectly stable though (319, 319, 322 completion tokens across the 3 runs) — small run-to-run variance even under nominally deterministic settings.<br>
**Deterministic fallback evaluator:** `evaluate_position_3`, my best deterministic evaluator from Stage A1.

### Prompt and Response Contract

`build_evaluation_prompt` (in `student_strategies.py`) builds three chat messages. The system message locks down the response format and tells the model not to reason:

```
You output only raw JSON, no reasoning, no explanation, no thinking, no markdown.
The response must match this exact schema: {"score": NUMBER}
NUMBER must be a plain finite number between -1 and 1, from the requested
player's perspective: 1 means that player is winning, -1 means that player is
losing, and 0 means the position is even.
/no_think
```

The user message gives the model a compact piece list for each side (e.g. `Warrior@a0, Padwar@b0, ...`) instead of a rendered board grid, states whose turn's perspective the score should be from, and repeats the `{"score": NUMBER}` format. A third, `assistant`-role message seeds the response with `{"score": ` so the model continues from that point instead of starting a fresh turn — this is a genuine response-shortening trick, not just a formatting choice (see Prompt Revision below).

On the response side, `parse_evaluation_response` re-attaches that same prefix to whatever the model returned, then parses strictly:
- It parses with `json.loads` using a custom `object_pairs_hook` that raises if the same key shows up twice (duplicate-key rejection), instead of quietly keeping whichever one came last like the default parser would.
- It requires the parsed object to have exactly one key, `"score"` — no missing key, and no extra keys either.
- It checks the value is an actual number (not a bool, since `bool` is technically an `int` in Python) and that it's finite (`math.isfinite`).
- It requires the number to fall inside `[-1, 1]`, matching the range I told the model to use, and rejects anything outside that range instead of clamping it.

Any of those checks failing raises a `ValueError`, and I don't have to do anything else about it — `LLMEvaluationFunction` (the supplied framework code) already catches any exception from the prompt/parse step and automatically calls my fallback evaluator instead, so a bad or malformed response just quietly costs one model call and falls back to `evaluate_position_3` rather than crashing the search.

### Prompt Revision

| Version | Representative response and limitation | Exact change | Evidence after change |
|---|---|---|---|
| Initial | System prompt described the schema in prose; user message rendered the board as the full 10x10 ASCII grid (via `JetanBoard.__str__`). Representative failure: the model produces long `reasoning_content` describing/re-deriving the grid ("*Game: Jetan (Martian Chess). *Board Size: 10x10...") and never reaches the `content` field — `finish_reason=length`, `content=None` — **even at 8192 completion tokens**, the largest budget I tested. | Replaced the ASCII grid with a compact `Name@col-row` piece list (`_piece_list` in `student_strategies.py`) and shortened the system prompt to a blunt no-reasoning instruction with a trailing `/no_think` directive. | On the initial board, this alone brought convergence down from "never" to ~600-1200 completion tokens depending on position complexity (still `content=None` at 128 tokens, but now finite instead of unbounded). |
| Revised | Piece-list format converged, but still needed 600-1200 tokens — far above the framework's fixed 128-token cap (`play.py` hardcodes `max_tokens=128`, which I can't change since it's supplied framework code), so real matches would still fall back almost every time. | Added a trailing `assistant`-role message seeding the response with `{"score": ` (`_RESPONSE_PREFIX`), so the server continues generation from that point instead of a fresh turn. I confirmed this only works when the prefix ends cleanly (right after `": "`) — committing an actual digit/sign (e.g. `{"score": 0.5`) made the server ignore the seed entirely and reason from scratch, so I kept the prefix minimal on purpose. | Measured completion tokens dropped from ~750 (piece-list alone) to ~319-322 on the initial board, confirmed consistent across 3 repeated live calls. Still short of 128 in absolute terms, but it's the best result across every variant I tested (grid, piece-list, `/no_think` alone, budget-awareness instructions, deeper prefills) — see the live Stage B data below for the real-world consequence of that remaining gap. |

## 3. Direct LLM Move Agent

### Prompt, Sensors, and Response Contract

`build_move_prompt` (in `student_strategies.py`) applies the same design I already validated in Section 2 (compact piece list, `/no_think`, `assistant`-message prefix), since the underlying model behavior — it needs far more than 128 tokens to reason about anything beyond restating a given fact — is the same regardless of whether the task is scoring a position or choosing a move. The system message: `{"move": "SRC-DST"}` schema, no reasoning, `/no_think`. The user message includes:
- Both sides' pieces as a compact list (same `_piece_list` helper as the evaluator).
- Whose turn it is.
- **Princess escape state** for both players (`board.used_escape`), so the model knows whether either Princess still has her one-time board-wide escape available.
- Recent move history (oldest first, up to `history_limit=6` moves, supplied by the framework).
- The exact list of legal moves in `SRC-DST` notation (`str(Move)`), and an instruction to pick one from that list.
- A trailing `assistant`-role message seeding `{"move": "` for the same token-reduction reason as Section 2.

`parse_move_response` re-attaches that prefix, parses strictly (duplicate-key rejection, exactly one `"move"` key, value must be a string), then looks the resulting string up in a `{str(action): action}` dictionary built from the actual legal `actions` tuple — a hallucinated or malformed move string simply isn't a key, so it raises and falls back, exactly like an invalid score does in Section 2.

`choose_fallback` is not "pick the first legal move" — it's a genuine one-ply greedy search: for each legal action, evaluate the resulting board with `evaluate_position_3` (or `board.utility` if it's terminal), and take the best one. This is the same computation `DepthLimitedMinimaxAgent` does internally at depth 1, so the fallback plays at roughly `minimax_3` strength rather than randomly.

### Prompt Revision

| Version | Representative response and limitation | Exact change | Evidence after change |
|---|---|---|---|
| Initial | Not tested as a separate first draft — Section 2 already established (across grid vs. piece-list, several system-prompt phrasings, and several board sizes down to 2 pieces) that this model needs far more than 128 tokens for any real judgment task, only ever finishing in-budget when asked to simply restate a given fact. Move selection among 50+ options is at least as demanding a judgment task as scoring one position, so I built directly on that finding instead of re-discovering it. | — | A live test of the piece-list format at a generous 2048-token budget still hit my diagnostic client's 120s timeout without finishing, confirming move selection is at least as expensive as, and likely more expensive than, position scoring. |
| Revised | (see above; this is the version actually used) | Piece list + `/no_think` + `assistant`-prefix seeding `{"move": "`, applying Section 2's revisions from the start rather than repeating the grid-format mistake. | Real Stage B data (`results/stage-b.csv`, `llm_direct`, 4 matches): mean 6.5 model calls and 5.75 fallback calls per match — i.e. roughly 0.75 of 6.5 attempts (~11.5%) returned a usable model-chosen move, noticeably better than the evaluator's ~0% success rate in the same run, though the large majority of moves still come from `choose_fallback`. |

Note: the Princess escape-state line was added to the user message as a final
refinement after the Stage B run above, to fully satisfy the sensor set this
section requires. It only adds one line of context and does not change the
schema, parsing, or fallback mechanism, so I don't expect it to change the
success rate — but the exact numbers above were produced by the version
without that line. Stage C will run with the complete version described here.

## 4. Stage A: Deterministic Evaluator and Depth Development

### Stage A1: Evaluation Functions

**Depth:** 1<br>
**Opponent:** supplied random agent<br>
**Seeds:** 0, 1<br>
**Colors:** both<br>
**Maximum plies:** 100<br>
**Other command/configuration details:** `python run_experiments.py a1 --output results/stage-a1.csv`, run through the project's Python 3.12 virtualenv.<br>
**Depth verification from `maximum_depth`:** Every raw row in `results/stage-a1.csv` reports `maximum_depth=1` for the focal agent, confirming the search never exceeded the configured depth-1 cutoff.

For every table below, “Mean think time” means the focal tested agent's
color-specific cumulative time: `orange_time` when it is Orange and `black_time`
when it is Black. Retain all scheduled attempts. Mark unavailable metrics as
`N/A`, never as zero, and state the contributing attempt count for every mean.

| Agent | Games | W | D | L | Mean utility | Mean plies | Mean think time | Mean generated actions | Mean evaluated states | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `minimax_1` | 4 | 2 | 2 | 0 | 0.5 | 69.5 | 0.1559s | 1900.75 | 1900.75 | 0 |
| `minimax_2` | 4 | 2 | 2 | 0 | 0.5 | 61.0 | 0.1702s | 2005.5 | 2005.5 | 0 |
| `minimax_3` | 4 | 4 | 0 | 0 | 1.0 | 8.0 | 0.0397s | 256.0 | 256.0 | 0 |

All means above are computed over all 4 contributing attempts (denominator = 4
for every metric; see `results/stage-a1-summary.csv`).

### Provisional Evaluator Selection

**Selected: `minimax_3` (Evaluation 3).** By utility first: it is the only
evaluator with a perfect record (4W-0D-0L, mean utility 1.0) against random,
while `minimax_1` and `minimax_2` each only manage 2W-2D-0L (mean utility 0.5)
— `minimax_3` strictly dominates on the primary performance measure. Resource
evidence reinforces the same conclusion rather than complicating it:
`minimax_3` also wins in far fewer plies (mean 8.0 vs. 61-69.5), with far less
think time (0.040s vs. 0.156-0.170s) and roughly 87-88% fewer generated/evaluated
states (256 vs. ~1900-2000) per match. The threat-awareness term added in
Evaluation 3 (over Evaluation 2's material-plus-advancement) appears to let it
punish random's undefended pieces quickly rather than needing to grind out a
slow positional edge, which is consistent with the large gap in plies-to-win.

### Stage A2: Depths 1 and 2

**Selected evaluator:** `minimax_3` (Evaluation 3)<br>
**Opponent:** supplied random agent<br>
**Seeds:** 0, 1<br>
**Colors:** both<br>
**Maximum plies:** 100<br>
**Depth verification from `maximum_depth`:** Every raw row in `results/stage-a2.csv` reports `maximum_depth=2` for the focal agent, confirming the depth-2 search actually reached its configured cutoff.

| Depth | Games | W | D | L | Mean utility | Mean plies | Mean think time | Mean generated actions | Mean evaluated states | Failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4 | 4 | 0 | 0 | 1.0 | 8.0 | 0.0397s | 256.0 | 256.0 | 0 |
| 2 | 4 | 4 | 0 | 0 | 1.0 | 8.5 | 2.9163s | 17144.75 | 16870.25 | 0 |

(Depth 1 row reused from Stage A1's `minimax_3` result, per `results/stage-a1-summary.csv`; depth 2 row from `results/stage-a2-summary.csv`.)

**Match segment where depth changed the actual moves played:** Orange vs. random, seed 0. Both depths reached from the identical starting position and identical opening Black reply (`f9-d7`, since the random agent is seeded). At depth 1, Orange's second move was `a3-d6`; at depth 2, from the same resulting position, Orange instead played `g0-d3` — a different piece and destination entirely. Depth 1 went on to win in 11 plies; depth 2, starting from that diverging second move, won the same matchup in only 7 plies.

**Effectiveness and decision-cost tradeoff:** Aggregated over all 4 games, depth 2 does not actually win more often or with a better mean utility than depth 1 — both are already a perfect 4-0-0 against random. In the one game above, depth 2 did find a faster win, but averaged across all 4 games its mean plies-to-win is slightly *higher* (8.5 vs. 8.0), so the single-game speedup isn't a consistent pattern. What depth 2 costs is unambiguous and large: ~73x the think time (2.916s vs. 0.040s) and ~67x the generated/evaluated states (~17,000 vs. 256) per match, for a random opponent that depth 1 was already beating every time. Against this specific weak opponent, depth 1 gets effectively all of the achievable benefit at a small fraction of the cost.

### Final Deterministic Configuration

**Evaluator:** `minimax_3` (Evaluation 3)<br>
**Depth (`1` or `2`):** 1<br>
**Selection justification using the supplied performance priorities:** Utility is the first priority, and depth 1 and depth 2 are tied on it (both 4W-0D-0L, mean utility 1.0) — depth 2 offers no measurable improvement in the one metric that matters most. With utility tied, resource cost becomes the deciding factor, and depth 1 wins decisively there: ~73x less think time and ~67x fewer generated/evaluated states per match. Depth 1 is therefore the better configuration under the stated priorities — it achieves the same outcome for a small fraction of the computational cost.

## 5. Stage B: Final Modality Comparison

**Seeds:** 0, 1<br>
**Colors:** both<br>
**Maximum plies:** 60<br>
**Cumulative think-time limit per player:** 3600 seconds or documented override<br>
**Selected deterministic depth:** 1<br>
**LLM cutoff-evaluator depth:** 1<br>
**LLM-evaluator request limit:** 100 per move search<br>
**Commands and machine/runtime context:** `python run_experiments.py b --evaluator minimax_3 --depth 1 --output results/stage-b.csv --max-think-seconds 3600`, run through the project's Python 3.12 virtualenv against the live course endpoint. Total wall time was approximately 120 minutes (2 hours), summed directly from the 12 matches' recorded think times in `results/stage-b.csv`; almost all of that (~117 of 120 minutes) came from the 4 `llm_evaluator` matches alone. For this section I did a lot of testing with Claude before running the experiment. I first ran tests with a 360-second time limit to verify the functionality of the program and that the LLM evaluator was losing due to think time, not gameplay.<br>
**Depth verification from `maximum_depth`:** Raw rows in `results/stage-b.csv` report `maximum_depth=1` for both `minimax_3` and `llm_evaluator`; `llm_direct` has no depth field since `DirectLLMAgent` does not run a search.

| Agent | Depth | Games | W | D | L | Mean utility | Mean plies | Mean think time | Mean generated actions | Mean evaluated states | Termination reasons/failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Selected deterministic minimax (`minimax_3`) | 1 | 4 | 4 | 0 | 0 | 1.0 | 8.0 | 0.0398s | 256.0 | 256.0 | all `win`, 0 failures |
| LLM cutoff evaluator (`llm_evaluator`) | 1 | 4 | 4 | 0 | 0 | 1.0 | 8.0 | 1757.93s | 256.0 | 256.0 | all `win`, 0 failures |
| Direct LLM move agent (`llm_direct`) | N/A | 4 | 4 | 0 | 0 | 1.0 | 12.5 | 42.07s | N/A | N/A | all `win`, 0 failures |

| Mean per-match LLM metric | Cutoff evaluator | Direct move agent | Denominator |
|---|---:|---:|---|
| Model calls | 255.0 | 6.5 | All 4 matches |
| Fallback calls | 255.0 | 5.75 | All 4 matches |
| Cache hits | 0.0 | N/A | All 4 matches |

All figures above come from `results/stage-b-summary.csv` with denominator 4
(every scheduled match completed with a result; no failures to exclude).

**The headline finding: `llm_evaluator`'s decisions are indistinguishable from `minimax_3`'s.** Both won all 4 matches with identical mean plies (8.0) and identical mean generated/evaluated state counts (256.0) — and per-match, the ply counts match exactly move-for-move (11, 6, 7, 8 for both). This is not a coincidence: `mean_model_calls` (255.0) equals `mean_fallback_calls` (255.0), meaning essentially every single live model call failed validation and fell back to `evaluate_position_3` — the same function `minimax_3` calls directly. `llm_evaluator` is functionally running `minimax_3`'s exact algorithm, just at ~44,000x the wall-clock cost (1757.93s vs. 0.0398s mean think time) because of the ~255 wasted live calls per match.

**The two LLM modalities do not do comparable amounts of work.** `llm_evaluator` is called up to 100 times *per move* (it must score every candidate leaf at the search's cutoff), while `llm_direct` is called exactly once per move to choose a full move outright. That is why `llm_evaluator` racks up a mean of 255 calls per match while `llm_direct` only needs 6.5 — they are not doing "the same job with different success rates," they are doing fundamentally different amounts of model work, and `llm_evaluator`'s cost scales with branching factor while `llm_direct`'s does not.

**`llm_direct`'s win record is mostly attributable to its fallback, not the model.** With only ~0.75 of 6.5 calls per match (~11.5%) succeeding, most of its moves come from `choose_fallback`, which is itself a `minimax_3`-strength one-ply greedy evaluator — so its 4-0-0 record is real evidence that our fallback design is strong, not strong evidence that the live model is choosing good moves.

**Depth is not actually unequal here, but it doesn't distort the comparison in this run.** The selected deterministic agent and the LLM cutoff evaluator are both configured at depth 1, so there's no depth mismatch to account for in this particular Stage B run — worth noting explicitly since the instructions ask for it, but the more consequential asymmetry this run actually demonstrates is the unequal call budget (100/move vs. 1/move) and its wildly different cost/outcome trade-off, not depth.

## 6. Stage C: Deterministic Versus Direct LLM

**Deterministic evaluator:** `minimax_3` (Evaluation 3)<br>
**Deterministic depth:** 1<br>
**Direct LLM prompt version:** The "Revised" version from Section 3 (piece list + `/no_think` + `assistant`-prefix seeding `{"move": "`, including the Princess escape-state line) — this is the final version and no further prompt changes are planned before Stage C runs.<br>
**Model seeds:** 0, 1<br>
**Colors:** both<br>
**Maximum plies:** 60<br>
**Cumulative think-time limit per player:** 3600 seconds or documented override<br>
**Depth verification from `maximum_depth`:** Every raw row in `results/stage-c.csv` reports `maximum_depth=1` for `minimax_3`.<br>
**Commands and machine/runtime context:** `python run_experiments.py c --evaluator minimax_3 --depth 1 --output results/stage-c.csv`, run through the project's Python 3.12 virtualenv against the live course endpoint. Fast overall — well under a minute of real model-call time across all 4 matches, since `llm_direct` makes only 1 call per move.<br>
**Seed/color configuration evidence:** In every one of the 4 raw rows, `minimax_3` is assigned a seed value (0 in three rows, its configured seed is irrelevant either way), and its `maximum_depth` and outcome pattern are identical regardless of that value. This matches the code directly: `DepthLimitedMinimaxAgent` (`minimax_agent.py`) never takes or reads a seed parameter at all — it's a pure deterministic search, so no configured seed could possibly change its behavior.

`llm_direct` fell back on **every single call in every match** (5 of 5 model calls per match), so this entire Stage C run was effectively `evaluate_position_3`-as-one-ply-greedy (`llm_direct`'s fallback) playing against `evaluate_position_3`-as-depth-1-minimax (`minimax_3`) — almost exactly the near-mirror match predicted going in.

| Direct LLM color | Model seed | Deterministic utility | Direct LLM utility | Plies | Termination reason/failure |
|---|---:|---:|---:|---:|---|
| Orange | 0 | +1 | −1 | 10 | win |
| Orange | 1 | +1 | −1 | 10 | win |
| Black | 0 | −1 | +1 | 10 | win |
| Black | 1 | −1 | +1 | 10 | win |

Record each utility from the named agent's perspective: `+1` for that agent's
win, `0` for a draw, and `-1` for that agent's loss. Compute each aggregate W/D/L
record and mean utility from the perspective of the agent named in that row.

**Every match was decided by color, not by which agent was playing it:** whichever agent played Black won, in all 4 matches, regardless of whether that was `minimax_3` or `llm_direct`. Both agents finish with an identical 2W-0D-2L record and mean utility 0.0 — a dead-even split.

| Agent | Games | W | D | L | Mean utility | Mean own-agent think time | Mean generated actions | Mean evaluated states | Mean model calls | Mean fallback calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Selected deterministic minimax | 4 | 2 | 0 | 2 | 0.0 | 0.0786s | 277.0 | 277.0 | N/A | N/A |
| Final direct LLM | 4 | 2 | 0 | 2 | 0.0 | 32.73s | N/A | N/A | 5.0 | 5.0 |

Use each agent's color-specific time. Mark unavailable values as `N/A` and
cite the generated denominator for each mean. Explain one position or match
segment that helps account for the head-to-head result. Limit conclusions to
this pairing, these seeds, and these match conditions.

**Match segment (Orange = `llm_direct`, Black = `minimax_3`, seed 0, reproducible via `play.py --orange-agent llm_direct --black-agent minimax_3 --orange-depth 1 --black-depth 1 --orange-seed 0 --black-seed 0 --live --view text`):** The opening mirrors exactly — Orange plays `d0-a3`, Black replies with the mirrored `d9-a6`, which is expected since both sides start from an identical, symmetric position and are (in this run) both effectively evaluating with `evaluate_position_3`. The decisive moment comes at ply 3: after Orange advances its Dwar to `d6` (ply 2), Black's Flier jumps `g9-d6`, **capturing that Dwar outright**. `evaluate_position_3`'s threat term is designed to penalize a piece sitting on a square the opponent attacks, but at only 0.3 of the total weight it wasn't enough to override whatever advancement/material considerations made `d6` still look attractive to the one-ply-greedy fallback that chose it — a concrete instance of the "known weakness" already written up in Section 1 for Evaluation 3. From there, Black converts the material lead, and the match ends at ply 9 when Black's piece reaches `f0` — **Orange's own Princess square** — capturing the Princess for the win. This is a clean, complete account of how the match was actually decided: not a difference in overall strategy quality (both sides ran the same evaluator), but a single one-ply blind spot exploited three plies in.

**Bounded interpretation:** With only 4 matches, 2 seeds, and one deterministic evaluator/depth pairing, this cannot support a general claim like "Black wins in Jetan" or "`minimax_3` beats `llm_direct`" — the clean color split is consistent with a first-move/tempo effect specific to this exact opening and evaluator, but 4 games is nowhere near enough to distinguish that from coincidence. It also cannot support a claim about the *live model's* move-choice quality at all, since `llm_direct` fell back on 100% of its calls in every match — everything observed here is a property of `evaluate_position_3` playing against itself in two different search shapes (one-ply greedy vs. depth-1 minimax), not a property of the LLM.

## 7. Analysis Questions

1. I think `minimax_3` (Evaluation 3) at depth 1 gives the strongest evidence
   of playing well. It's the only evaluator with a perfect record in Stage A1
   (4W-0D-0L, mean utility 1.0), and it stayed perfect across every single
   seed and color, not just on average. `minimax_1` and `minimax_2` both only
   manage 2W-2D-0L (mean utility 0.5), and interestingly they split by color
   instead of just being consistently okay: `minimax_1` only wins as Black
   and only draws as Orange, and `minimax_2` is the exact opposite. That
   tells me those two evaluators aren't actually robust, they just happen to
   be decent from one side.

   For the revisions: going from Evaluation 1 to Evaluation 2 (adding the
   advancement term) didn't change the overall record at all (0.5 to 0.5),
   but it did change how the games played out — `minimax_2` wins faster as
   Orange (61.0 vs. 69.5 mean plies). That's a real effect, just a small one.
   Going from Evaluation 2 to Evaluation 3 (adding the threat term) was a
   much bigger jump: perfect utility, winning in 8.0 plies on average instead
   of 61-69.5, and using way fewer generated/evaluated states (256 vs.
   ~1900-2000). That's not just a small improvement, it's a completely
   different level of play.

   For depth: depth 2 ties depth 1 on utility (both 4-0-0, both 1.0) but
   costs about 73x more think time and 67x more generated/evaluated states,
   and its average plies-to-win is actually a little worse (8.5 vs. 8.0). In
   one specific game (Orange, seed 0), depth 2 did find a faster path by
   picking a different second move (`g0-d3` instead of `a3-d6`) and won in 7
   plies instead of 11, so depth clearly can change the outcome of a specific
   game. But averaged out over all 4 games, it isn't worth the extra cost
   here. Depth 1 gets basically all of the benefit for a lot less
   computation.

2. The two prompt revisions (switching from the raw board grid to a piece
   list, then adding the `assistant`-message prefill) made a real difference
   in how often the model gave back something I could actually parse — going
   from never finishing even at 8192 tokens, down to consistently finishing
   around 319-322 tokens. But neither revision got it under the framework's
   fixed 128-token limit (`play.py` hardcodes `max_tokens=128`, which I can't
   change), so in real matches the model still fails almost every single
   call.

   That means the fallback policy ends up mattering a lot more than the
   prompt for what "LLM utility" actually shows. Since `choose_fallback` and
   the evaluator's fallback (`evaluate_position_3`) are both strong,
   deterministic, one-ply-greedy policies, and the model fails almost every
   time, the reported win rate for `llm_evaluator`/`llm_direct` is basically
   just measuring how good the fallback is, not the model. This isn't a small
   effect either — `llm_evaluator`'s Stage B results are identical to
   `minimax_3`'s move-for-move (same plies, same generated/evaluated counts),
   because `mean_model_calls` equals `mean_fallback_calls` exactly (255.0
   both) — every single real call failed.

   On reliability though, the actual parsing/validation code did exactly
   what it was supposed to: duplicate-key rejection, requiring exactly one
   key, checking the range — zero crashes and zero bad values ever made it
   into the game across the whole run. So the validation is reliable even
   though the model almost never gives it something valid to work with.
   Basically: before trusting any LLM-modality utility number, check the
   model-calls vs. fallback-calls ratio first.

3. All three modalities went a perfect 4-0-0 against random in Stage B, with
   no difference across seeds or colors — but they got there for totally
   different reasons. `minimax_3` won with its own real depth-1 search.
   `llm_evaluator` basically won by accident: with 255 of 255 mean model
   calls failing every match, it fell back to `evaluate_position_3` on every
   single evaluated leaf, so it was really just playing `minimax_3`'s exact
   algorithm — you can see this directly since their per-match plies match
   exactly (11, 6, 7, 8 for both). There weren't any real strategic
   disagreements between them because there wasn't an independent strategy
   running in the first place.

   `llm_direct` is the one case where the live model actually contributed
   something. Only about 88.5% of its calls fell back (5.75 of 6.5 mean per
   match), and its mean plies (12.5) are different from `minimax_3`'s (8.0),
   which fits with the idea that its rare real model-chosen moves sometimes
   led the game somewhere different than pure greedy play would have —
   though I don't have a per-ply diff to say exactly which moves those were.

   Depth wasn't actually unequal in this run (`minimax_3` and `llm_evaluator`
   are both depth 1), so that's not a factor here even though it could be in
   general. The clearest difference between the two LLM modalities is how
   many times they actually call the model: `llm_evaluator` used a mean of
   255 live calls per match versus `llm_direct`'s 6.5 — about 39x more —
   just because it has to call the model once per candidate leaf instead of
   once per move.

4. Stage C came out a perfectly even 2-2 split, and the outcome was decided
   entirely by color — whoever played Black won, all 4 times, no matter if
   that was `minimax_3` or `llm_direct`. That makes sense once you know
   `llm_direct` fell back on 100% of its calls in every match (5 of 5 each
   time), so this whole stage was really `evaluate_position_3` as a one-ply
   greedy chooser playing against `evaluate_position_3` inside a depth-1
   minimax — basically the same policy in two different shapes, decided by
   tempo instead of actual strategy. I pulled one concrete replay (Orange
   `llm_direct`, Black `minimax_3`, seed 0) to see what happened: Black
   captures an over-advanced Orange piece at ply 3 (the threat term's 0.3
   weight wasn't enough to stop the move that walked into it), then turns
   that into a Princess capture by ply 9.

   Given all that, the claims here are pretty narrow. With only 4 matches, 2
   seeds, and one evaluator/depth pairing, I can't say anything general like
   "Black has an advantage in Jetan" or "`minimax_3` beats `llm_direct`." I
   also can't say anything about how good the live model actually is at
   choosing moves, since it never once produced a usable move in this whole
   run — everything here is really just `evaluate_position_3` playing
   against itself.

5. [Pending — I could not find a PEAS performance-measure specification
   anywhere in this repository, so I can't draft this one responsibly without
   guessing at criteria the assignment didn't actually ask about. Please
   paste in the PEAS measure from the assignment instructions and I'll write
   this against our actual Stage A/B (and eventually C) evidence.]

## 8. Testing and Reproducibility

**Test command/result:** `.venv/bin/python -m unittest discover -p 'test_*.py'` — `Ran 52 tests in 0.017s / OK`, confirmed after every code change made to `student_strategies.py` throughout this project (evaluators, LLM evaluator prompt/parser, and direct-move prompt/parser).<br>
**Offline/live separation:** Every function was verified offline first, using `ScriptedClient` (the supplied deterministic fake) with canned responses covering valid input, duplicate JSON keys, extra keys, out-of-range values, and non-JSON prose, before any live request was made against the course endpoint. Live calls were only made afterward, deliberately, to confirm real-world behavior (token budgets, timing, actual model responses) — never as part of the offline test suite itself.<br>
**Registered agent names confirmed:** All five required agent names (`minimax_1`, `minimax_2`, `minimax_3`, `llm_evaluator`, `llm_direct`) were exercised through real matches via `play.py` and/or `run_experiments.py` (Stages A1, A2, B, and C), not just imported — each one successfully ran a full match and produced a `MatchResult`.

Testing covered each layer of my work:
- **Evaluators** (`evaluate_position_1/2/3`): hand-checked against constructed positions with known expected values (symmetric initial board, a forced material deficit, a one-sided threat via a Thoat attacking an undefended Warrior) — see Section 1's Hand-Checked Positions table for the deterministic evaluators, and this section covers the LLM-facing functions.
- **Response parsers** (`parse_evaluation_response`, `parse_move_response`): unit-style checks (run manually, not added as new files in `test_*.py`, since the assignment's existing test files already cover the framework side) confirmed valid responses parse correctly, and each rejection category — duplicate keys, wrong/extra keys, non-finite or out-of-range scores, non-JSON text, and hallucinated/illegal moves — correctly raises instead of silently accepting bad data.
- **Fallback behavior**: confirmed via `LLMEvaluationFunction`/`DirectLLMAgent` with `ScriptedClient` that any parser exception (malformed JSON, duplicate key) correctly triggers the deterministic fallback (`evaluate_position_3` / `choose_fallback`) rather than propagating an exception or crashing the search.
- **Budget behavior**: confirmed live that `llm_evaluator` respects `max_llm_calls` per move-search (verified with a small budget of 3 and 5 during early smoke tests) and that the framework's own cumulative `max_think_seconds` forfeit logic is what actually terminates a match, not something student code needs to enforce itself.
- **Scripted-client tests**: `ScriptedClient` was used throughout for offline verification instead of the live endpoint, per the README's guidance to keep development offline and enable live requests only deliberately for real timing/behavior checks.

## 9. AI-Assistance Disclosure

**Tools:** Claude (Anthropic), used interactively throughout this assignment.<br>
**Material effect:** Used for: discussing and designing the three
deterministic evaluation functions (piece weights, the advancement term, the
threat term) and their revision rationale; implementing
`student_strategies.py` (evaluators, LLM evaluation prompt/parser, direct-move
prompt/parser, and fallback policy); diagnosing and iterating on the LLM
prompt design against the live course endpoint (grid vs. piece-list format,
`/no_think`). I specifically used the LLM to help test my experiment before
running it, due to the predicted time the experiment was going to take.<br>
**Verification:** I ran the code through tests and the experiment gave results that were within parameters for the expected behavior of the agents. 

## Submission Checklist

- [ ] Accessible `lastname-firstname-jetan-agents.pdf` with selectable text and
      semantic headings/tables.
- [ ] Three deterministic evaluation functions, depth comparison, and Stage A
      results.
- [ ] Initial and revised prompts for both LLM modalities.
- [ ] Complete Stage B results, failures, and summary denominator fields.
- [ ] Complete Stage C head-to-head results and bounded interpretation.
- [ ] Five analysis questions answered from evidence.
- [ ] Raw and summary CSV files for Stages A1, A2, B, and C retained in the
      repository.
- [ ] Private repository URL, submitted commit, and `fractal13` read access.
- [ ] Tests pass; `.env`, credentials, and secrets are absent from the repository.
- [ ] References and AI-assistance disclosure are complete.

# Alpha-Beta Pruning and Move Ordering in Jetan

**Student:** Jonny Lightheart  
**Private repository:** <https://github.com/jlightheart24/lightheart-jonny-jetan>  
**Access:** Yes, `fractal13` has read access  
**Submitted commit:** [Hash]  
**Inherited U0-HW-05 commit:** `ad15c05573f04b0003143ad209f6680ab738fd08`

## 1. Selected U0-HW-05 Evaluation Function

**Definition:** I am using Evaluation 3 from U0-HW-05, a weighted linear
combination of three terms, each computed from the perspective player `p`
against `p.opponent`:
`0.5 * material + 0.2 * advancement + 0.3 * threat`, where

- `material = (my_weight - opp_weight) / 49`, with piece weights Panthan 1,
  Warrior 2, Padwar 2, Thoat 3, Dwar 4, Flier 4, Chief 5, Princess 6 (49 is the
  weight of one full starting army);
- `advancement = (my_adv - opp_adv) / (19 * 9)`, where a non-Princess piece's
  advancement is the number of rows it has moved from its own back rank (`y` for
  Orange, `9 - y` for Black);
- `threat = (opp_attacked - my_attacked) / 49`, where `X_attacked` is the total
  weight of X's pieces standing on squares in `board.attacked_locations(X.opponent)`.

**Range and perspective:** The function is a finite float for every legal board.
It needs no clamp to stay inside `[-0.99, 0.99]`. Terminal states are tested
before the depth cutoff, so the evaluator is only called on non-terminal boards.
On a non-terminal board both Princesses are still present, because capturing a
Princess ends the game. That means each side keeps at least 6 of its 49 material
points, so `|material| <= 43/49`. The advancement and threat terms are each
bounded by 1 in absolute value. The worst case is therefore
`0.5(43/49) + 0.2 + 0.3 ≈ 0.939`, which is strictly inside the terminal
utilities of ±1, so no heuristic value can tie with or beat a real win or loss.
Every term is a difference "mine minus opponent's", so
`eval(b, p) = -eval(b, p.opponent)`. The search fixes `p` as the root player
(`perspective = board.player()` at the root) and passes it unchanged through the
recursion, so every leaf is scored from the root player's point of view and
MAX/MIN is decided by whether `board.player() is perspective`.

**Existing function and registration:** `evaluate_position_3` in
`student_strategies.py`, registered in U0-HW-05 as the `minimax_3` agent
(`register_minimax_agent("minimax_3", evaluate_position_3, "Minimax 3")` in
`student_agents.py`). It was also the deterministic fallback for the LLM
evaluator.

**Unchanged-behavior verification:** `hw06_config.py` only imports the existing
function (`from student_strategies import evaluate_position_3 as
SELECTED_EVALUATOR`); I did not copy, wrap, or edit it. `hw06_run_experiments.py`
passes that same `hw06_config.SELECTED_EVALUATOR` to `MeasuredMinimaxAgent` and to
both `AlphaBetaAgent` orderings, and `hw06_registration.py` passes it to the
`alpha_beta_order_1` and `alpha_beta_order_2` agents. I checked at runtime that
`hw06_config.SELECTED_EVALUATOR is student_strategies.evaluate_position_3`
returns `True`. On the six supplied positions it returns the same values from
both perspectives with opposite signs (for example, `middle-capture`: Orange
+0.0085 / Black -0.0085; `initial`, `early-a`, and `early-b`: 0.0).
`student_strategies.py` is unchanged from the inherited U0-HW-05 commit.

## 2. Alpha-Beta Implementation

`AlphaBetaAgent._value(board, depth, perspective, alpha, beta)` in
`alpha_beta_agent.py` is a depth-first recursion. Alpha is the best value MAX is
already guaranteed on the current path, and beta is the best value MIN is
already guaranteed.

**Terminal before cutoff.** Each call first counts the visit. It then calls
`_evaluate_leaf`, which checks `board.is_terminal()` *before* comparing `depth`
to the depth limit. A win, loss, or draw is therefore scored with the exact
utility (±1 or 0) even when it sits exactly at the cutoff depth, and the
heuristic is only used on non-terminal boards at the depth limit.

**Expansion.** For an internal node, `board.actions()` is called exactly once.
The full tuple is counted in `generated_actions`, and that same tuple is passed to
the move-ordering function.

**MAX and MIN recurrence.** A node is MAX when `board.player() is perspective`
and MIN otherwise, so the recursion does not depend on depth parity.

- MAX: `v = max(v, value(child, α, β))` for each child in order. If `v >= β`,
  stop (beta cutoff). Otherwise `α = max(α, v)`.
- MIN: `v = min(v, value(child, α, β))` for each child in order. If `v <= α`,
  stop (alpha cutoff). Otherwise `β = min(β, v)`.

A cutoff returns `v` itself (fail-soft). I cut on equality (`>=`/`<=`) because a
branch that can only tie what the parent already has can never replace it,
since the root replaces its best move only on a strictly greater value.

**Pruning count.** On a cutoff after child `i` of `n`, the remaining `n - i - 1`
generated siblings are never searched, and that number is added to
`pruned_actions`. Measured minimax never prunes, so its count is always 0.

**Fixed root perspective.** `choose_action` sets `perspective = board.player()`
once at the root and passes it unchanged through every recursive call. Every
terminal utility and heuristic value is therefore from the root player's point
of view, and MIN nodes minimize that same number instead of negating it.

**Canonical root and tie handling.** The root is not reordered: it loops over
`board.actions()` in canonical order and searches each child with window
`(α, +∞)`, raising α after each child. The best move changes only when
`value > best_value`, so on a tie the first canonical move is kept, which is the
same rule measured minimax uses. The move ordering is applied only below the
root. When a root child is cut off, it returns some `v <= α`, which is at most
the best value found so far, so it cannot be selected. Its true minimax value is
also at most `v`, so exhaustive minimax would not select it either. Alpha-beta
therefore returns the same move and the same root value as minimax.

### Student-Created Alpha-Cutoff Test

**Depth-2 tree:** Orange (MAX) root with three Black (MIN) children. The leaves
are non-terminal boards at the depth-2 cutoff, so their values come from the
evaluator. Canonical traversal order is `a → a1, a2`, then `b → b1, b2, b3`, then
`c → c1, c2`.

```
                 root (MAX, Orange)
         /               |                  \
     a (MIN)          b (MIN)            c (MIN)
     /     \        /    |    \          /     \
   a1      a2     b1     b2    b3      c1      c2
  0.6     0.8    0.4   -0.9   0.9     0.7     0.9
```

**Leaf values:** a1 = 0.6, a2 = 0.8, b1 = 0.4, b2 = -0.9, b3 = 0.9, c1 = 0.7,
c2 = 0.9 (from Orange's fixed perspective).  
**Alpha established by the first root branch:** `a = min(0.6, 0.8) = 0.6`, so the
root sets α = 0.6 before searching `b`.  
**Later MIN-node value that triggers the cutoff:** At `b`, the first child `b1`
returns 0.4. Since 0.4 ≤ α = 0.6, MIN node `b` can do no better than 0.4 for
Orange, and Orange already has 0.6 through `a`. `b` stops immediately and
returns 0.4.  
**Sibling skipped by the cutoff:** `b2` and `b3` (2 pruned actions). `b2 = -0.9`
would have made `b` even worse for Orange, so skipping it does not change the
answer. Branch `c` is then searched with α = 0.6. Neither 0.7 nor 0.9 is ≤ 0.6,
so there is no cutoff there, and `c = 0.7` becomes the best move.  
**Test assertion and observed result:** `test_student_alpha_cutoff_case` in
`test_hw06_student.py` asserts that alpha-beta:

- chooses `c` with root value 0.7;
- calls the evaluator on exactly `[a1, a2, b1, c1, c2]`, so `b2` and `b3` are
  never evaluated;
- reports visited 9, expanded 4, generated 10, evaluated 5, pruned 2, max depth 2;
- agrees with `MeasuredMinimaxAgent`, which also chooses `c` with value 0.7 but
  evaluates all 7 leaves.

The test passes (`.venv/bin/python -m unittest -v test_hw06_student.py`).

## 3. Move-Ordering Strategies

Both keys are scored from the fixed root perspective. The supplied wrapper sorts
largest-first at MAX nodes and smallest-first at MIN nodes, so a move that is
good for the opponent must get a *low* key in order to be searched first at a
MIN node.

### Ordering 1: Capture Value (Victim Minus Attacker)

**Ranking procedure:** `ordering_key_1` looks only at the moving piece
(`board.at(action.source)`) and the piece on the destination square.

- A quiet (non-capture) move scores 0.
- A capture that ends the game (taking the Princess, or Chief taking Chief)
  scores 100.
- Any other capture scores `weight(victim) - weight(attacker) / 10`, using the
  U0-HW-05 piece weights (Panthan 1 through Princess 6). This is the standard
  "most valuable victim, least valuable attacker" ordering: take the biggest
  piece first, and when two captures take the same piece, prefer the cheaper
  attacker.
- The score is positive when the root player is moving and negated when the
  opponent is moving.

**Expected cutoff relationship:** Material is 50% of the evaluator, so captures
are the moves most likely to swing the backed-up value.

- At a MAX node, searching the root player's best capture first raises α early.
- At a MIN node, the opponent's best capture sorts first (most negative key),
  so it lowers the node's value below α quickly, which causes an alpha cutoff.
- Winning captures return ±1 immediately, which is the strongest possible
  cutoff.

**Computational cost:** O(1) per move: two board lookups and a table lookup. It
measured about 0.2 µs per move, so ordering a node costs `O(b log b)` for the
sort, which is negligible compared with searching the node's children.

**Canonical tie rule:** All quiet moves tie at 0, and equal captures (same
victim and attacker weights) also tie. The stable sort keeps tied moves in
canonical `board.actions()` order, including when it sorts in reverse at MAX
nodes.

**Likely weakness:** It says nothing about quiet moves, which are most moves.
In the four supplied positions where the side to move has no capture
(`initial`, `early-a`, `early-b`, `reduced`), Ordering 1 leaves the root's
children in exactly canonical order. It can only help at nodes deeper in the
tree, where captures exist. It also ignores recapture, so it ranks a capture
highly even when the capturing piece can be taken right back.

### Ordering 2: One-Ply Evaluator Lookahead

**Ranking procedure:** `ordering_key_2` builds the successor
`board.result(action)`.

- If the successor is terminal, it returns the exact utility
  `successor.utility(perspective)` (±1 or 0).
- Otherwise it returns `hw06_config.SELECTED_EVALUATOR(successor, perspective)`,
  the same unchanged `evaluate_position_3` the search uses at its leaves.

The evaluator's output stays within about ±0.939 (Section 1), so wins always rank
above every heuristic score and losses below.

**Expected cutoff relationship:** This is a one-ply preview using the leaf
evaluator itself, so it should predict each node's best move better than
Ordering 1.

- It still finds captures, because material is 50% of the score.
- It also ranks quiet moves by advancement and threats: moving pieces forward,
  attacking enemy pieces, and moving your own pieces out of attack.
- Good first choices raise α at MAX nodes and lower β at MIN nodes sooner, so
  both kinds of cutoff should happen sooner and alpha-beta should expand fewer
  nodes.

**Computational cost:** For each move it builds a new board, then runs the
evaluator on it: O(n) over the pieces plus `attacked_locations` for both sides.
That measured about 22–160 µs per move on the supplied positions, around 750×
more than Ordering 1. Ordering a node with `b` moves costs
`O(b · (n + attack generation))` plus the sort. This cost is paid at every
internal node below the root, even when a cutoff happens after the first child.

**Canonical tie rule:** Moves whose successors get equal scores (common early in
the game, when many quiet moves change neither material nor threats) keep
canonical order through the stable sort.

**Likely weakness:** Its cost. Scoring every move at every internal node may
take longer than the node expansions it saves, so fewer expanded nodes might not
mean less time. It also only looks one ply ahead: it can rank a capture that
loses the piece right back as the best move. Finally, it depends on
the quality of Eval 3, which ignores whose turn it is.

**Distinctness evidence:** The two orderings return different action sequences
on all six supplied positions.

- In `initial`, Ordering 1 has no captures, so it returns the canonical sequence
  (`a1-a2, a1-b3, a1-c2, b1-a2, ...`). Ordering 2 starts with
  `d0-a3, d0-c3, d0-e3, d0-g3`, which are moves with better advancement and
  threat scores.
- In `middle-capture`, both put the only capture `e4-f5` first. After that,
  Ordering 1 falls back to canonical order (`a0-a1, a0-a2, ...`) while
  Ordering 2 continues with `c3-f6, c3-f4, c3-b6`.

For every position and both orderings I checked that the returned sequence has
the same length as `board.actions()`, has no duplicates, and contains the same
set of moves, so each is a permutation containing every legal action exactly
once. `test_each_ordering_is_a_repeatable_permutation_of_canonical_input` and
`test_orderings_differ_on_at_least_one_supplied_position` both pass.

## 4. Stage A: Minimax Equivalence

**Combined command:** `.venv/bin/python hw06_run_experiments.py --output hw06_results`
(the README's `python3` command run with the Python 3.12 virtual environment; see Section 7)  
**Rows used here:** Stage A rows from the combined output  
**Positions:** Six supplied positions  
**Depth:** 2  
**Scheduled searches:** 18

| Position                           | Completed agent searches / 3 | Same root value? | Same selected move? | Maximum-depth check | Timeouts/errors |
|------------------------------------|-----------------------------:|------------------|---------------------|---------------------|-----------------|
| initial                            |                            3 | Yes (0.0)        | Yes (`d0-a3`)       | 2 = 2 for all three | 0 / 0           |
| early-a                            |                            3 | Yes (0.0)        | Yes (`d0-e3`)       | 2 = 2 for all three | 0 / 0           |
| early-b                            |                            3 | Yes (-0.0061)    | Yes (`g0-h3`)       | 2 = 2 for all three | 0 / 0           |
| middle-capture                     |                            3 | Yes (1.0)        | Yes (`e4-f5`)       | 2 = 2 for all three | 0 / 0           |
| middle-threat                      |                            3 | Yes (1.0)        | Yes (`e4-d3`)       | 2 = 2 for all three | 0 / 0           |
| reduced                            |                            3 | Yes (-0.0114)    | Yes (`c2-b4`)       | 2 = 2 for all three | 0 / 0           |

All 18 scheduled Stage A searches completed with status `ok`. On every position
the runner's `equivalent_move` and `equivalent_value` fields are both `True`: both
alpha-beta orderings return exactly the same root value and selected move as
exhaustive measured minimax. There were no failed or unavailable comparisons.

This supports correctness in three ways.

1. **Same answers across very different orderings.** Measured minimax evaluates
   every leaf. Alpha-beta skipped between 524 and 5,749 generated actions per
   search (roughly 70–94% of all generated actions), yet it still reached the same
   minimax value. So the cutoffs removed only branches that could not affect the
   result. Ordering 1 and Ordering 2 search children in different orders and
   prune different numbers of actions (for example, 2,349 vs. 2,530 on `initial`),
   but they still agree with each other and with minimax. So the result does not
   depend on move ordering, as it should not.
2. **Tie handling holds.** On `initial`, 8 root moves tie at the best value 0.0.
   On `early-a`, 6 root moves tie at 0.0. All three agents still choose the first
   of those in canonical order (root move 10 of 52 and root move 16 of 56). This
   confirms that the root keeps the first move on a tie and that moves are only
   reordered below the root.
3. **Terminal states are scored before the depth cutoff.** On `middle-capture` and
   `middle-threat` the root value is exactly 1.0. That is a real win found within
   two plies, scored by terminal utility rather than by the evaluator, whose
   values stay within about ±0.939.

The work counts are also internally consistent. For every row,
`visited = expanded + evaluated` (every node is either expanded or is a leaf),
and for alpha-beta
`generated = (visited - 1) + pruned` (every generated action is either searched or
pruned). Examples: `initial`, Ordering 1: 387 = 53 + 334 and 2,735 = 386 + 2,349.
The maximum depth is 2 in every row.

All three agents always expand the same number of nodes. At depth 2 the root and
every one of its children must be expanded, because a MIN node needs at least one
leaf value before it can be cut off, so pruning can only skip leaves. The saving
therefore shows up in visited, evaluated, and pruned counts, not in expanded
nodes.

| Position                                | Agent            | Status/error | Visited states | Expanded nodes | Generated actions | Evaluated states | Pruned actions | Max depth | Seconds |
|-----------------------------------------|------------------|--------------|---------------:|---------------:|------------------:|-----------------:|---------------:|----------:|--------:|
| initial                                 | Measured minimax | ok           |          2,736 |             53 |             2,735 |            2,683 |              0 |         2 |   0.440 |
| initial                                 | Ordering 1       | ok           |            387 |             53 |             2,735 |              334 |          2,349 |         2 |   0.065 |
| initial                                 | Ordering 2       | ok           |            206 |             53 |             2,735 |              153 |          2,530 |         2 |   0.464 |
| early-a                                 | Measured minimax | ok           |          3,171 |             57 |             3,170 |            3,114 |              0 |         2 |   0.512 |
| early-a                                 | Ordering 1       | ok           |            482 |             57 |             3,170 |              425 |          2,689 |         2 |   0.078 |
| early-a                                 | Ordering 2       | ok           |            329 |             57 |             3,170 |              272 |          2,842 |         2 |   0.554 |
| early-b                                 | Measured minimax | ok           |          4,008 |             62 |             4,007 |            3,946 |              0 |         2 |   0.662 |
| early-b                                 | Ordering 1       | ok           |          1,186 |             62 |             4,007 |            1,124 |          2,822 |         2 |   0.194 |
| early-b                                 | Ordering 2       | ok           |            312 |             62 |             4,007 |              250 |          3,696 |         2 |   0.695 |
| middle-capture                          | Measured minimax | ok           |          3,473 |             31 |             3,472 |            3,442 |              0 |         2 |   0.318 |
| middle-capture                          | Ordering 1       | ok           |            411 |             31 |             3,472 |              380 |          3,062 |         2 |   0.040 |
| middle-capture                          | Ordering 2       | ok           |            411 |             31 |             3,472 |              380 |          3,062 |         2 |   0.345 |
| middle-threat                           | Measured minimax | ok           |          6,096 |             79 |             6,095 |            6,017 |              0 |         2 |   0.970 |
| middle-threat                           | Ordering 1       | ok           |            347 |             79 |             6,095 |              268 |          5,749 |         2 |   0.052 |
| middle-threat                           | Ordering 2       | ok           |            347 |             79 |             6,095 |              268 |          5,749 |         2 |   1.016 |
| reduced                                 | Measured minimax | ok           |            755 |             27 |               754 |              728 |              0 |         2 |   0.018 |
| reduced                                 | Ordering 1       | ok           |            231 |             27 |               754 |              204 |            524 |         2 |   0.006 |
| reduced                                 | Ordering 2       | ok           |            134 |             27 |               754 |              107 |            621 |         2 |   0.021 |

Ordering 1 and Ordering 2 report identical work counts on `middle-capture` and
`middle-threat`. In both positions the side to move has a root move that captures
the enemy Princess and wins immediately:

- `e4-f5` is root move 31 of 31 in `middle-capture`.
- `e4-d3` is root move 16 of 79 in `middle-threat`.

The root is always searched in canonical order, so both orderings reach that move
at the same point. Once it is found, α = 1. That is the highest possible value,
so every later root child is cut off after its first leaf no matter how its moves
are ordered. Before that point, the opponent's replies in these tactical
positions include captures. Both keys rank those large material swings first, so
the resulting cutoffs are likely the same. The identical counts show only that
the two orderings produced the same cutoffs on these positions, not that the
keys are the same; Section 3 shows they order moves differently.

Ordering 2 took 3.4–19.6x longer than Ordering 1 on every position, because it
builds and evaluates a successor board for every move at every internal node. It
was also slower than unpruned measured minimax on all six positions, even though
it evaluated 85–96% fewer leaves. Ordering 1 evaluated 72–96% fewer leaves than
minimax and was 3–19x faster than it.

## 5. Stage B: Depth-3 Ordering Comparison

**Combined command:** `.venv/bin/python hw06_run_experiments.py --output hw06_results`
(the README's `python3` command run with the Python 3.12 virtual environment; see Section 7)  
**Rows used here:** Stage B rows from the combined output  
**Per-search limit:** 30 seconds  
**Timing repetitions:** 3  
**Scheduled attempts:** 36

Use one row for each position and ordering. Mark unavailable values `N/A`.

| Position                                       | Ordering   | Completed / 3 | Move | Root value | Visited states | Expanded nodes | Generated actions | Evaluated states | Pruned actions | Max depth | Median seconds | Timeouts/errors |
|------------------------------------------------|------------|--------------:|------|-----------:|---------------:|---------------:|------------------:|-----------------:|---------------:|----------:|---------------:|-----------------|
| initial                                        | Ordering 1 |             3 | `d0-a3` |     0.0709 |         10,985 |            468 |            26,731 |           10,517 |         15,747 |         3 |          1.760 | 0 / 0           |
| initial                                        | Ordering 2 |             3 | `d0-a3` |     0.0709 |          4,205 |            409 |            23,474 |            3,796 |         19,270 |         3 |          4.476 | 0 / 0           |
| early-a                                        | Ordering 1 |             3 | `d0-a3` |     0.0720 |         11,201 |            453 |            26,690 |           10,748 |         15,490 |         3 |          1.811 | 0 / 0           |
| early-a                                        | Ordering 2 |             3 | `d0-a3` |     0.0720 |          5,534 |            529 |            31,305 |            5,005 |         25,772 |         3 |          5.850 | 0 / 0           |
| early-b                                        | Ordering 1 |             3 | `d0-a3` |     0.0464 |         18,095 |            717 |            46,914 |           17,378 |         28,820 |         3 |          3.012 | 0 / 0           |
| early-b                                        | Ordering 2 |             3 | `d0-a3` |     0.0464 |          5,543 |            534 |            34,842 |            5,009 |         29,300 |         3 |          6.712 | 0 / 0           |
| middle-capture                                 | Ordering 1 |             3 | `e4-f5` |     1.0000 |          8,875 |            526 |            19,942 |            8,349 |         11,068 |         3 |          0.765 | 0 / 0           |
| middle-capture                                 | Ordering 2 |             3 | `e4-f5` |     1.0000 |          2,437 |            541 |            20,803 |            1,896 |         18,367 |         3 |          2.002 | 0 / 0           |
| middle-threat                                  | Ordering 1 |             3 | `e4-d3` |     1.0000 |          1,316 |            279 |            19,105 |            1,037 |         17,790 |         3 |          0.193 | 0 / 0           |
| middle-threat                                  | Ordering 2 |             3 | `e4-d3` |     1.0000 |          1,411 |            279 |            19,200 |            1,132 |         17,790 |         3 |          3.308 | 0 / 0           |
| reduced                                        | Ordering 1 |             3 | `c2-d4` |    -0.0079 |          3,909 |            270 |             8,206 |            3,639 |          4,298 |         3 |          0.098 | 0 / 0           |
| reduced                                        | Ordering 2 |             3 | `c2-d4` |    -0.0079 |          1,208 |            170 |             4,565 |            1,038 |          3,358 |         3 |          0.135 | 0 / 0           |

All 36 scheduled Stage B attempts (6 positions × 2 orderings × 3 repetitions)
completed with status `ok`, with no timeouts or errors, so no values are `N/A`.
In every group, all six work counts were identical across the three repetitions
because the search is deterministic; only the time varied, by at most about 5%
(the largest spread was `middle-threat`, Ordering 2, at 3.28–3.44 s). "Median
seconds" is the median `thinking_time` of the three repetitions. There is no
depth-3 unpruned baseline, so these rows compare the orderings with each other,
not against minimax. On every position the two orderings still selected the same
move with the same root value, which is consistent with Stage A: ordering changes
the work done, not the answer.

### Relative Expanded-Node Change

Use the `summary.csv` field
`relative_expanded_nodes_change_order2_vs_order1`, calculated as
`(Ordering 1 - Ordering 2) / Ordering 1`. Ordering 1 supplies the denominator;
it is not an unpruned reference.

| Position                  | Ordering 1 expanded nodes | Ordering 2 expanded nodes | Relative change | Interpretation |
|---------------------------|--------------------------:|--------------------------:|----------------:|----------------|
| initial                   |                       468 |                       409 |         +0.1261 | Ordering 2 expanded 12.6% fewer nodes |
| early-a                   |                       453 |                       529 |         -0.1678 | Ordering 2 expanded 16.8% **more** nodes |
| early-b                   |                       717 |                       534 |         +0.2552 | Ordering 2 expanded 25.5% fewer nodes |
| middle-capture            |                       526 |                       541 |         -0.0285 | Ordering 2 expanded 2.9% more nodes (nearly equal) |
| middle-threat             |                       279 |                       279 |          0.0000 | Same count, since the immediate root win dominates both searches |
| reduced                   |                       270 |                       170 |         +0.3704 | Ordering 2 expanded 37.0% fewer nodes |
| **All six (totals)**      |                 **2,713** |                 **2,462** |     **+0.0925** | Ordering 2 expanded 9.3% fewer nodes overall |

The per-position values are the runner's
`relative_expanded_nodes_change_order2_vs_order1` field. The totals row is my own
calculation using the same formula on summed counts.

- **Overall:** Ordering 2 expanded fewer nodes on three positions (`initial`,
  `early-b`, `reduced`), more on two (`early-a`, `middle-capture`), and the same
  on one (`middle-threat`). Its overall advantage in expanded nodes is modest
  (9.3%) and not consistent across positions.
- **Evaluated leaves:** the difference is much larger. Ordering 2 evaluated
  17,876 leaves in total versus 51,668 for Ordering 1, 65.4% fewer. It evaluated
  fewer on every position except `middle-threat` (1,132 vs. 1,037).
- **Why the two measures differ:** at depth 3, a node is only "expanded" at the
  root, at depth 1 (MIN), and at depth 2 (MAX). Depth-3 nodes are leaves. The
  depth-2 MAX nodes are where Ordering 2's accurate one-ply preview seems to help
  most: it puts the root player's best move first, so a beta cutoff tends to come
  after the first leaf, and many leaves are skipped. Whether a depth-2 node is
  expanded at all depends on how quickly its depth-1 MIN parent is cut off. On
  `early-a`, a likely cause of the reversal is the one-ply horizon. The opponent
  reply that Ordering 2 ranks best by static score may be answered well by the
  root player at depth 2, so it is not actually the best reply, the alpha cutoff
  at that MIN node comes later, and more depth-2 nodes are expanded. This is an
  interpretation of the counts; I did not trace the individual search.
- **Time:** Ordering 2 was slower on every position, by 1.4x (`reduced`) to 17.1x
  (`middle-threat`), and by 2.9x in total (22.48 s vs. 7.64 s summed medians).
  Even where it expanded 37% fewer nodes (`reduced`), it took 38% longer. Scoring
  every move with a successor board and the evaluator costs more than the
  searching it saves at this depth.

## 6. Analysis Questions

1. **How does Stage A support alpha-beta correctness?**

   Alpha-beta is correct if it returns the same root value and selected move as
   exhaustive minimax at the same depth. It only skips branches that cannot
   change that result. Stage A tests exactly this.

   - **Same results everywhere.** On all six positions, all 18 depth-2 searches
     completed. Both orderings matched measured minimax on root value and
     selected move (`equivalent_move` and `equivalent_value` are `True`
     everywhere).
   - **The agreement is not trivial.** Alpha-beta skipped 70–94% of generated
     actions, the two orderings skipped different amounts, and all three still
     agreed.
   - **Specific rules held:**
     - *Canonical tie rule:* all agents chose the first of 8 tied root moves on
       `initial` and the first of 6 on `early-a`.
     - *Terminal before cutoff:* exact 1.0 wins on both middle positions.
     - *Consistent counts:* `generated = (visited - 1) + pruned`.
   - **The limit.** This is evidence, not proof: six positions at one depth.
     The ToyBoard tests add exact checks on small trees, including my
     alpha-cutoff case.

2. **How do the two move-ordering strategies seek earlier cutoffs at MAX and
   MIN nodes?**

   Both keys score a move from the fixed root perspective, and the wrapper
   sorts in opposite directions at the two node types.

   - **MAX nodes (largest key first).** The root player's strongest move is
     tried first. That raises α quickly, so a later MIN child can be cut off as
     soon as one of its replies is ≤ α, and a MAX node can reach β after its
     first child.
   - **MIN nodes (smallest key first).** The opponent's strongest reply is tried
     first. That drives the MIN value down to α or below after one child,
     causing an alpha cutoff of the remaining siblings.
   - **Ordering 1** guesses "strongest" by capture value, most valuable victim
     first. Winning captures get ±100.
   - **Ordering 2** guesses it by the selected evaluator applied one ply ahead,
     which also ranks quiet moves by advancement and threats.

   In both cases the key is signed or oriented by who moves, so the same key
   works at MAX and MIN nodes.

3. **Which strategy expanded fewer nodes overall and by position? Explain a weak,
   absent, or reversed advantage.**

   **Overall, Ordering 2 expanded 9.3% fewer nodes at depth 3** (2,462 vs.
   2,713).

   | Advantage for Ordering 2 | Positions |
   |---|---|
   | Fewer nodes | `reduced` (+37.0%), `early-b` (+25.5%), `initial` (+12.6%) |
   | More nodes (reversed) | `early-a` (−16.8%) |
   | Nearly equal (weak) | `middle-capture` (−2.9%) |
   | Same count (absent) | `middle-threat` (0) |

   - **Absent on `middle-threat`.** The side to move has an immediate winning
     capture at root move 16 of 79. Once it is found, α = 1, and every later
     root branch stops after its first leaf regardless of ordering. The same
     effect weakens the difference on `middle-capture`, where the win is the
     last root move.
   - **Reversed on `early-a`.** Ordering 2 ranks the opponent's replies by
     static one-ply score. A reply that looks strongest statically may be
     answered well by the root player one ply deeper. In that case it is not
     actually the opponent's best reply, the cutoff at that MIN node comes
     later, and more depth-2 nodes get expanded. I did not trace the search, so
     this is a likely explanation, not a verified one.
   - **Expanded nodes understate the difference.** In leaves evaluated,
     Ordering 2 was fewer on five of six positions and 65.4% fewer in total
     (17,876 vs. 51,668), mostly from beta cutoffs at depth-2 MAX nodes.

   The node-count advantage is real for these positions but small and not
   uniform. Six positions are not enough to generalize.

4. **Did fewer expanded nodes correspond to a shorter median time? Explain the
   effect of ranking-key computation cost.**

   No. Ordering 2 had a higher median time on all six positions:

   - 1.4x on `reduced` up to 17.1x on `middle-threat`;
   - 22.48 s vs. 7.64 s summed.

   That includes the three positions where it expanded fewer nodes. On
   `reduced` it expanded 37% fewer nodes and evaluated 71% fewer leaves, yet
   took 38% longer.

   The reason is the ranking-key cost:

   - Ordering 1's key is two board lookups, about 0.2 µs per move.
   - Ordering 2's key builds a successor board and runs the evaluator, including
     attack generation for both sides. That is about 22–160 µs per move,
     comparable to evaluating a leaf.
   - Ordering 2 pays this for every generated move at every internal node, even
     when a cutoff happens after the first child. At depth 3 it scored roughly 4,500 to
     34,800 moves per search just to order them.
   - `middle-threat` shows the extreme: equal expanded nodes and slightly more
     leaves, plus about 19,000 expensive key evaluations, gave 3.31 s vs. 0.19 s.

   Stage A showed the same pattern: at depth 2, Ordering 2 was slower than
   unpruned minimax on every position. Node counts measure search work, not
   total cost. An ordering pays off only when the subtrees it prunes cost more
   than scoring the moves. That is more likely at greater depths, where each
   pruned subtree is larger, but I did not measure that.

5. **Use the README primer to relate the measurements to the theoretical time
   bounds. Explain `O(bd)` depth-first space and state that memory was not
   measured.**

   With branching factor `b` and depth `d`:

   - depth-limited minimax takes `O(b^d)` time;
   - alpha-beta also has `O(b^d)` worst-case time, but only about `O(b^(d/2))`
     with ideal ordering.

   A common concrete form of the ideal case is Knuth and Moore's minimum leaf
   count, `b^⌈d/2⌉ + b^⌊d/2⌋ - 1`. That is `2b - 1` leaves at depth 2 and
   `b^2 + b - 1` at depth 3. I use the root branching factor as `b`, which is a
   rough approximation because branching varies from node to node.

   **Depth 2, `initial` (`b` = 52):**

   | Search | Leaves evaluated |
   |---|---:|
   | Measured minimax | 2,683 (`b^2` = 2,704) |
   | Ideal alpha-beta | 103 |
   | Ordering 1 | 334 |
   | Ordering 2 | 153 |

   Ordering 2 is within 1.5x of the ideal and Ordering 1 within about 3.2x.
   Both are far below `b^2`. `early-a` and `reduced` show the same pattern.

   **Depth 3, `initial`:** the ideal count is about 2,755 leaves, and unpruned
   minimax would need about `b^3` ≈ 140,600. That is only an estimate, since
   Stage B has no depth-3 unpruned run. Ordering 2 evaluated 3,796 (about 1.4x
   the ideal) and Ordering 1 evaluated 10,517 (about 3.8x).

   So both orderings land between the ideal and worst-case bounds, and
   Ordering 2 lands closer to the ideal.

   **Caveats:**

   - `middle-capture` breaks the root-`b` approximation: the opponent has about
     112 replies per node versus 31 root moves.
   - `middle-threat` falls below the "ideal" estimate, because terminal wins
     stop the search early.
   - Six finite measurements at two depths cannot show an asymptotic growth
     rate. They are consistent with the bounds but do not prove them.

   **Space:** both searches are depth-first. At any moment they hold only the
   current path of at most `d` boards. At each level of the path they hold that
   node's action tuple and its reordered copy of `O(b)` moves. That gives
   `O(bd)` space, and ordering does not change it. **Memory use was not
   measured** in this assignment; the runner records only time and node counts.

6. **Describe a broader position suite and depth study that would strengthen
   empirical node-reduction claims without changing the theoretical bounds.**

   **A larger position suite.** Six hand-picked positions, two of them decided
   by an immediate win, are too few and too unusual.

   - Generate many positions (for example, 100 or more per category) by playing
     seeded random or mixed-agent games and sampling at fixed ply counts:
     opening, middle game, and reduced material.
   - Keep the seeds so the suite is reproducible.
   - Report positions with an immediate win separately, since they hide
     ordering effects.
   - Include both sides to move and a range of branching factors.

   **A depth study.**

   - Run depths 2 through 4 or 5 where time allows.
   - At each depth, include an unpruned minimax baseline where feasible, so the
     reduction can be measured against `b^d` directly and not just between
     orderings.
   - Add a random-ordering control and a reverse (deliberately bad) ordering, to
     bracket the effect of good ordering.

   **Reporting.**

   - Report the per-position distribution (median and quartiles) of relative
     node change, not just totals.
   - Add paired comparisons or confidence intervals across positions.
   - Fit the growth of leaves against depth to estimate an effective branching
     factor, and compare it with `b` and `√b`.
   - Keep timing separate from node counts and report the key-computation cost
     directly (for example, the number of key evaluations).

   This would make the empirical claims about these orderings stronger. The
   theoretical bounds — `O(b^d)` worst case, about `O(b^(d/2))` best case, and
   `O(bd)` space — would stay the same.

## 7. Testing and Reproducibility

**Python version and machine context:** CPython 3.12.14 in the project virtual
environment (`.venv/bin/python`), on macOS 26.6.2 (arm64), an Apple M5 with 10
cores and 16 GB of RAM. All searches ran
single-threaded, one child process per search. The system `python3` on this
machine is 3.9.6, which cannot import the U0-HW-05 framework because of its
`X | None` type syntax. All commands therefore used the 3.12 virtual environment,
for example `.venv/bin/python hw06_run_experiments.py --output hw06_results`.
Timings are specific to this machine; node counts are deterministic and should
reproduce anywhere.  
**Test command:**
`.venv/bin/python -m unittest -v test_hw06_framework.py test_hw06_student.py`,
plus `.venv/bin/python -m unittest discover -p "test_*.py"` for the whole project,
including the U0-HW-05 tests.  
**Test result:** All 18 HW06 tests pass (10 framework and 8 student tests,
including `test_student_alpha_cutoff_case` and both ordering requirements). The
full project suite runs 70 tests with result `OK`, so the U0-HW-05 code was not
broken.  
**Raw CSV:** `hw06_results/raw.csv`  
**Summary CSV:** `hw06_results/summary.csv`  
**Timeout/error handling verified:** Yes, three ways.

1. **Final run.** In the final run (`hw06_results/`) all 54 scheduled attempts
   (18 Stage A, 36 Stage B) finished with status `ok`. No timeout or error
   occurred under the 30-second limit, so every value in Sections 4–5 comes from
   a completed search.
2. **Supplied framework tests.**
   - `test_search_timeout_terminates_child_process` runs a search with a
     0.001-second limit and confirms it is recorded as `timeout` /
     `SearchTimeout` in under 5 seconds.
   - `test_matrix_shape_and_status_summary` forces every attempt to `error` and
     confirms that all 54 rows are kept and every summary group counts the
     errors.
   - `test_cli_rejects_zero_timeout` confirms that a non-positive limit is
     rejected.
3. **Live forced timeout.** I ran the slowest Stage B search (`middle-threat`,
   Ordering 2, normally about 3.3 s) with a 1-second limit:
   `.venv/bin/python hw06_run_experiments.py --stage B --position middle-threat --agent alpha_beta_order_2 --repeats 1 --timeout 1 --output hw06_trial_timeout`.
   - The run ended after about 1.06 s of wall time.
   - `raw.csv` recorded status `timeout`, error type `SearchTimeout`, message
     "search exceeded 1 seconds", and `N/A` for the move, value, and all seven
     metrics.
   - `summary.csv` reported 1 attempt, 0 completed, 1 timeout, and `N/A`
     values.

   This trial directory is git-ignored and is not part of the reported
   evidence.

## 8. References

1. S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th
   ed., Pearson, 2020, Ch. 5 ("Adversarial Search and Games"). Source of the
   depth-limited minimax and alpha-beta recurrences (α/β updates and cutoff
   tests) implemented in Section 2 and of the move-ordering motivation in
   Section 3.
2. D. E. Knuth and R. W. Moore, "An Analysis of Alpha-Beta Pruning,"
   *Artificial Intelligence*, 6(4), 1975, pp. 293–326,
   https://doi.org/10.1016/0004-3702(75)90019-3. Source of the best-case leaf
   count `b^⌈d/2⌉ + b^⌊d/2⌋ - 1` used in Analysis Question 5.
3. Chess Programming Wiki, "MVV-LVA" (Most Valuable Victim – Least Valuable
   Aggressor), https://www.chessprogramming.org/MVV-LVA. Capture-ordering idea
   adapted for Ordering 1.
4. A. L. Samuel, "Some Studies in Machine Learning Using the Game of Checkers,"
   *IBM Journal of Research and Development*, 3(3), 1959, pp. 210–229,
   https://doi.org/10.1147/rd.33.0210. Basis for the linear scoring polynomial
   behind the inherited U0-HW-05 evaluator (`evaluate_position_3`).
5. CS 4300 U0-HW-06 assignment package, `HW06-README.md`. Complexity primer
   (`O(b^d)`, `O(b^(d/2))`, `O(bd)`), experiment design, and the supplied
   measured-minimax reference and runner.

## 9. AI-Assistance Disclosure

**Tools:** Claude Code (Anthropic, Claude Opus 5.5), used interactively in my
editor throughout this assignment.  
**Material effect:** I used Claude as a pair programmer and writing assistant.
We talked through each part before making changes, and I chose the design
decisions: cutting off on equality with a fail-soft return, and a cheap
capture-based ordering versus an expensive one-ply evaluator ordering.

The inherited U0-HW-05 evaluator was not changed.  
**Verification:**

- All 18 HW06 tests and the full 70-test project suite pass.
- Alpha-beta matched the supplied measured minimax on all six Stage A positions.
- The work counts satisfy `visited = expanded + evaluated` and
  `generated = (visited - 1) + pruned`.
- Every number in Sections 4–6 was rechecked against `hw06_results/summary.csv`
  and `raw.csv`. Several first-draft figures were wrong and were corrected during
  this check.
- Explanations that were not directly measured, such as the likely cause of the
  `early-a` reversal, are labeled as interpretations.
- I reviewed the report and code before submitting.

## Submission Checklist

- [x] Accessible `lastname-firstname-alpha-beta.pdf`.
- [x] Selected U0-HW-05 evaluator and inherited commit identified.
- [x] Alpha-beta and two distinct ordering functions implemented.
- [x] All required tests pass.
- [x] All Stage A and B attempts, statuses, and denominators reported.
- [ ] Raw and summary CSV files retained.
- [x] Six analysis questions answered with bounded claims.
- [ ] Repository URL, submitted commit, and `fractal13` access included.
- [x] References and AI-assistance disclosure completed.
- [ ] No credentials, `.env` files, or private endpoint details committed.

# STL AST node reference

This document explains the semantics of **every node type** in the JSON AST produced by
`stl_parser.py`. Whenever a node appears in the visualization tool, look it up here.

## Common schema

Every node is a JSON object with at least an `op` field:

```json
{
  "op":       "atom | not | and | or | imply | iff | globally | finally | until",
  "interval": [low, high] | null,   // only the temporal operators (globally/finally/until) use this
  "children": [...],                // present on every non-atom; length depends on op (see table)
  "name":     "prop_1"              // only atoms have this
}
```

**Number of children at a glance:**

| op | children |
|---|---|
| `atom` | 0 (no `children` field; has `name`) |
| `not` | 1 |
| `globally`, `finally` | 1 |
| `until`, `imply`, `iff` | 2 |
| `and`, `or` | ≥ 2 (chains of the same operator are flattened into one multi-child node) |

**The `interval` field:** meaningful only for `globally` / `finally` / `until`, where it
gives the **time window** `[low, high]` (the unit is decided by the application:
seconds, steps, ticks, and so on). `high` may be `"inf"` for an unbounded window. On all
other nodes `interval` is `null`.

---

## 1. `atom` — atomic predicate (the leaf of the formula)

**Meaning:** an indivisible proposition or event, typically `prop_1` or `prop_2`,
corresponding in an application to a **Boolean signal** such as "the robot is in region A"
or "heart rate > 100".

**JSON:**
```json
{ "op": "atom", "name": "prop_1" }
```

**Visualization:** a yellow box showing the raw name.

**Key point:** an atom has no `children`. It is a leaf of the tree.

---

## 2. `not` — logical negation ¬

**Meaning:** the sub-formula does not hold. `!P` means "P does not hold".

**JSON:**
```json
{ "op": "not", "interval": null,
  "children": [ <sub-formula> ] }
```

**Visualization:** a light red box labelled "¬ NOT".

**Example:** `!prop_1` means "prop_1 does not occur".

---

## 3. `and` — logical conjunction ∧

**Meaning:** all children hold simultaneously.

**JSON:**
```json
{ "op": "and", "interval": null,
  "children": [ <child 1>, <child 2>, <child 3>, ... ] }
```

**Visualization:** a grey box labelled "∧ AND".

**Example:** `prop_1 & prop_2 & prop_3` becomes **one** and node with three children,
since chains of the same operator are flattened. This keeps the visualization compact,
the JSON shorter, and the structure easier for an LLM to learn.

---

## 4. `or` — logical disjunction ∨

**Meaning:** at least one child holds.

**JSON:**
```json
{ "op": "or", "interval": null,
  "children": [ <child 1>, <child 2>, ... ] }
```

**Visualization:** a grey box labelled "∨ OR".

**Flattening rule:** the same as `and`.

---

## 5. `imply` — implication →

**Meaning:** "if the antecedent holds, the consequent holds". `A -> B` is equivalent to
`!A | B`, but keeping an `imply` node is more readable than expanding it into or-not.

**JSON:**
```json
{ "op": "imply", "interval": null,
  "children": [ <antecedent A>, <consequent B> ] }
```

**Visualization:** a purple box labelled "→ IMPLY".

**Note:** `imply` is an **ordered** binary operator: `children[0]` is the antecedent and
`children[1]` is the consequent. Unlike and/or, the order matters.

**Example:** `prop_1 -> F[0,5] prop_2` means "once prop_1 occurs, prop_2 must occur within
5 seconds".

---

## 6. `iff` — equivalence ↔

**Meaning:** implication in both directions. `A <-> B` means A and B either both hold or
both fail. The NL2TL dataset writes this as `equal`.

**JSON:**
```json
{ "op": "iff", "interval": null,
  "children": [ <A>, <B> ] }
```

**Visualization:** a purple box labelled "↔ IFF".

---

## 7. `globally` — always □ (G)

**Meaning:** the sub-formula holds at **every instant within the given time window**. The
classical STL symbol is □ or G.

**JSON:**
```json
{ "op": "globally", "interval": [low, high],
  "children": [ <sub-formula> ] }
```

**Visualization:** a light blue box labelled "□ G (globally)" with the interval
`[low, high]`.

**Example:** `G[0,10] prop_1` means "prop_1 holds continuously between time 0 and 10".

**Key point:** this is a temporal operator over a **single** sub-formula. The interval is
the **window in which the requirement is enforced**, not a duration.

---

## 8. `finally` — eventually ◇ (F)

**Meaning:** there **exists an instant** within the given time window at which the
sub-formula holds. The classical STL symbol is ◇ or F.

**JSON:**
```json
{ "op": "finally", "interval": [low, high],
  "children": [ <sub-formula> ] }
```

**Visualization:** a light blue box labelled "◇ F (finally)" with the interval.

**Example:** `F[2,5] prop_2` means "prop_2 must hold at least once at some instant between
time 2 and 5".

**globally vs. finally:** globally means "at all times", finally means "at least once"; for
the same interval the two impose requirements of different strength.

---

## 9. `until` — until U

**Meaning:** `A U[a,b] B` means "from now on **A holds continuously** until, at some instant
inside the window `[a,b]`, **B holds**; and A holds at every instant before B does". This
is the **only binary** temporal operator in STL.

**JSON:**
```json
{ "op": "until", "interval": [low, high],
  "children": [ <A>, <B> ] }
```

**Visualization:** a light blue box labelled "U (until)" with the interval.

**Child order matters:** `children[0]` is **A** (which must hold before B), and
`children[1]` is **B** (the condition that eventually holds and terminates A). As with
imply, the order cannot be swapped.

**Example:** `prop_1 U[0,5] prop_2` means "prop_1 holds continuously until prop_2 holds at
some instant within 5 seconds". The NL2TL text form is `prop_1 until [0,5] prop_2`.

---

## Node type → visualization colour

| Node | Colour | Label |
|---|---|---|
| `atom` | yellow | the raw name |
| `not` | red | ¬ NOT |
| `and` | grey | ∧ AND |
| `or` | grey | ∨ OR |
| `imply` | purple | → IMPLY |
| `iff` | purple | ↔ IFF |
| `globally` | blue | □ G (globally) [a, b] |
| `finally` | blue | ◇ F (finally) [a, b] |
| `until` | blue | U (until) [a, b] |

Everything blue is a **temporal operator** (carrying an interval); the rest are logical
operators.

---

## A complete worked example

**Natural language:** "if prop_2 holds continuously until prop_1 occurs between time 176
and 415, and prop_3 also holds, then this is equivalent to prop_4"

**STL formula (NL2TL word forms):**
```
( ( ( prop_2 until [176,415] prop_1 ) and prop_3 ) equal prop_4 )
```

**JSON AST, annotated level by level:**
```json
{
  "op": "iff",                        // the top level is an equivalence
  "interval": null,
  "children": [
    {
      "op": "and",                    // the left side is an and
      "interval": null,
      "children": [
        {
          "op": "until",              // until is a binary temporal operator
          "interval": [176, 415],     // prop_1 must occur between 176 and 415
          "children": [
            {"op": "atom", "name": "prop_2"},  // prop_2 holds continuously before that
            {"op": "atom", "name": "prop_1"}   // the terminating condition
          ]
        },
        {"op": "atom", "name": "prop_3"}       // prop_3 holds as well
      ]
    },
    {"op": "atom", "name": "prop_4"}            // equivalent to prop_4
  ]
}
```

**Tree shape:**
```
        iff (↔)
        /    \
       and   prop_4
      /   \
   until   prop_3
   [176,415]
    /   \
prop_2  prop_1
```

This is the tree the visualization tool should show. **Verification checklist:**
- the top-level operator is iff ✓
- the right child of iff is the single atom `prop_4` ✓
- the left child of iff is an and ✓
- the and has 2 children: until and prop_3 ✓
- the interval of until is `[176, 415]` ✓
- the two children of until are in the order prop_2 (first), prop_1 (second) ✓

---

## Operator precedence (highest to lowest)

```
!         (unary not)
G, F      (unary temporal)
U         (binary temporal)
&         (and)
|         (or)
->        (imply)
<->       (iff)
```

**Example:** `prop_1 & prop_2 | prop_3` parses as `or(and(prop_1, prop_2), prop_3)`,
because `&` binds tighter than `|`. Add parentheses to change the grouping.

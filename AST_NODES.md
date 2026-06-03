# STL AST 节点含义参考

这份文档解释 `stl_parser.py` 产出的 JSON AST 里**每一种节点**的语义。可视化工具里看到任何一个节点,可以查这里。

## 通用 schema

每个节点都是一个 JSON 对象,至少有 `op` 字段:

```json
{
  "op":       "atom | not | and | or | imply | iff | globally | finally | until",
  "interval": [low, high] | null,   // 只有时间算子(globally/finally/until)用到
  "children": [...],                // 非 atom 都有,长度由 op 决定(见下表)
  "name":     "prop_1"              // 只有 atom 有
}
```

**children 数量速查:**

| op | children 数量 |
|---|---|
| `atom` | 0(没有 children 字段,有 name) |
| `not` | 1 |
| `globally`, `finally` | 1 |
| `until`, `imply`, `iff` | 2 |
| `and`, `or` | ≥ 2(同操作符链会被扁平化合并成多 children) |

**interval 字段:** 只有 `globally` / `finally` / `until` 有意义,表示**时间窗口** `[low, high]`(单位由你的应用语义决定:秒、step、tick……)。`high` 可以是 `"inf"` 表示无穷。其余节点 `interval` 为 `null`。

---

## 1. `atom` — 原子谓词(逻辑公式的"叶子")

**含义:** 一个不可再分的命题/事件,典型是 `prop_1`、`prop_2`,在你的应用里对应"机器人在区域 A"、"心率 > 100" 这种**布尔信号**。

**JSON:**
```json
{ "op": "atom", "name": "prop_1" }
```

**可视化:** 黄色框,显示原始名字。

**关键点:** atom 没有 `children`。它就是树的叶子。

---

## 2. `not` — 逻辑非 ¬

**含义:** 子公式不成立。`!P` 表示"P 不成立"。

**JSON:**
```json
{ "op": "not", "interval": null,
  "children": [ <子公式> ] }
```

**可视化:** 浅红框,标 "¬ NOT"。

**例子:** `!prop_1` → "prop_1 不发生"

---

## 3. `and` — 逻辑与 ∧

**含义:** 所有 children 同时成立。

**JSON:**
```json
{ "op": "and", "interval": null,
  "children": [ <子1>, <子2>, <子3>, ... ] }
```

**可视化:** 灰色框,标 "∧ AND"。

**例子:** `prop_1 & prop_2 & prop_3` 会变成**一个** and 节点带 3 个 children(同操作符链已扁平化)——这样可视化更紧凑,JSON 更短,LLM 学起来也更容易。

---

## 4. `or` — 逻辑或 ∨

**含义:** 至少一个 children 成立。

**JSON:**
```json
{ "op": "or", "interval": null,
  "children": [ <子1>, <子2>, ... ] }
```

**可视化:** 灰色框,标 "∨ OR"。

**扁平化规则:** 同 `and`。

---

## 5. `imply` — 蕴含 →

**含义:** "如果前件成立,则后件成立"。`A -> B` 等价于 `!A | B`,但保留 `imply` 节点比展开成 or-not 更可读。

**JSON:**
```json
{ "op": "imply", "interval": null,
  "children": [ <前件 A>, <后件 B> ] }
```

**可视化:** 紫色框,标 "→ IMPLY"。

**注意:** `imply` 是**有序的**两元运算:`children[0]` 是前件,`children[1]` 是后件。不像 and/or,顺序很重要。

**例子:** `prop_1 -> F[0,5] prop_2` = "一旦 prop_1 发生,则 5 秒内 prop_2 必然发生"。

---

## 6. `iff` — 等价 ↔

**含义:** 双向蕴含。`A <-> B` 表示 A 和 B 要么同时成立要么同时不成立。NL2TL 数据集里写作 `equal`。

**JSON:**
```json
{ "op": "iff", "interval": null,
  "children": [ <A>, <B> ] }
```

**可视化:** 紫色框,标 "↔ IFF"。

---

## 7. `globally` — 全局 □ (G)

**含义:** 在指定**时间窗口内每一时刻**,子公式都成立。STL 经典符号 □ 或 G。

**JSON:**
```json
{ "op": "globally", "interval": [low, high],
  "children": [ <子公式> ] }
```

**可视化:** 浅蓝框,标 "□ G (globally)" 和区间 `[low, high]`。

**例子:** `G[0,10] prop_1` = "在时刻 0 到 10 之间,prop_1 始终成立"。

**关键点:** 这是**单一子公式**的时序算子。区间是**强制有效区间**,不是"持续时间"。

---

## 8. `finally` — 终将 ◇ (F)

**含义:** 在指定时间窗口内**存在某个时刻**,子公式成立。STL 经典符号 ◇ 或 F。

**JSON:**
```json
{ "op": "finally", "interval": [low, high],
  "children": [ <子公式> ] }
```

**可视化:** 浅蓝框,标 "◇ F (finally)" 和区间。

**例子:** `F[2,5] prop_2` = "在时刻 2 到 5 之间,某一时刻 prop_2 至少要成立一次"。

**globally vs. finally:** globally 是"始终",finally 是"至少一次"——区间相同时两者要求强度不同。

---

## 9. `until` — 直到 U

**含义:** `A U[a,b] B` 表示"从现在起,**A 持续成立**,直到在 `[a,b]` 时间窗口内某一时刻 **B 成立**;并且在 B 成立之前 A 始终成立"。STL 里时序算子里**唯一的二元算子**。

**JSON:**
```json
{ "op": "until", "interval": [low, high],
  "children": [ <A>, <B> ] }
```

**可视化:** 浅蓝框,标 "U (until)" 和区间。

**注意 children 顺序:** `children[0]` 是 **A**(在 B 之前必须成立),`children[1]` 是 **B**(终将成立、终止 A 的那个条件)。和 imply 一样,顺序不可调换。

**例子:** `prop_1 U[0,5] prop_2` = "prop_1 一直成立,直到 5 秒内某时刻 prop_2 成立"。NL2TL 文本写作 `prop_1 until [0,5] prop_2`。

---

## 节点类型 → 可视化颜色速查

| 节点 | 颜色 | 标签 |
|---|---|---|
| `atom` | 🟡 黄 | 原始 name |
| `not` | 🔴 红 | ¬ NOT |
| `and` | ⚪ 灰 | ∧ AND |
| `or` | ⚪ 灰 | ∨ OR |
| `imply` | 🟣 紫 | → IMPLY |
| `iff` | 🟣 紫 | ↔ IFF |
| `globally` | 🔵 蓝 | □ G (globally) [a, b] |
| `finally` | 🔵 蓝 | ◇ F (finally) [a, b] |
| `until` | 🔵 蓝 | U (until) [a, b] |

蓝色都是**时序算子**(带区间),其余是逻辑算子。

---

## 完整示例(对照看)

**自然语言:** "if prop_2 holds continuously until prop_1 occurs between time 176 and 415, and prop_3 also holds, then this is equivalent to prop_4"

**STL 公式(NL2TL 词形):**
```
( ( ( prop_2 until [176,415] prop_1 ) and prop_3 ) equal prop_4 )
```

**JSON AST(每层带含义注解):**
```json
{
  "op": "iff",                        // 顶层是"等价"
  "interval": null,
  "children": [
    {
      "op": "and",                    // 左边是"and"
      "interval": null,
      "children": [
        {
          "op": "until",              // until 是二元时序算子
          "interval": [176, 415],     // 在 176~415 之间 prop_1 必须发生
          "children": [
            {"op": "atom", "name": "prop_2"},  // 在那之前 prop_2 一直成立
            {"op": "atom", "name": "prop_1"}   // 终止条件
          ]
        },
        {"op": "atom", "name": "prop_3"}       // 同时 prop_3 也成立
      ]
    },
    {"op": "atom", "name": "prop_4"}            // 等价于 prop_4
  ]
}
```

**树形结构:**
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

这就是你可视化工具里要看到的树。**verify checklist:**
- 顶层操作符是 iff ✓
- iff 的右孩子是单个 atom `prop_4` ✓
- iff 的左孩子是 and ✓
- and 有 2 个 children:until 和 prop_3 ✓
- until 区间是 `[176, 415]` ✓
- until 的两个 children 顺序是 prop_2(前)、prop_1(后)✓

---

## 优先级速查(从高到低)

```
!         (一元 not)
G, F      (一元时序)
U         (二元时序)
&         (and)
|         (or)
->        (imply)
<->       (iff)
```

**例:** `prop_1 & prop_2 | prop_3` 会被解析为 `or(and(prop_1, prop_2), prop_3)`,因为 `&` 优先级高于 `|`。要改顺序请加括号。

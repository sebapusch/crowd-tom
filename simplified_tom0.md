# Current model (what the code does)

This is the readable math for **this repository as it runs now** (`tom0-environments`, including seeing other agents in view range). It is not a proposal to change the simulator.

Cursor’s markdown preview only typesets **display** equations (the centred blocks). Single-dollar inline TeX is left raw, which is why those bits looked broken. This file therefore puts every formula in a display block. Longer writeups of an earlier commit live in `base-agent-model.md` and `core-model.md` (Seb). Those still match Helbing and door memory; they are **out of date** on follow and occupancy, which now require line of sight. Tiny numerical guards (epsilon = 10⁻¹², exact ties, degenerate walls) are in the code and in those files. You do not need them to understand the behaviour.

ToM-1 and ToM-2 are **not** implemented. `tom_order` 1 or 2 still runs ToM-0.

The loop is: pick a desired direction **eᵢ⁰**, then Helbing moves the body. Theory of mind only affects **eᵢ⁰**.

---

## 1. Bodies (Helbing)

Each agent *i* has position **rᵢ**, velocity **vᵢ**, radius 0.3, mass 1, desired speed 2, and relaxation time τᵢ = 1.

$$
m_i \frac{d\mathbf{v}_i}{dt}
= m_i \frac{v_i^0 \mathbf{e}_i^0 - \mathbf{v}_i}{\tau_i}
+ \sum_{j \neq i} \mathbf{f}_{ij}
+ \sum_{O} \mathbf{f}_{iO}.
$$

- **Drive:** relax toward the intended velocity

$$
v_i^0 \mathbf{e}_i^0.
$$

- **Other people fᵢⱼ:** exponential social push, plus extra compression and sliding friction when bodies overlap

$$
A = 2000,\quad B = 0.08,\quad k = 1.2 \times 10^5,\quad \kappa = 2.4 \times 10^5.
$$

Pairs farther than about 1.2 are ignored (`F_CUT`). This is contact, not vision.

- **Walls and circles fᵢₒ:** same idea against the nearest point on the obstacle. No cutoff. Distance to a circle is **signed** (negative if you are inside it).

Time step:

$$
\Delta t = \frac{1}{60}.
$$

Update velocity, **clip speed** at 1.5 vᵢ⁰, then move. An agent **leaves** when its **centre** crosses any exit segment, whether or not it chose that exit.

Default scene: 100 × 100 room, 400 agents, two north doors at y = 0 and one south door at y = 100. In this project “north” means y = 0 (top of the window).

---

## 2. Seeing a door

An exit is a line segment. The agent looks at the **closest point q** on that segment (not the whole doorway).

A door is visible if **all** of these hold:

- distance from the agent to **q** is at most view range R = 40

$$
\lVert \mathbf{q} - \mathbf{r}_i \rVert \le R, \qquad R = 40;
$$

- it lies in the field of view (default 360°, so this is always true);
- the segment from the agent to **q** does not hit a wall or circle.

**Other people do not block sight.** Heading used for a narrower FOV is the **previous desired direction**, not the actual velocity.

Reactive agents (`tom_order < 0`) only use currently visible doors. They do not write memory.

---

## 3. What ToM-0 is (and is not)

ToM-0: the agent has **beliefs about the room** (where doors were), not about other people’s minds.

When a ToM-0 agent **sees** door *k*, it stores:

- **μᵢₖ**: that closest point **q**;
- **sᵢₖ**: the time of this sighting.

If the door disappears, **μ** stays put. Age grows as

$$
a_{ik} = t - s_{ik}.
$$

Seeing it again overwrites both. Memories are never forgotten. Switching **T** to reactive does not wipe them; they are ignored until you switch back.

A newly observed door has age 0. Uncertainty used in the score is

$$
u_{ik} = \min\left(1,\frac{a_{ik}}{T_u}\right),
$$

with

$$
T_u = 20.
$$

The remembered point does not drift. **u** is only a “how stale is this?” penalty: it is 0 for a fresh memory and reaches 1 after $T_u=20$ seconds. It remains capped at 1 after that.

---

## 4. ToM-0 chooses among remembered doors

For every door the agent has ever seen, compute three numbers, then a score. Walk toward the **μ** of the **highest** score. Weights are negative, so **small** distance, crowding, and **u** win.

**Distance** — how far you are from the remembered point (which may no longer be the true closest point on the door):

$$
d_{ik} = \lVert \boldsymbol{\mu}_{ik} - \mathbf{r}_i \rVert.
$$

**Crowding** — not a blob around the door. From **you**, aim a slice **toward** **μᵢₖ**:

- length Rₒ = 12;
- opening 60°, which is 30° either side of the line to **μ**.

Count other agents in that slice, then divide by the slice area so it is a density:

$$
\frac{1}{2} R_o^2 \psi, \qquad R_o = 12, \qquad \psi = 60^\circ.
$$

**Also** (current code): that person must be **visible to you** (range 40, FOV, not behind a wall). With 360° and R = 40 greater than 12, the extra filter is mostly **line of sight**. Someone behind you is outside the 60° slice. Someone 20 units toward the door is outside the 12 cutoff. Someone 8 units toward the door but behind a pillar is **not** counted.

This is “who is in my way on the first 12 units toward that memory?”, not “how many people are standing at the exit.”

**Normalize** distance, crowding, and **u** **separately** across **this agent’s** remembered doors, each mapped to the interval from 0 to 1 (min maps to 0, max maps to 1). If all remembered doors have the same value on a criterion, that criterion is 0 for all of them. Then

$$
S_{ik} = w_d \widehat{d}_{ik} + w_o \widehat{o}_{ik} + w_u \widehat{u}_{ik},
\qquad
w_d = w_o = w_u = -1.
$$

With **one** remembered door, every hat-value is 0, so S = 0; that door is still chosen. The agent does **not** abandon a door because it is old or crowded in absolute terms — only **relative** to other memories.

Seeing a door **updates memory** but does **not** force picking that door. An unseen remembered door can win the score.

If the environment has **no exits at all**, everyone keeps the previous heading (no wander).

---

## 5. If you have never seen any door

The score is not used.

1. Among agents you **can currently see**, pick the nearest one who is **informed**.
2. If there is no such person, **wander**: rotate the previous heading by a small random angle. The angle is Gaussian with standard deviation

$$
\omega \Delta t, \qquad \omega = 1.5, \qquad \Delta t = \frac{1}{60}.
$$

**Informed** is a flag the simulator sets, not something you infer from walking:

- reactive: they **currently see** a door;
- ToM-0: they **have any door memory** (they may no longer see it).

You walk toward their **position**. You do not copy their door beliefs. Following does not make **you** informed.

So: vision limits **who you may follow**; the code still **tells** you who is informed.

---

## 6. Reactive policy (for comparison)

If any door is visible: walk toward the **nearest** visible closest-point. No crowding, no memory.

If none: same follow / wander as above, but “informed” means **sees a door now**.

---

## 7. One step, in order

1. From current positions and last headings, mark visible doors; ToM-0 writes **μ** and **s**.
2. Choose **eᵢ⁰** (score, or nearest visible door, or follow, or wander).
3. Compute Helbing forces from the **same** positions/velocities.
4. Velocity, speed cap, position.
5. Remove anyone whose centre crossed an exit. Time *t* then increases by Δt.

HUD numbers such as “seeing a door” use the **pre-move** visibility; ages use the clock after adding Δt, so a door just seen can already show age Δt.

---

## 8. Defaults (YAML can change some of these)

$$
\begin{aligned}
N &= 400, \quad L_x = L_y = 100, \\
m &= 1,\quad r = 0.3,\quad v^0 = 2,\quad \tau = 1, \\
A &= 2000,\quad B = 0.08,\quad k = 1.2\times 10^5,\quad \kappa = 2.4\times 10^5, \\
R &= 40,\quad \phi = 360^\circ, \\
R_o &= 12,\quad \psi = 60^\circ, \\
T_u &= 20, \\
w_d = w_o = w_u &= -1, \\
c &= 1.5,\quad \Delta t = 1/60.
\end{aligned}
$$

YAML sets room, doors, agent count/radius/speed, view range/FOV, and ToM vs reactive. Force coefficients and ToM score parameters are code defaults unless you change them in Python.

---

## 9. Where this lives in code

| Idea | File |
|---|---|
| Step, Helbing, memory update, follow/wander | `environment.py` |
| Score, occupancy slice | `tom0.py` |
| Door visibility and “can I see that agent?” | `perception.py` |
| Wall/circle geometry, exit crossing | `obstacle.py`, `exit.py` |
| YAML and spawn | `experiment.py`, `layout.py` |

`Agent.update_desired_direction` still “walk to nearest door in the whole room.” The window does **not** call it; `Environment.tick` does.

---

## 10. Later ToM-1 / ToM-2 (not built)

ToM-0 stops at “I remember doors.” ToM-1 would be “I model what **another agent** remembers or will choose.” That should stay a new chooser for **eᵢ⁰**, not a change to Helbing.

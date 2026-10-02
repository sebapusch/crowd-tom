# Crowd model: equations, implementation, and context

This document consolidates the reactive, ToM0, and ToM1 model notes. The reactive and ToM0 sections were checked against commit `2ca14f1` on 2 October 2026; ToM1 was checked against the working-tree implementation. Part I states the physical and behavioral model. Part II records implementation details that affect trajectories and measurements. Part III distinguishes the original paper's additional mechanisms and future proposals from implemented behavior.

## Contents

- [[#I. Core model|I. Core model]]
  - [[#1. State and notation|State and notation]]
  - [[#2. Physical motion|Physical motion]]
  - [[#3. Exit and agent perception|Exit and agent perception]]
  - [[#4. ToM0 exit memory|ToM0 exit memory]]
  - [[#5. ToM0 exit selection|ToM0 exit selection]]
  - [[#ToM1 exit-choice extension|ToM1 exit-choice extension]]
    - [[#Observed agents and witnessed-sighting memory|Observed agents and witnessed-sighting memory]]
    - [[#Estimated knowledge and memory uncertainty|Estimated knowledge and memory uncertainty]]
    - [[#Estimated ToM0 scores of observed agents|Estimated ToM0 scores of observed agents]]
    - [[#Predicted exit demand and ToM1 choice|Predicted exit demand and ToM1 choice]]
  - [[#6. Shared movement policy|Shared movement policy]]
  - [[#7. Departure|Departure]]
- [[#II. Implementation-specific details|II. Implementation-specific details]]
  - [[#1. Numerical integration and actual-speed clipping|Numerical integration and actual-speed clipping]]
  - [[#2. Small-magnitude guards and force cutoff|Small-magnitude guards and force cutoff]]
  - [[#3. Geometric calculations|Geometric calculations]]
  - [[#4. Exit-crossing calculation|Exit-crossing calculation]]
  - [[#5. Storage, selection, and special cases|Storage, selection, and special cases]]
  - [[#6. Current parameters and initialization|Current parameters and initialization]]
  - [[#ToM1 implementation choices|ToM1 implementation choices]]
  - [[#7. Diagnostics and source mapping|Diagnostics and source mapping]]
- [[#III. Paper context and future work|III. Paper context and future work]]
  - [[#Original paper versus this implementation|Original paper versus this implementation]]
  - [[#Higher-order ToM proposal|Higher-order ToM proposal]]
  - [[#Spatial-grid note and simplification ideas|Spatial-grid note and simplification ideas]]
  - [[#ToM1 research basis and modeling choices|ToM1 research basis and modeling choices]]
  - [[#References|References]]

## I. Core model

### 1. State and notation

Agent $i$ has position $\mathbf r_i$, velocity $\mathbf v_i$, radius $r_i$, mass $m_i$, fixed desired speed $v_i^0$, and relaxation time $\tau_i$. Its desired direction is $\mathbf e_i^0$. The heading $\mathbf h_i$ is the desired direction from its previous decision.

Let $\mathcal A$ be the active agents, $\mathcal O$ the stationary obstacles, and $E_k$ the exit segments. At each decision time $t$, all agents use the same current positions, velocities, and previous headings. Time arguments are omitted where unambiguous; $t^-$ denotes memory immediately before the current observation update.

Define normalization, a perpendicular direction, contact overlap, and closest-point projection:

$$
\mathcal N(\mathbf z)=\frac{\mathbf z}{\|\mathbf z\|},\qquad
\mathbf z^\perp=(-z_2,z_1),\qquad
g(x)=\max(0,x),\qquad
\Pi_S(\mathbf r)=\underset{\mathbf q\in S}{\arg\min}\|\mathbf q-\mathbf r\|.
\tag{1}
$$

Normalized vectors here are assumed nonzero. The contact function $g$ is part of the physical model: compression and friction act only when bodies overlap.

### 2. Physical motion

The intended velocity and equations of motion are

$$
\mathbf u_i^0=v_i^0\mathbf e_i^0,\qquad
\frac{d\mathbf r_i}{dt}=\mathbf v_i,\qquad
m_i\frac{d\mathbf v_i}{dt}
=m_i\frac{\mathbf u_i^0-\mathbf v_i}{\tau_i}
+\sum_{j\in\mathcal A\setminus\{i\}}\mathbf f_{ij}
+\sum_{O\in\mathcal O}\mathbf f_{iO}.
\tag{2}
$$

For a pair of agents, define

$$
d_{ij}=\|\mathbf r_i-\mathbf r_j\|,\qquad
r_{ij}=r_i+r_j,\qquad
\mathbf n_{ij}=\mathcal N(\mathbf r_i-\mathbf r_j),\qquad
\mathbf t_{ij}=\mathbf n_{ij}^{\perp},\qquad
\Delta v_{ji}^t=(\mathbf v_j-\mathbf v_i)\cdot\mathbf t_{ij}.
\tag{3}
$$

The interaction force is

$$
\boxed{
\mathbf f_{ij}
=\left[A_i\exp\!\left(\frac{r_{ij}-d_{ij}}{B_i}\right)
+k\,g(r_{ij}-d_{ij})\right]\mathbf n_{ij}
+\kappa\,g(r_{ij}-d_{ij})\Delta v_{ji}^t\mathbf t_{ij}.}
\tag{4}
$$

Here $A_i$ and $B_i$ set social repulsion, while $k$ and $\kappa$ set compression and sliding friction.

For obstacle $O$, let $d_{iO}$ be the surface distance, $\mathbf n_{iO}$ the normal pointing away from the obstacle, and $\mathbf t_{iO}=\mathbf n_{iO}^\perp$. For a solid circular obstacle, distance is negative inside its surface. Then

$$
\boxed{
\mathbf f_{iO}
=\left[A_i\exp\!\left(\frac{r_i-d_{iO}}{B_i}\right)
+k\,g(r_i-d_{iO})\right]\mathbf n_{iO}
-\kappa\,g(r_i-d_{iO})(\mathbf v_i\cdot\mathbf t_{iO})\mathbf t_{iO}.}
\tag{5}
$$

### 3. Exit and agent perception

The target point and distance for exit $k$ are

$$
\mathbf q_{ik}=\Pi_{E_k}(\mathbf r_i),\qquad
d_{ik}^{E}=\|\mathbf q_{ik}-\mathbf r_i\|.
\tag{6}
$$

For view range $R$ and full field-of-view angle $\phi$, the visible-exit set is

$$
\mathcal V_i=
\left\{k:
0<d_{ik}^{E}\le R,\quad
\operatorname{FOV}(\mathbf h_i,\mathbf q_{ik}-\mathbf r_i;\phi),\quad
\operatorname{LOS}(\mathbf r_i,\mathbf q_{ik})
\right\}.
\tag{7}
$$

$\operatorname{LOS}$ means that obstacles do not block the sight segment. Other agents do not block sight. Visibility refers to the closest point on the exit. $\operatorname{FOV}$ compares normalized heading and target direction against $\cos(\phi/2)$; for panoramic $\phi=2\pi$ the angular test is bypassed.

The same range, field of view, previous heading, and obstacle line-of-sight rules determine whether agent $i$ sees agent $j$. Let $P_{ij}=1$ denote this directional pairwise visibility. It restricts both following and the agents counted as ToM0 congestion. The simulator supplies a visible agent's informed flag; it is not inferred from that agent's movement.

### 4. ToM0 exit memory

For each known exit, agent $i$ remembers its last observation time $s_{ik}$ and target point $\boldsymbol\mu_{ik}$. Let $\mathcal K_i$ be its set of known exits, initially empty. At a ToM0 decision,

$$
\mathcal K_i(t)=\mathcal K_i(t^-)\cup\mathcal V_i(t),
\qquad
(s_{ik},\boldsymbol\mu_{ik})(t)=
\begin{cases}
(t,\mathbf q_{ik}(t)),&k\in\mathcal V_i(t),\\
(s_{ik},\boldsymbol\mu_{ik})(t^-),&k\in\mathcal K_i(t^-)\setminus\mathcal V_i(t).
\end{cases}
\tag{8}
$$

For each known exit, memory age and uncertainty are

$$
a_{ik}=\max(t-s_{ik},0),\qquad
\sigma_{ik}=\sigma_0+\beta\sqrt{a_{ik}},\qquad
u_{ik}=1-\exp\!\left(-\frac{\sigma_{ik}}{\max(\sigma_{\mathrm{ref}},\epsilon)}\right).
\tag{9}
$$

The remembered position stays fixed between sightings; only its uncertainty increases. There is no forgetting or sampled positional error. Reactive agents do not use or update this memory. The older $u=\min(1,a/T_u)$ rule in a short note is not implemented.

### 5. ToM0 exit selection

For $k\in\mathcal K_i$, define remembered distance and the cone direction:

$$
\widetilde d_{ik}=\|\boldsymbol\mu_{ik}-\mathbf r_i\|,\qquad
\mathbf c_{ik}=\mathcal N(\boldsymbol\mu_{ik}-\mathbf r_i).
\tag{10}
$$

With occupancy radius $R_o$ and full cone angle $\psi$, congestion is

$$
o_{ik}=
\frac{1}{\max(\tfrac12 R_o^2\psi,\epsilon)}
\sum_{j\in\mathcal A\setminus\{i\}}
\mathbf 1\{\epsilon<d_{ij}\le R_o\}\mathbf 1\{P_{ij}=1\}
\mathbf 1\{\mathcal N(\mathbf r_j-\mathbf r_i)\cdot\mathbf c_{ik}\ge\cos(\psi/2)\}.
\tag{11}
$$

This counts currently visible agents in a cone toward the remembered target. The cone is not clipped to the room or shortened at the exit, but agents hidden behind obstacles or outside view range/FOV are excluded. This measures the first part of the route toward the remembered point, not the crowd immediately around the exit.

Each criterion $x\in\{\widetilde d,o,u\}$ is normalized over the same agent's known exits:

$$
x_{i,\min}=\min_{k\in\mathcal K_i}x_{ik},\qquad
x_{i,\max}=\max_{k\in\mathcal K_i}x_{ik},\qquad
\widehat x_{ik}=
\begin{cases}
\dfrac{x_{ik}-x_{i,\min}}{x_{i,\max}-x_{i,\min}},&x_{i,\max}-x_{i,\min}\ge\epsilon,\\
0,&x_{i,\max}-x_{i,\min}<\epsilon.
\end{cases}
\tag{12}
$$

The score and chosen exit are

$$
S_{ik}=w_d\widehat d_{ik}+w_o\widehat o_{ik}+w_u\widehat u_{ik},\qquad
k_i^{\mathrm{ToM0}}=\underset{k\in\mathcal K_i}{\arg\max}\ S_{ik}.
\tag{13}
$$

The weights are negative, so shorter distance, lower congestion, and lower uncertainty are preferred. Selection is repeated at each decision. A single known exit is selected regardless of its absolute uncertainty, because all its normalized criteria are zero.

### ToM1 exit-choice extension

A ToM1 agent starts from its own ToM0 exit scores. It predicts the ToM0 choices of currently visible nearby agents using bounded records of witnessed sightings and priors for unknown exit knowledge. A negative predicted-demand term can change its own exit selection. These estimates use information available to the observer or an explicit prior; they do not read another agent's stored beliefs.

#### Observed agents and witnessed-sighting memory

Let $\mathcal K_i(t)$ be $i$'s own ToM0 set of known exits, and let $P_{ij}(t)=1$ mean that $i$ currently sees $j$ according to the current agent-perception rule. Define a separate ToM1 modeling radius $R_{\mathrm T}$ and the agents whose choices $i$ predicts now:

$$
\mathcal A^{(i)}_{\mathrm{obs}}(t)
=\{j\in\mathcal A(t)\setminus\{i\}:P_{ij}(t)=1,\ d_{ij}(t)\le R_{\mathrm T}\}.
\tag{T1}
$$

$R_{\mathrm T}$ need not equal the ToM0 occupancy-cone radius $R_o$. The visibility flag $P_{ij}$ already includes the simulator's view range, field of view, and obstacle line of sight. The following estimate is made only for $j\in\mathcal A^{(i)}_{\mathrm{obs}}$; a record of someone no longer visible may be retained briefly but does not add to current predicted congestion.

Let $\tau_{ijk}$ be the latest time at which $i$ inferred that $j$ saw exit $k$; it is undefined if this has never happened. Agent $i$ keeps a bounded set of these **witnessed exit sightings**:

$$
\mathcal M_i(t)
=\{(j,k,\tau_{ijk}):
  k\in\mathcal K_i(t),
  \tau_{ijk}\text{ is defined},
  0\le t-\tau_{ijk}\le T_{\mathrm{mem}}\},
\qquad
\left|\{j:\exists k,\tau\ (j,k,\tau)\in\mathcal M_i(t)\}\right|\le M.
\tag{T2}
$$

For a particular $j$, the exits that $i$ has a current sighting record for are

$$
\mathcal W^{(i)}_j(t)
=\{k\in\mathcal K_i(t):\exists\tau\ (j,k,\tau)\in\mathcal M_i(t)\}.
\tag{T2a}
$$

$\mathcal W^{(i)}_j$ is a set of **witnessed exits**, not $j$'s true known-exit set; $j$ may know exits outside it.

When the $M$-agent limit is reached, $i$ discards all records of the agent whose latest witnessed sighting is oldest. Records older than $T_{\mathrm{mem}}$ also expire. These are **$i$'s observations about $j$**, not copies of $j$'s memory. A witnessed sighting is an estimate: $i$ must see $j$ and infer that the exit is visible from $j$'s position, using the exit information and obstacle geometry known to $i$. If a restricted field of view is used, $i$ must also estimate $j$'s heading; the default panoramic view does not require that heading.

Not witnessing a sighting is not evidence that $j$ does not know the exit: $j$ may have seen it before meeting $i$ or while outside $i$'s view.

#### Estimated knowledge and memory uncertainty

Use the ToM0 age and uncertainty rule in (9), with $a=\max(t-\tau_{ijk},0)$ and $u(a)=1-\exp[-(\sigma_0+\beta\sqrt a)/\max(\sigma_{\mathrm{ref}},\epsilon)]$. Here $a$ measures time since **$i$ last inferred a sighting by $j$**, rather than time since $j$ actually last saw the exit.

Define a separate confidence-decay function for that witnessed record:

$$
\mathsf{Conf}(a)=v_{\mathrm{prior}}+(1-v_{\mathrm{prior}})e^{-a/T_v},\qquad a\ge0.
\tag{T3a}
$$

$v_{\mathrm{prior}}\in[0,1)$ is $i$'s baseline confidence when it has no retained sighting record for this agent–exit pair. $T_v>0$ is the e-folding time: after $T_v$ seconds, the confidence above baseline is $1/e$ of its initial value. A sighting resets $\mathsf{Conf}(0)=1$; a record's expiration at $T_{\mathrm{mem}}$ switches the estimate to $v_{\mathrm{prior}}$. Thus the finite record horizon can cause a downward jump. $\mathsf{Conf}$ is a **heuristic belief about what $j$ knows**, not a measured probability that $j$ forgets.

Now $v^{(i)}_{jk}(t)$, **$i$'s confidence that $j$ knows exit $k$**, is

$$
v^{(i)}_{jk}(t)=
\begin{cases}
\mathsf{Conf}\!\left(\max(t-\tau_{ijk},0)\right),
  &(j,k,\tau_{ijk})\in\mathcal M_i(t),\\
v_{\mathrm{prior}},&\text{otherwise},
\end{cases}
\qquad k\in\mathcal K_i(t).
\tag{T3}
$$

A newly witnessed sighting gives $v^{(i)}_{jk}=1$. As the record ages, $i$'s confidence approaches the prior; expiration returns it to the prior. This decay represents **limited memory or tracking by $i$**, not forgetting by $j$. The implementation uses common $v_{\mathrm{prior}},T_v,T_{\mathrm{mem}},M$ for all agents; their values need calibration.

The second uncertainty is conditional: **if $j$ knows $k$, how stale does $i$ think $j$'s exit memory is?** For a witnessed sighting use the existing ToM0 age penalty with the witnessed time as a proxy for $j$'s last observation:

$$
\widehat u^{(i)}_{jk}(t)=
\begin{cases}
u\!\left(\max(t-\tau_{ijk},0)\right),
&(j,k,\tau_{ijk})\in\mathcal M_i(t),\\
u_{\mathrm{prior}},&\text{otherwise}.
\end{cases}
\tag{T4}
$$

Here $u_{\mathrm{prior}}\in[0,1]$ is an assumed penalty for an exit whose sighting history is unknown. Reusing the ToM0 age penalty keeps the ToM1 estimate on the same scale, but is a modeling assumption, not an observation of $j$'s actual memory. Equation (T4) can overestimate staleness if $j$ saw $k$ again without $i$ witnessing it. Equations (T3) and (T4) have different roles: $v^{(i)}_{jk}$ describes **whether** $j$ knows $k$; $\widehat u^{(i)}_{jk}$ estimates **how stale** that knowledge is if it exists.

#### Estimated ToM0 scores of observed agents

The point remembered by $j$ is hidden from $i$. As a first approximation, $i$ uses its own remembered point $\boldsymbol\mu_{ik}$ for exit $k$:

$$
\widehat d^{(i)}_{jk}=\|\boldsymbol\mu_{ik}-\mathbf r_j\|,\qquad
\mathbf a^{(i)}_{jk}
=\frac{\boldsymbol\mu_{ik}-\mathbf r_j}
{\max(\|\boldsymbol\mu_{ik}-\mathbf r_j\|,\epsilon)}.
\tag{T5}
$$

Thus $\widehat d^{(i)}_{jk}$ is **not** $j$'s exact remembered-target distance. To estimate the congestion $j$ would perceive, let $\mathcal O_i=\{i\}\cup\mathcal A^{(i)}_{\mathrm{obs}}$ be positions currently available to $i$. Let $\widehat P^{(i)}_{jy}$ be $i$'s estimate that $j$ sees $y$, and $\mathbf p_{jy}=(\mathbf r_y-\mathbf r_j)/d_{jy}$ when $d_{jy}>\epsilon$. Then

$$
\widehat o^{(i)}_{jk}
=\frac{1}{\max(\tfrac12R_o^2\psi,\epsilon)}
\sum_{y\in\mathcal O_i\setminus\{j\}}
\mathbf1\{\epsilon<d_{jy}\le R_o\}
\mathbf1\{\widehat P^{(i)}_{jy}=1\}
\mathbf1\{\mathbf p_{jy}\cdot\mathbf a^{(i)}_{jk}\ge\cos(\psi/2)\}.
\tag{T6}
$$

For $\widehat P^{(i)}_{jy}$, $i$ can apply its known geometry to $j$ and $y$, using an estimated heading when needed. This remains a **partial observation**: another agent visible to $j$ but not to $i$ is missing from $\mathcal O_i$. The vector in the cone test points from $j$ to $y$, and $y=j$ is excluded.

For each observed $j$, normalize each criterion $x\in\{\widehat d^{(i)},\widehat o^{(i)},\widehat u^{(i)}\}$ across $k\in\mathcal K_i$, using the same min-max rule and $\epsilon$ constant-range guard as ToM0. Denote the normalized values by $\overline d^{(i)}_{jk}$, $\overline o^{(i)}_{jk}$, and $\overline u^{(i)}_{jk}$. The **estimated ToM0 score** is

$$
\widehat S^{(i,0)}_{jk}
=w_d\overline d^{(i)}_{jk}
+w_o\overline o^{(i)}_{jk}
+w_u\overline u^{(i)}_{jk}.
\tag{T7}
$$

There is no ToM1 congestion term in (T7): $i$ models $j$ as a ToM0 chooser, avoiding a recursive score definition.

#### Predicted exit demand and ToM1 choice

Let $\theta>0$ be a softmax temperature. The chance, under $i$'s independent-knowledge approximation, that $j$ knows at least one exit in $\mathcal K_i$ is

$$
q_j^{(i)}=1-\prod_{k\in\mathcal K_i}(1-v^{(i)}_{jk}).
\tag{T8}
$$

As a tractable heuristic, condition on $j$ knowing at least one modeled exit and weight each estimated ToM0 score by the chance that $j$ knows that exit. The resulting **unconditional** predicted choice probability is

$$
p^{(i)}_{jk}=
\begin{cases}
q_j^{(i)}
\dfrac{v^{(i)}_{jk}\exp(\widehat S^{(i,0)}_{jk}/\theta)}
{\sum_{\ell\in\mathcal K_i}
v^{(i)}_{j\ell}\exp(\widehat S^{(i,0)}_{j\ell}/\theta)},
&\text{if the denominator is positive},\\[1.2ex]
0,&\text{otherwise}.
\end{cases}
\tag{T9}
$$

The unassigned probability $1-q_j^{(i)}$ represents no known exit among $i$'s modeled candidates; $j$ might follow, wander, or know an exit outside $\mathcal K_i$. If $v^{(i)}_{jk}=0$, exit $k$ receives zero probability. This fixes the problem with multiplying a score by a zero belief flag before softmax, which would still yield a nonzero probability. Equation (T9) is an approximation rather than an exact average over every possible exit set known by $j$. $\theta$ controls choice sharpness rather than physical or memory uncertainty.

Predicted demand for exit $k$ is the expected number of currently observed agents choosing it:

$$
C_{ik}=\sum_{j\in\mathcal A^{(i)}_{\mathrm{obs}}}p^{(i)}_{jk}.
\tag{T10}
$$

Normalize $C_{ik}$ across $k\in\mathcal K_i$ with the ToM0 min-max rule to obtain $\widehat C_{ik}$. Since that normalization removes the absolute scale of predicted demand, retain an overall confidence factor

$$
\rho_i=\begin{cases}
\dfrac{1}{|\mathcal A^{(i)}_{\mathrm{obs}}|}
\displaystyle\sum_{j\in\mathcal A^{(i)}_{\mathrm{obs}}}q_j^{(i)},
&\mathcal A^{(i)}_{\mathrm{obs}}\ne\varnothing,\\
0,&\text{otherwise}.
\end{cases}
\tag{T10a}
$$

Let $S^{(0)}_{ik}$ be $i$'s own ToM0 score, computed with **its own** memory and current occupancy. Then

$$
S^{(1)}_{ik}=S^{(0)}_{ik}+w_c\rho_i\widehat C_{ik},\qquad
k_i^{(1)}=\arg\max_{k\in\mathcal K_i}S^{(1)}_{ik},\qquad w_c\le0.
\tag{T11}
$$

ToM1 changes only $i$'s exit choice. If $\mathcal K_i$ is empty, use the existing follow/wander fallback; if no other agent is observed, all $C_{ik}=0$ and the ToM1 term vanishes. Small knowledge confidence also weakens the extra term through $\rho_i$. The existing force law and movement update remain unchanged. The cone density in ToM0 measures current visible crowding along a route, while $C_{ik}$ estimates future exit choices; they may be correlated, so $w_c$ needs calibration rather than automatically taking the same value as $w_o$.

### 6. Shared movement policy

Each policy supplies a personal target $\mathbf y_i$, when available, and a leader-eligibility flag $H_i$:

| Policy | Personal target | Eligible leader |
|---|---|---|
| Reactive | $\mathbf y_i=\mathbf q_{ik_i}$, where $k_i=\arg\min_{k\in\mathcal V_i}d_{ik}^{E}$, if $\mathcal V_i\ne\varnothing$ | $H_i=\mathbf1\{\mathcal V_i\ne\varnothing\}$ |
| ToM0 or ToM1 | $\mathbf y_i=\boldsymbol\mu_{ik_i}$, using that mode's exit score, if $\mathcal K_i\ne\varnothing$ | $H_i=\mathbf1\{\mathcal K_i\ne\varnothing\}$ |

Define the eligible leaders and the nearest one:

$$
\mathcal F_i=\{j\in\mathcal A\setminus\{i\}:H_j=1\text{ and }P_{ij}=1\},\qquad
j_i=\underset{j\in\mathcal F_i}{\arg\min}\|\mathbf r_j-\mathbf r_i\|
\quad\text{when }\mathcal F_i\ne\varnothing.
\tag{14}
$$

For exploration, draw an independent angular perturbation $\eta_i\sim\mathcal N(0,s_w^2)$ at each decision, where $s_w$ is the angular standard deviation, and let

$$
\mathsf R(\eta)=
\begin{pmatrix}
\cos\eta&-\sin\eta\\
\sin\eta&\cos\eta
\end{pmatrix}.
$$

All three policies then use

$$
\boxed{
\mathbf e_i^0=
\begin{cases}
\mathcal N(\mathbf y_i-\mathbf r_i),&\text{personal target available},\\
\mathcal N(\mathbf r_{j_i}-\mathbf r_i),&\text{no personal target and }\mathcal F_i\ne\varnothing,\\
\mathsf R(\eta_i)\mathbf h_i,&\text{otherwise},
\end{cases}
\qquad
\mathbf h_i\leftarrow\mathbf e_i^0.}
\tag{15}
$$

Leader selection uses eligibility flags and positions but requires follower-to-leader visibility. Following targets the leader's position and does not transfer its exit beliefs. In ToM0 and ToM1, a remembered exit may be preferred over a currently visible one.

### 7. Departure

An agent is removed when its center trajectory crosses any exit segment:

$$
T_i=\inf\{t:\mathbf r_i\text{ crosses some }E_k\text{ at }t\},\qquad
\mathcal A(t)=\{i:t<T_i\}.
\tag{16}
$$

Departure does not depend on which exit was selected or perceived.

## II. Implementation-specific details

### 1. Numerical integration and actual-speed clipping

The simulator uses a fixed step $\Delta t=1/60$. It observes exits and other agents, updates ToM0 memory, and selects directions using the pre-movement snapshot. It then evaluates the forces for all agents and applies semi-implicit Euler:

$$
\mathbf a_i^n=
\frac{v_i^0\mathbf e_i^{0,n}-\mathbf v_i^n}{\tau_i}
+\frac{1}{m_i}\left(\sum_{j\ne i}\mathbf f_{ij}^{\mathrm{impl},n}
+\sum_O\mathbf f_{iO}^n\right),\qquad
\widehat{\mathbf v}_i^{n+1}=\mathbf v_i^n+\Delta t\,\mathbf a_i^n.
\tag{I1}
$$

Actual speed is capped at $v_{i,\mathrm{cap}}=c\,v_i^0$, with $c=1.5$:

$$
\mathbf v_i^{n+1}=
\begin{cases}
\widehat{\mathbf v}_i^{n+1},&\|\widehat{\mathbf v}_i^{n+1}\|\le c\,v_i^0,\\
\dfrac{c\,v_i^0}{\max(\|\widehat{\mathbf v}_i^{n+1}\|,\epsilon)}\widehat{\mathbf v}_i^{n+1},&\text{otherwise},
\end{cases}
\qquad
\mathbf r_i^{n+1}=\mathbf r_i^n+\Delta t\,\mathbf v_i^{n+1}.
\tag{I2}
$$

The cap changes motion when activated. It caps actual velocity after acceleration; desired speed remains fixed. A tentative zero velocity stays zero.

The angular standard deviation in (15) is implemented as $s_w=\omega\Delta t$, with $\omega=1.5$. At the default step, $s_w=0.025$ radians. Its variance therefore scales with $\Delta t^2$; changing the timestep changes exploration statistics over a fixed duration.

### 2. Small-magnitude guards and force cutoff

The principal tolerance is $\epsilon=10^{-12}$. Normalization in direction selection is replaced by

$$
\mathcal N_\epsilon(\mathbf z)=
\begin{cases}
\mathbf z/\|\mathbf z\|,&\|\mathbf z\|\ge\epsilon,\\
\mathbf0,&\|\mathbf z\|<\epsilon.
\end{cases}
\tag{I3}
$$

Agent-agent forces are truncated using $\gamma=1$ newton:

$$
D_{ij}=r_i+r_j-B_i\ln(\gamma/A_i),\qquad
\mathbf f_{ij}^{\mathrm{impl}}=
\begin{cases}
\mathbf f_{ij},&\epsilon\le d_{ij}<D_{ij},\\
\mathbf0,&\text{otherwise}.
\end{cases}
\tag{I4}
$$

This masks the whole pair force, including contact terms, and excludes self-interaction. With the current parameters, $D_{ij}\approx1.208$ meters and is beyond contact distance. Obstacle forces have no cutoff.

Other guards are local numerical conventions:

- Exit and agent visibility require target distance in $(\epsilon,R]$; occupancy counts require $d_{ij}>\epsilon$ and agent visibility.
- The cone axis uses division by $\max(\|\boldsymbol\mu_{ik}-\mathbf r_i\|,\epsilon)$; unlike (I3), it does not explicitly set a small nonzero vector to zero.
- Memory age is computed as $\max(t-s_{ik},0)$. The uncertainty denominator uses $\max(\sigma_{\mathrm{ref}},\epsilon)$, and cone area uses $\max(\tfrac12R_o^2\psi,\epsilon)$.
- Min-max normalization treats any criterion range smaller than $\epsilon$ as constant and returns zero.
- Panoramic vision bypasses the angular test when $\phi\ge2\pi-10^{-9}$, for both exits and other agents.

At initialization and repacking after departures, headings are normalized and near-zero headings become $(1,0)$. During ordinary direction selection, a zero target displacement returns zero desired direction. There is no additional recovery rule for reaching a remembered target exactly.

### 3. Geometric calculations

For a nondegenerate segment $S=[\mathbf a,\mathbf b]$, the projection used for walls and exits is

$$
\Pi_S(\mathbf r)=\mathbf a+
\operatorname{clip}_{[0,1]}\left(
\frac{(\mathbf r-\mathbf a)\cdot(\mathbf b-\mathbf a)}{\|\mathbf b-\mathbf a\|^2}
\right)(\mathbf b-\mathbf a).
\tag{I5}
$$

Wall distance is $\|\mathbf r_i-\Pi_O(\mathbf r_i)\|$, with its normal pointing from that closest point toward the agent. Within $\epsilon$ of the wall, the implementation sets distance to zero and uses the normalized perpendicular to its segment. Segments with squared length below $\epsilon$ are treated as points, with normal $(1,0)$ at coincidence.

For a circle of center $\mathbf c_O$ and radius $R_O$,

$$
d_{iO}=\|\mathbf r_i-\mathbf c_O\|-R_O,\qquad
\mathbf n_{iO}=\mathcal N(\mathbf r_i-\mathbf c_O),\qquad
\mathbf q_{iO}=\mathbf c_O+R_O\mathbf n_{iO}.
\tag{I6}
$$

Within $\epsilon$ of the circle center, the normal is set to $(1,0)$.

Sight tests exclude intersections at the sight segment's endpoints: an obstacle intersection counts only at a fraction strictly between $\epsilon$ and $1-\epsilon$. Wall tests include wall endpoints but ignore parallel or collinear segments when the cross-product magnitude is below $\epsilon$. Circle tests solve the quadratic boundary-intersection equation, reject negative discriminants, and ignore sight segments with squared length below $\epsilon$.

### 4. Exit-crossing calculation

Define $\mathbf x\times\mathbf y=x_1y_2-x_2y_1$. For a movement segment and exit $E_k=[\mathbf a_k,\mathbf b_k]$, let

$$
\mathbf p_i=\mathbf r_i^{n+1}-\mathbf r_i^n,\qquad
\mathbf s_k=\mathbf b_k-\mathbf a_k,\qquad
\mathbf w_{ik}=\mathbf a_k-\mathbf r_i^n.
$$

If $|\mathbf p_i\times\mathbf s_k|\ge\epsilon$, compute

$$
\lambda_{ik}=\frac{\mathbf w_{ik}\times\mathbf s_k}{\mathbf p_i\times\mathbf s_k},\qquad
\nu_{ik}=\frac{\mathbf w_{ik}\times\mathbf p_i}{\mathbf p_i\times\mathbf s_k}.
\tag{I7}
$$

The agent leaves when both fractions belong to $[0,1]$ for any exit. Movement/exit endpoints count, either crossing direction is accepted, and parallel or collinear movement does not count. If a movement hits multiple exits, the first one in exit-list order receives the escape count. The final departure is timestamped at the end of the step, rather than at the interpolated crossing time.

### 5. Storage, selection, and special cases

- Known-exit membership is stored in a Boolean array. Separate arrays store observation times and remembered positions. Unknown entries contain placeholder zeros, receive an ineligible score of $-\infty$, and are excluded from normalization.
- Exact score ties and leader-distance ties select the first entry in the relevant list. Reactive movement uses the strictly nearest visible exit; its diagnostic selected-exit index uses an additional $10^{-9}$ distance tolerance.
- Surviving agents retain their beliefs, selected exits, and visibility entries when arrays are repacked. The next step recomputes visibility.
- If the environment contains no exits, direction selection returns the previous heading without wandering. If no agents remain, simulation time stops advancing.
- Negative `tom_order` selects reactive behavior; zero selects ToM0; one selects ToM1. Values two and above remain placeholders that execute ToM0. In YAML, an omitted value defaults to zero; `null`, `none`, and `reactive` select the reactive policy. A `tom_proportions` mapping assigns ToM0 and ToM1 to individual agents at spawn, overriding the homogeneous order. Both proportions are in $[0,1]$ and sum to one; the ToM1 count is rounded to the nearest agent and assignments are shuffled. The T key cycles through reactive, ToM0, and ToM1, and returns to the mixture in mixed scenes.
- Toggling the mode changes the policy without clearing memory. Reactive mode leaves existing memories untouched; returning to ToM0 reuses them. Resetting or editing the experiment creates a new environment with empty memory.

### 6. Current parameters and initialization

These values instantiate the equations; they are not universal model constants.

| Category | Default values |
|---|---|
| Agent properties | $m_i=1$, $r_i=0.3$, $v_i^0=2$, $\tau_i=1$ |
| Force parameters | $A_i=2000$, $B_i=0.08$, $k=1.2\times10^5$, $\kappa=2.4\times10^5$ |
| Exit perception | $R=40$, $\phi=2\pi$ |
| Agent perception | The same $R$ and $\phi$ as exit perception, plus obstacle line of sight |
| Memory uncertainty | $\sigma_0=0.1$, $\beta=0.5$, $\sigma_{\mathrm{ref}}=8$ |
| Exit score | $w_d=w_o=w_u=-1$ |
| Occupancy cone | $R_o=12$, $\psi=\pi/3$ |
| ToM1 predicted demand | See [[#ToM1 implementation choices|ToM1 implementation choices]] |
| Numerical settings | $\Delta t=1/60$, $\epsilon=10^{-12}$, $\gamma=1$, $c=1.5$, $\omega=1.5$ |
| Startup scene | $400$ ToM0 agents in a $100\times100$ room |

Under the paper's unit convention, lengths are meters, time is seconds, mass is kilograms, and forces are newtons. The current mass and relaxation time differ from the original paper's calibration.

The startup file is `two-north-one-south.yaml`, with exits $[(20,0),(23,0)]$, $[(77,0),(80,0)]$, and $[(50,100),(53,100)]$. Perimeter walls are generated around exit gaps; YAML can also specify interior walls and circles. YAML exposes scene geometry, agent count/radius/desired speed, homogeneous or mixed ToM mode, and exit-perception settings. The `tom1` block overrides ToM1 parameters; the remaining parameters use code defaults, with programmatic overrides available for environment settings and ToM0 parameters.

Initial velocities are zero. Initial headings have uniformly sampled angles in $[0,2\pi)$. Positions are sampled by sequential rejection in the room with margin $r_i+0.05+1$, pair separation at least $2r_i+0.05$, and obstacle clearance at least $r_i+0.05$. All exit memories start empty.

### ToM1 implementation choices

| Parameter | Code field | Default |
|---|---|---:|
| Modeling radius $R_{\mathrm T}$ | `model_range` | $12$ |
| Tracked agents $M$ | `memory_agents` | $16$ |
| Record lifetime $T_{\mathrm{mem}}$ | `memory_horizon` | $30$ s |
| Confidence decay $T_v$ | `confidence_decay` | $10$ s |
| Knowledge prior $v_{\mathrm{prior}}$ | `knowledge_prior` | $0.2$ |
| Unknown-history penalty $u_{\mathrm{prior}}$ | `uncertainty_prior` | $0.5$ |
| Choice temperature $\theta$ | `choice_temperature` | $1$ |
| Predicted-demand weight $w_c$ | `demand_weight` | $-1$ |

These are heuristic starting values, not calibrated behavioral measurements. They can be overridden in a YAML `tom1` block or through `ToM1Params`. The current implementation uses observed velocity as a proxy for another agent's hidden heading when estimating their visibility under restricted FOV. With panoramic FOV, the heading estimate is irrelevant. For predicted occupancy, the implementation evaluates visibility among $i$ and the agents currently observed by $i$; it cannot include agents outside that set. Witness records persist when ToM1 is temporarily disabled and expire by age when it resumes; resetting the experiment creates a fresh memory state.

### 7. Diagnostics and source mapping

Diagnostics do not feed back into the core equations. The simulator records escape counts per exit, evacuation time, agents seeing exits, agents holding beliefs, agents selecting an unseen exit, and agents holding beliefs while seeing no exit. These last two conditions are distinct. It also tracks memory ages, uncertainty, distances to remembered targets, peaks, and accumulated blind-commitment agent-seconds.

For surviving agents after a step, `seeing` counts those who saw at least one exit at the decision snapshot; `with_belief` counts those holding any remembered exit; `no_belief` is the remaining survivors. `memory_guided` counts selected exits not visible at that snapshot, including when a different exit was visible. `blind_committed` counts belief holders who saw no exit. The latter two are distinct conditions. Escape counts are accumulated per exit; evacuation time is the first completed step with no agents remaining.

Metrics are collected after movement and departure removal, but use visibility and choices from the preceding decision. Mean ages and $\sigma$ use the updated clock; maximum commitment distance uses the updated positions. A just-seen exit can already show age $\Delta t$. Peaks are recorded for memory-guided count and commitment distance, while blind commitment is summed as agent-seconds. The CSV contains `t,remaining,remaining_tom0,remaining_tom1,remaining_reactive,escaped,seeing,with_belief,memory_guided,no_belief`; the remaining diagnostics are available separately. In mixed scenes, the type counts follow agents' assigned types even if the T key temporarily overrides their behavior. Mode toggles can leave beliefs from earlier ToM0 steps. In reactive mode, selected-exit age and distance statistics can use zero placeholder memory entries and do not then describe real memories.

| Responsibility | Source |
|---|---|
| Force calculation, policy selection, memory, numerical update | [environment.py](/home/seb/Code/uni/crowd-tom/environment.py) |
| Cone density, normalization, uncertainty, exit score | [tom0.py](/home/seb/Code/uni/crowd-tom/tom0.py) |
| ToM1 witnessed-sighting records and predicted demand | [tom1.py](/home/seb/Code/uni/crowd-tom/tom1.py) |
| Visibility and line-of-sight tests | [perception.py](/home/seb/Code/uni/crowd-tom/perception.py) |
| Wall/circle geometry and departure tests | [obstacle.py](/home/seb/Code/uni/crowd-tom/obstacle.py), [exit.py](/home/seb/Code/uni/crowd-tom/exit.py) |
| Scene loading, spawning, and perimeter construction | [experiment.py](/home/seb/Code/uni/crowd-tom/experiment.py), [layout.py](/home/seb/Code/uni/crowd-tom/layout.py) |
| Fixed timestep and CSV history | [main.py](/home/seb/Code/uni/crowd-tom/main.py), [plots.py](/home/seb/Code/uni/crowd-tom/plots.py) |

The running loop calls `Environment.tick`. The older `Agent.update_desired_direction` method still exists but does not specify current behavior.

## III. Paper context and future work

### Original paper versus this implementation

The social-force terms in Part I come from Helbing, Farkas, and Vicsek's escape-panic paper. The paper also used behavioral mechanisms that **are not in the current code**. Its $p_i$ denotes different things in different simulations; here $p_i^{\mathrm{herd}}$ and $p_i^{\mathrm{imp}}$ keep those uses distinct.

| Paper mechanism | Formulation | Current code |
|---|---|---|
| Herding | $\mathbf e_i^0=\mathcal N[(1-p_i^{\mathrm{herd}})\mathbf e_i+p_i^{\mathrm{herd}}\langle\mathbf e_j^0\rangle_i]$: blend independent search with nearby agents' average desired direction. The expected observed-neighbor count is $h=\pi R^2\rho$ for density $\rho=N/A$. | Target, follow, or wander branches; no weighted average of directions. |
| Impatience | $p_i^{\mathrm{imp}}=1-\bar v_i/v_i^0$ and $v_i^0(t)=[1-p_i^{\mathrm{imp}}]v_i^0(0)+p_i^{\mathrm{imp}}v_i^{\max}$: lack of forward progress can increase desired speed. | Desired speed is fixed. The implemented cap acts on **actual** speed after integration. |
| Injury | In one paper simulation, radial contact force per body circumference above $1{,}600\ \mathrm{N\,m^{-1}}$ makes a pedestrian an immobile obstacle. | No injury rule or conversion. |
| Corridor efficiency | $E=\langle\mathbf v_i\cdot\mathbf e_i^0\rangle/v_0$. | Not an exported metric. |
| Two-exit imbalance | $\lvert N_1-N_2\rvert$ for numbers using each door. | Per-exit counts are available; imbalance is not a dedicated metric. |

The paper's indicative parameter choices include $m_i=80\ \mathrm{kg}$, $\tau_i=0.5\ \mathrm{s}$, $A_i=2{,}000\ \mathrm N$, $B_i=0.08\ \mathrm m$, $k=1.2\times10^5\ \mathrm{kg\,s^{-2}}$, $\kappa=2.4\times10^5\ \mathrm{kg\,m^{-1}\,s^{-1}}$, and body diameters in $[0.5,0.7]\ \mathrm m$. It describes desired speeds around $0.6$ m/s when relaxed, $1$ m/s under ordinary conditions, and about $1.5$ m/s when nervous; extreme **desired**, rather than achieved, speed may exceed $5$ m/s. One ordinary-flow calibration reports roughly $0.73$ persons/s through an effectively one-meter door at desired speed around $0.8$ m/s. The current code's mass $1$ and relaxation time $1$ differ, so its defaults should not be presented as a reproduction of that calibration.

The paper explains temporary blocking arches and intermittent outflow through compression and tangential friction at narrow exits. In one room study, raising desired speed beyond roughly $1.5$ m/s causes a faster-is-slower effect. A corridor widening can lower efficiency when pedestrians fan out and then interfere while merging. Strong herding can overuse one of two exits, while an intermediate mix of search and imitation can evacuate fastest. These are **paper findings**, not results established for the current ToM0 simulator. The original study is primarily mechanistic and qualitative; it did not have direct escape-panic data for precise quantitative calibration.

Both the paper and this simulation simplify people as two-dimensional circular bodies with stylized forces. Neither current policy represents detailed communication, social groups, or a complete map of the building. The paper's injury threshold is a simplified operational rule, not a biomechanical injury model. These limits matter when interpreting evacuation predictions.

### Higher-order ToM proposal

The second DMAS assignment proposes adjusting a base desired velocity with order-$\ell$ mental-state beliefs $B_i^{(\ell)}$:

$$
\widetilde{\mathbf u}_i^{(\ell)}
=\mathbf u_i^{\mathrm{base}}
+\alpha\,\Delta\mathbf u_i^{(\ell)}(B_i^{(\ell)}),\qquad
\mathbf e_i^0=\mathcal N(\widetilde{\mathbf u}_i^{(\ell)}),\qquad
v_i^0=\min(v_i^{\max},\|\widetilde{\mathbf u}_i^{(\ell)}\|).
\tag{P1}
$$

For a zero adjusted vector, the proposal specifies zero desired speed and retention of the previous unit direction. This **velocity-adjustment interface is not active code**. The implemented ToM1 heuristically predicts nearby agents' ToM0 exit choices, but does not implement the assignment's nested mental-state beliefs, $\alpha$, $\Delta\mathbf u_i^{(\ell)}$, or desired-speed cap $v_i^{\max}$. ToM2 is not implemented; setting `tom_order` to 2 still runs ToM0.

### Spatial-grid note and simplification ideas

The old grid note derived a cell width from the repulsive-force threshold. For a spatial lookup that searches the current and eight adjacent square cells, a sufficient width is

$$
s\ge D_{\max}=\max_{i\ne j}\left[r_i+r_j-B_i\ln\left(\frac{\gamma}{A_i}\right)\right].
\tag{P2}
$$

When all $A_i>\gamma$ and $B_i>0$, $2r_{\max}+B_{\max}\ln(A_{\max}/\gamma)$ is a convenient upper bound. The radius term is the **sum** of two radii, so the old note's single maximum radius is insufficient. The running vectorized force calculation computes pairwise distances directly and applies (I4); the grid helper does not select force pairs. Thus cell width does not currently affect interactions.

Several presentation or refactoring ideas from the earlier notes remain useful. Equation (I5) defines the same closest-point operation for walls and exits; substituting an exit midpoint would change model behavior. The cone area in (11) is common to all candidate exits, so raw cone counts produce the same min-max-normalized congestion score when the criterion is nonconstant; an exact code refactor must retain the scaled $\epsilon$ threshold. The uncertainty penalty in (9) has a common offset and positive scale across an agent's exits, so its normalized ranking can be written using $1-\exp[-(\beta/\sigma_{\mathrm{ref}})\sqrt a]$, again retaining the original near-equality guard for exact behavior. With the default panoramic FOV, the angular condition drops out, while range and obstacle occlusion still apply to both exits and agents.

### ToM1 research basis and modeling choices

| Component | Research connection | Status in this model |
|---|---|---|
| Bounded tracking of observed agents | [Pylyshyn and Storm (1988)](https://pubmed.ncbi.nlm.nih.gov/3153671/) found limited multiple-object tracking. | Motivates a finite record capacity; $M$, $T_{\mathrm{mem}}$, and oldest-record eviction are uncalibrated choices. |
| Confidence that another agent knows an exit | Inferring hidden beliefs from observations is related to inverse-planning accounts of action understanding ([Baker, Saxe, and Tenenbaum, 2009](https://www.sciencedirect.com/science/article/pii/S0010027709001607)). | $\mathsf{Conf}$ uses inferred sighting events and a prior, not a fitted inverse-planning model. Its exponential curve is convenient, not an empirical forgetting law. |
| Time-dependent uncertainty of an exit memory | State-estimation models allow uncertainty to increase between observations ([Kalman, 1960](https://www.cs.unc.edu/~welch/kalman/media/pdf/Kalman1960.pdf)); the ToM0 definitions are in (9). | The age-dependent spread and staleness map are inherited heuristics. ToM1 substitutes $i$'s last witnessed sighting of $j$ for $j$'s unknown last sighting. |
| Shape of time decay | [Wixted and Ebbesen (1991)](https://www.psychologicalscience.org/journals/psychological-science/j.1467-9280.1991.tb00175.x/) compared forgetting functions and favored a power law in their tasks. | No cited study calibrates the exponential $\mathsf{Conf}$ curve, the saturating ToM0 age penalty in (9), or their default parameters for evacuation decisions. |

### References

1. Helbing, D., Farkas, I., and Vicsek, T. (2000). [*Simulating Dynamical Features of Escape Panic*](/home/seb/Vaults/University/26-27/DMAS/papers/panic-model.pdf), especially physical equations (1)–(3).
2. [Second DMAS assignment](/home/seb/Vaults/University/26-27/DMAS/assignments/2_DMAS___Crowd_cooperation_ToM.pdf), Sections 3.1–3.2.
3. Active implementation: [`crowd-tom`](/home/seb/Code/uni/crowd-tom); the reactive and ToM0 baseline was checked at commit `2ca14f1` on 2 October 2026, with ToM1 added in the current working tree.

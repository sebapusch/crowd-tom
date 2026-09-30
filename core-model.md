# Crowd model: core equations and implementation details

Based on the reactive and ToM0 model in [base-agent-model.md](/home/seb/Vaults/University/26-27/DMAS/model/base-agent-model.md), corresponding to code commit `23b8468`. Prepared on 30 September 2026.

Part I specifies the physical and behavioral model. Part II specifies its numerical realization, including force truncation and speed clipping, which can affect simulated trajectories. The physical notation follows Helbing et al. and Section 3.1 of the second DMAS assignment. Higher-order ToM extensions are outside the current model.

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

### 3. Exit perception

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
d_{ik}^{E}\le R,\quad
\mathbf h_i\cdot\mathcal N(\mathbf q_{ik}-\mathbf r_i)\ge\cos(\phi/2),\quad
\operatorname{LOS}(\mathbf r_i,\mathbf q_{ik})
\right\}.
\tag{7}
$$

$\operatorname{LOS}$ means that obstacles do not block the sight segment. Other agents do not block sight. Visibility refers to the closest point on the exit. For panoramic perception, $\phi=2\pi$ and the angular condition is always satisfied.

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
a_{ik}=t-s_{ik},\qquad
\sigma_{ik}=\sigma_0+\beta\sqrt{a_{ik}},\qquad
u_{ik}=1-\exp\!\left(-\frac{\sigma_{ik}}{\sigma_{\mathrm{ref}}}\right).
\tag{9}
$$

The remembered position stays fixed between sightings; only its uncertainty increases. There is no forgetting or sampled positional error. Reactive agents do not use or update this memory.

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
\frac{1}{\tfrac12 R_o^2\psi}
\sum_{j\in\mathcal A\setminus\{i\}}
\mathbf 1\{d_{ij}\le R_o\}
\mathbf 1\{\mathcal N(\mathbf r_j-\mathbf r_i)\cdot\mathbf c_{ik}\ge\cos(\psi/2)\}.
\tag{11}
$$

This counts current positions in a cone toward the remembered target. The cone is not clipped at obstacles or at the exit, and occupancy does not require line of sight.

Each criterion $x\in\{\widetilde d,o,u\}$ is normalized over the same agent's known exits:

$$
x_{i,\min}=\min_{k\in\mathcal K_i}x_{ik},\qquad
x_{i,\max}=\max_{k\in\mathcal K_i}x_{ik},\qquad
\widehat x_{ik}=
\begin{cases}
\dfrac{x_{ik}-x_{i,\min}}{x_{i,\max}-x_{i,\min}},&x_{i,\max}>x_{i,\min},\\
0,&x_{i,\max}=x_{i,\min}.
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

### 6. Shared movement policy

Each policy supplies a personal target $\mathbf y_i$, when available, and a leader-eligibility flag $H_i$:

| Policy | Personal target | Eligible leader |
|---|---|---|
| Reactive | $\mathbf y_i=\mathbf q_{ik_i}$, where $k_i=\arg\min_{k\in\mathcal V_i}d_{ik}^{E}$, if $\mathcal V_i\ne\varnothing$ | $H_i=\mathbf1\{\mathcal V_i\ne\varnothing\}$ |
| ToM0 | $\mathbf y_i=\boldsymbol\mu_{ik_i^{\mathrm{ToM0}}}$, if $\mathcal K_i\ne\varnothing$ | $H_i=\mathbf1\{\mathcal K_i\ne\varnothing\}$ |

Define the eligible leaders and the nearest one:

$$
\mathcal F_i=\{j\in\mathcal A\setminus\{i\}:H_j=1\},\qquad
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

Both policies then use

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

Leader selection uses global eligibility and positions, without a follower-to-leader visibility restriction. Following targets the leader's position and does not transfer its exit beliefs. In ToM0, a remembered exit may be preferred over a currently visible one.

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

The simulator uses a fixed step $\Delta t=1/60$. It observes, updates memory, and selects directions using the pre-movement snapshot. It then evaluates the forces for all agents and applies semi-implicit Euler:

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

The cap changes motion when activated. It caps actual velocity after acceleration; desired speed remains fixed.

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

- Exit visibility requires $d_{ik}^{E}>\epsilon$; occupancy counts require $d_{ij}>\epsilon$.
- The cone axis uses division by $\max(\|\boldsymbol\mu_{ik}-\mathbf r_i\|,\epsilon)$; unlike (I3), it does not explicitly set a small nonzero vector to zero.
- Memory age is computed as $\max(t-s_{ik},0)$. The uncertainty denominator uses $\max(\sigma_{\mathrm{ref}},\epsilon)$, and cone area uses $\max(\tfrac12R_o^2\psi,\epsilon)$.
- Min-max normalization treats any criterion range smaller than $\epsilon$ as constant and returns zero.
- Panoramic vision bypasses the angular test when $\phi\ge2\pi-10^{-9}$.

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
- Negative `tom_order` selects reactive behavior. Zero and positive values currently all use ToM0. In YAML, an omitted value defaults to zero; `null`, `none`, and `reactive` select the reactive policy.
- Toggling the mode changes the policy without clearing memory. Reactive mode leaves existing memories untouched; returning to ToM0 reuses them. Resetting or editing the experiment creates a new environment with empty memory.

### 6. Current parameters and initialization

These values instantiate the equations; they are not universal model constants.

| Category | Default values |
|---|---|
| Agent properties | $m_i=1$, $r_i=0.3$, $v_i^0=2$, $\tau_i=1$ |
| Force parameters | $A_i=2000$, $B_i=0.08$, $k=1.2\times10^5$, $\kappa=2.4\times10^5$ |
| Exit perception | $R=40$, $\phi=2\pi$ |
| Memory uncertainty | $\sigma_0=0.1$, $\beta=0.5$, $\sigma_{\mathrm{ref}}=8$ |
| Exit score | $w_d=w_o=w_u=-1$ |
| Occupancy cone | $R_o=12$, $\psi=\pi/3$ |
| Numerical settings | $\Delta t=1/60$, $\epsilon=10^{-12}$, $\gamma=1$, $c=1.5$, $\omega=1.5$ |
| Startup scene | $400$ ToM0 agents in a $100\times100$ room |

Under the paper's unit convention, lengths are meters, time is seconds, mass is kilograms, and forces are newtons. The current mass and relaxation time differ from the original paper's calibration.

The startup file is `two-north-one-south.yaml`, with exits $[(20,0),(23,0)]$, $[(77,0),(80,0)]$, and $[(50,100),(53,100)]$. Perimeter walls are generated around exit gaps; YAML can also specify interior walls and circles. YAML exposes scene geometry, agent count/radius/desired speed, mode, and exit-perception settings. The remaining parameters use code defaults, with programmatic overrides available for environment settings and ToM0 parameters.

Initial velocities are zero. Initial headings have uniformly sampled angles in $[0,2\pi)$. Positions are sampled by sequential rejection in the room with margin $r_i+0.05+1$, pair separation at least $2r_i+0.05$, and obstacle clearance at least $r_i+0.05$. All exit memories start empty.

### 7. Diagnostics and source mapping

Diagnostics do not feed back into the core equations. The simulator records escape counts per exit, evacuation time, agents seeing exits, agents holding beliefs, agents selecting an unseen exit, and agents holding beliefs while seeing no exit. These last two conditions are distinct. It also tracks memory ages, uncertainty, distances to remembered targets, peaks, and accumulated blind-commitment agent-seconds.

Metrics are collected after movement and departure removal, but use visibility and choices from the preceding decision. Ages use the updated clock and distances use the updated positions. The CSV contains `t,remaining,escaped,seeing,with_belief,memory_guided,no_belief`; the remaining diagnostics are available separately. Full definitions and mode-toggle caveats remain in Section 8 of [base-agent-model.md](/home/seb/Vaults/University/26-27/DMAS/model/base-agent-model.md).

| Responsibility | Source |
|---|---|
| Force calculation, policy selection, memory, numerical update | [environment.py](/home/seb/Code/uni/crowd-tom/environment.py) |
| Cone density, normalization, uncertainty, exit score | [tom0.py](/home/seb/Code/uni/crowd-tom/tom0.py) |
| Visibility and line-of-sight tests | [perception.py](/home/seb/Code/uni/crowd-tom/perception.py) |
| Wall/circle geometry and departure tests | [obstacle.py](/home/seb/Code/uni/crowd-tom/obstacle.py), [exit.py](/home/seb/Code/uni/crowd-tom/exit.py) |
| Scene loading, spawning, and perimeter construction | [experiment.py](/home/seb/Code/uni/crowd-tom/experiment.py), [layout.py](/home/seb/Code/uni/crowd-tom/layout.py) |

Physical reference: Helbing, Farkas, and Vicsek, [*Simulating Dynamical Features of Escape Panic*](/home/seb/Vaults/University/26-27/DMAS/papers/panic-model.pdf), equations (1)-(3). Notation and project context: [second DMAS assignment](/home/seb/Vaults/University/26-27/DMAS/assignments/2_DMAS___Crowd_cooperation_ToM.pdf), Section 3.

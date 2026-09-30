# Social-force model with reactive and ToM0 agent behavior

This document gives the equations of the implementation in `/home/seb/Code/uni/crowd-tom`, inspected on 29 September 2026 at commit `23b8468`. The physical starting point is Helbing, Farkas, and Vicsek's *Simulating Dynamical Features of Escape Panic*, equations (1)-(3), pp. 3-4. Notation follows Section 3.1 of the second DMAS assignment: bold $\mathbf r_i$ denotes position, scalar $r_i$ denotes body radius, and $O$ denotes an obstacle.

The code now supports two movement policies: a reactive policy using current exit visibility, and a ToM0 policy using remembered exits, congestion, and uncertainty. Both retain following and random exploration as fallbacks. Desired speed remains constant, and the original paper's average-direction herding and impatience mechanisms are absent. ToM1/ToM2 reasoning and the assignment's additional mental-state velocity adjustment remain proposals.

The mode is selected by `tom_order`: negative values select reactive behavior; zero selects ToM0. Positive values currently execute the same ToM0 policy, without additional reasoning. In experiment YAML, `null`, `none`, and `reactive` select reactive behavior; an omitted field defaults to ToM0. The application starts in ToM0 mode with `two-north-one-south.yaml`. 

## 1. State and notation

Let $t_n=n\Delta t$ and $\mathcal A_n$ be the set of agents still in the environment at step $n$. Agent $i$ has state

$$
\mathbf r_i^n\in\mathbb R^2,\qquad
\mathbf v_i^n\in\mathbb R^2,\qquad
\mathbf h_i^n\in\mathbb R^2,
$$

and fixed parameters $m_i,r_i,v_i^0,\tau_i,A_i,B_i$. Here $\mathbf h_i^n$ is the stored heading used for perception and exploration. It is the previously selected desired direction, rather than the direction of actual velocity. Except for the degenerate cases described below, it has unit length. ToM0 additionally stores an existence flag, last-observation time, and remembered target point for each exit; these are defined in Section 5.

Define the safe normalization used by `perception.unit_rows`:

$$
\mathcal N_\epsilon(\mathbf z)=
\begin{cases}
\mathbf z/\|\mathbf z\|,&\|\mathbf z\|\ge\epsilon,\\
\mathbf 0,&\|\mathbf z\|<\epsilon,
\end{cases}
\qquad \epsilon=10^{-12}.
$$

The base desired velocity is

$$
\mathbf u_i^{\mathrm{base},n}=v_i^0\mathbf e_i^{0,n},
$$

where the behavioral rules below select $\mathbf e_i^{0,n}$. All decisions and forces use the same pre-movement snapshot $(\mathbf r_i^n,\mathbf v_i^n,\mathbf h_i^n)$.

## 2. Physical motion

The continuous-time force law inherited from the panic model is

$$
m_i\frac{d\mathbf v_i}{dt}
=m_i\frac{v_i^0\mathbf e_i^0-\mathbf v_i}{\tau_i}
+\sum_{j\ne i}\mathbf f_{ij}
+\sum_{O\in\mathcal O}\mathbf f_{iO},
\qquad
\frac{d\mathbf r_i}{dt}=\mathbf v_i.
\tag{1}
$$

The implementation truncates agent-agent interactions and applies a speed cap after its numerical velocity update. Thus equation (1) specifies the force calculation; Section 7 specifies the actual discrete dynamics, including these changes. These physical equations are shared by reactive and ToM0 agents.

### 2.1 Agent-agent forces

For $i\ne j$, define

$$
d_{ij}=\|\mathbf r_i-\mathbf r_j\|,\qquad
r_{ij}=r_i+r_j,\qquad
g(x)=\max(0,x),
$$

$$
\mathbf n_{ij}=\frac{\mathbf r_i-\mathbf r_j}{d_{ij}},\qquad
\mathbf t_{ij}=(-n_{ij}^{(2)},n_{ij}^{(1)}),\qquad
\Delta v_{ji}^t=(\mathbf v_j-\mathbf v_i)\cdot\mathbf t_{ij}.
$$

The original interaction is

$$
\mathbf f_{ij}^{\mathrm{HFV}}
=\left[A_i\exp\!\left(\frac{r_{ij}-d_{ij}}{B_i}\right)
+k\,g(r_{ij}-d_{ij})\right]\mathbf n_{ij}
+\kappa\,g(r_{ij}-d_{ij})\Delta v_{ji}^t\mathbf t_{ij}.
\tag{2}
$$

These terms represent social repulsion, normal body compression, and tangential contact friction. To discard negligible interactions, the code uses a threshold $\gamma=$ `F_CUT` and distance cutoff

$$
D_{ij}=r_{ij}-B_i\ln\!\left(\frac{\gamma}{A_i}\right).
\tag{3}
$$

This follows by solving $A_i\exp[(r_{ij}-d_{ij})/B_i]>\gamma$. The force actually used is

$$
\mathbf f_{ij}=
\begin{cases}
\mathbf f_{ij}^{\mathrm{HFV}},&\epsilon\le d_{ij}<D_{ij},\\
\mathbf 0,&\text{otherwise}.
\end{cases}
\tag{4}
$$

The entire pair force is masked, including contact terms. For the current parameters $\gamma<A_i$, the upper cutoff lies beyond contact distance, so ordinary contact interactions are retained. Pairs with almost coincident centers are skipped to avoid an undefined normal. With $r_i=r_j=0.3$, $A_i=2000$, $B_i=0.08$, and $\gamma=1$, $D_{ij}\approx1.208\ \mathrm m$.

The running vectorized calculation evaluates pair distances directly. The older spatial-grid helper is not the neighbor-selection mechanism used by `Environment.tick`. This force cutoff is distinct from the exit view range.

### 2.2 Obstacle forces

Let $d_{iO}$ and $\mathbf n_{iO}$ be the distance and normal returned by the obstacle geometry, and set $\mathbf t_{iO}=(-n_{iO}^{(2)},n_{iO}^{(1)})$. Then

$$
\mathbf f_{iO}
=\left[A_i\exp\!\left(\frac{r_i-d_{iO}}{B_i}\right)
+k\,g(r_i-d_{iO})\right]\mathbf n_{iO}
-\kappa\,g(r_i-d_{iO})(\mathbf v_i\cdot\mathbf t_{iO})\mathbf t_{iO}.
\tag{5}
$$

The friction sign opposes sliding against a stationary obstacle. Obstacle forces have no distance cutoff and are summed over every obstacle, regardless of visual occlusion.

For a finite wall with endpoints $\mathbf a_O,\mathbf b_O$, write $\mathbf s_O=\mathbf b_O-\mathbf a_O$ and

$$
\lambda_{iO}=\operatorname{clip}_{[0,1]}
\left(\frac{(\mathbf r_i-\mathbf a_O)\cdot\mathbf s_O}{\|\mathbf s_O\|^2}\right),
\qquad
\mathbf q_{iO}=\mathbf a_O+\lambda_{iO}\mathbf s_O,
$$

$$
d_{iO}=\|\mathbf r_i-\mathbf q_{iO}\|,\qquad
\mathbf n_{iO}=\frac{\mathbf r_i-\mathbf q_{iO}}{d_{iO}}.
\tag{6}
$$

These expressions apply to nondegenerate walls and nonzero distances. On a wall the code chooses the normalized perpendicular $(-s_O^{(2)},s_O^{(1)})$; a wall whose squared length is below $\epsilon$ is treated as a point, with $(1,0)$ as the normal if the agent coincides with it.

For a circular obstacle of center $\mathbf c_O$ and radius $R_O$,

$$
\mathbf n_{iO}=\frac{\mathbf r_i-\mathbf c_O}{\|\mathbf r_i-\mathbf c_O\|},
\qquad
d_{iO}=\|\mathbf r_i-\mathbf c_O\|-R_O,
\qquad
\mathbf q_{iO}=\mathbf c_O+R_O\mathbf n_{iO}.
\tag{7}
$$

Here the distance is **signed**: it is negative inside the circle. Consequently, $g(r_i-d_{iO})$ increases when an agent penetrates the obstacle. At the circle center the code chooses $\mathbf n_{iO}=(1,0)$. The default scene contains walls only, although the geometry supports circles.

## 3. Exit perception

An exit $E_k$ is a finite segment with endpoints $\mathbf a_k,\mathbf b_k$. Its closest point to agent $i$ is

$$
\mathbf q_{ik}^n=\mathbf a_k+
\operatorname{clip}_{[0,1]}\left(
\frac{(\mathbf r_i^n-\mathbf a_k)\cdot(\mathbf b_k-\mathbf a_k)}
{\|\mathbf b_k-\mathbf a_k\|^2}
\right)(\mathbf b_k-\mathbf a_k).
\tag{8}
$$

==Simplification opportunity — presentation only: equations (6) and (8) repeat the same closest-point projection onto a segment. Define that operation once as $\Pi_S(\mathbf r)$ and write $\mathbf q_{iO}=\Pi_O(\mathbf r_i)$ and $\mathbf q_{ik}=\Pi_{E_k}(\mathbf r_i)$. The projection formula, degenerate-geometry cases, and the segment-intersection arithmetic in Section 7 can sit together in a geometry appendix. This shortens the model description without changing target points, obstacle forces, visibility, or departure tests. Keep finite exit segments: replacing an exit with its midpoint would concentrate targets and could change congestion.==

For nondegenerate exits define

$$
\boldsymbol\delta_{ik}^n=\mathbf q_{ik}^n-\mathbf r_i^n,
\qquad d_{ik}^{E,n}=\|\boldsymbol\delta_{ik}^n\|,
\qquad \mathbf z_{ik}^n=\mathcal N_\epsilon(\boldsymbol\delta_{ik}^n).
$$

Let $R$ be the exit view range and $\phi$ the full field-of-view angle. An exit is visible if its closest point is in range, in the field of view, and unobstructed:

$$
V_{ik}^n=
\mathbf 1\{\epsilon<d_{ik}^{E,n}\le R\}
\,\mathbf 1\{\mathrm{FOV}_{ik}^n\}
\,\mathbf 1\{\neg L_{ik}^n\},
\tag{9}
$$

$$
\mathrm{FOV}_{ik}^n=
\begin{cases}
\mathrm{true},&\phi\ge2\pi-10^{-9},\\
\mathcal N_\epsilon(\mathbf h_i^n)\cdot\mathbf z_{ik}^n
\ge\cos(\phi/2),&\text{otherwise}.
\end{cases}
\tag{10}
$$

==Simplification opportunity — exact for the current panoramic setting: with $\phi=2\pi$, the field-of-view factor is always one. Equations (9)-(10) can therefore reduce to the range test multiplied by the unobstructed-line-of-sight test. Keep the angular formula as an optional extension if restricted vision is an experimental variable. This removes a condition from the default model, but does not justify dropping obstacle occlusion or stored heading: heading is still needed for wandering.==

Here $L_{ik}^n$ is true when an obstacle blocks the sight segment from $\mathbf r_i^n$ to $\mathbf q_{ik}^n$. Wall intersections count only at fractions $\epsilon<u<1-\epsilon$ along this sight segment, with the intersection lying on the wall segment. Parallel or collinear wall segments are ignored by the intersection routine. A circle blocks sight when a boundary-intersection root lies in that same interval; the routine ignores sight segments whose squared length is below $\epsilon$. Agents do not occlude exits.

Visibility tests only the **closest point** on an exit; it does not search the entire exit segment for a visible portion. Also, field of view uses the previous desired heading, even if contact forces have caused actual motion in another direction. %%should ideally search for entire segment?%%

Define the currently visible exits and agents seeing at least one exit:

$$
\mathcal V_i^n=\{k:V_{ik}^n=1\},\qquad
I_i^n=\mathbf 1\{\mathcal V_i^n\ne\varnothing\},\qquad
\mathcal I_n=\{j\in\mathcal A_n:I_j^n=1\}.
\tag{11}
$$

`_seen_exits` is overwritten by current visibility. In reactive mode $\mathcal I_n$ supplies eligible leaders. ToM0 separately maintains exit memory and uses agents with at least one remembered exit as leaders, as specified in Section 5.

## 4. Reactive behavior: reach, follow, or explore

This section specifies `tom_order < 0`. In this mode exit beliefs are neither updated nor used to choose movement.

### 4.1 Reach the nearest visible exit

When $\mathcal V_i^n\ne\varnothing$, select

$$
k_i^n=\underset{k\in\mathcal V_i^n}{\arg\min}\ d_{ik}^{E,n},
\qquad
\mathbf e_i^{\mathrm{exit},n}
=\mathcal N_\epsilon(\mathbf q_{ik_i^n}^n-\mathbf r_i^n).
\tag{12}
$$

Equal distances are resolved by exit-list order. This extends the assignment's nearest-exit rule by restricting selection to currently visible exits. Selection is recalculated each step; it includes no congestion score, route planning, or commitment to a previously selected exit.

### 4.2 Follow the nearest currently informed agent

When $i$ sees no exit, eligible leaders are

$$
\mathcal F_i^n=\mathcal I_n\setminus\{i\}.
$$

If this set is nonempty,

$$
j_i^n=\underset{j\in\mathcal F_i^n}{\arg\min}\
\|\mathbf r_j^n-\mathbf r_i^n\|,
\qquad
\mathbf e_i^{\mathrm{follow},n}
=\mathcal N_\epsilon(\mathbf r_{j_i^n}^n-\mathbf r_i^n).
\tag{13}
$$

Equal distances are resolved by current agent-list order. Following points toward the leader's **position**, rather than copying its velocity or desired direction. An agent that only follows another agent is not itself classified as informed; only direct exit visibility makes an agent eligible to lead.

Leader eligibility uses the simulator's global $I_j^n$ values. There is no range, field-of-view, or obstacle-occlusion test between follower and leader. Thus a follower can target an informed agent outside its view range or behind a wall. This information assumption also applies to ToM0 following, where the eligibility flag instead means that the leader holds an exit belief. The code does not infer knowledge from observed movement. %%restrict?%%

### 4.3 Explore by rotating the previous heading

When no exit is visible and no informed leader exists, draw an independent angular perturbation

$$
\eta_i^n\sim\mathcal N\!\left(0,(\omega\Delta t)^2\right),
\qquad
\mathbf e_i^{\mathrm{wander},n}
=\mathsf R(\eta_i^n)\mathbf h_i^n,
\tag{14}
$$

$$
\mathsf R(\eta)=
\begin{pmatrix}
\cos\eta&-\sin\eta\\
\sin\eta&\cos\eta
\end{pmatrix}.
$$

Here $\omega=$ `wander_turn` controls angular variation. The code passes $\omega\Delta t$ as the Gaussian **standard deviation**, so angular variance per step is $\omega^2\Delta t^2$. This is the exact timestep dependence implemented; it is not a Brownian angular increment with standard deviation proportional to $\sqrt{\Delta t}$.

### 4.4 Combined desired direction

For an environment containing at least one exit,

$$
\boxed{
\mathbf e_i^{0,n}=
\begin{cases}
\mathbf e_i^{\mathrm{exit},n},&\mathcal V_i^n\ne\varnothing,\\
\mathbf e_i^{\mathrm{follow},n},&\mathcal V_i^n=\varnothing,
\ \mathcal F_i^n\ne\varnothing,\\
\mathbf e_i^{\mathrm{wander},n},&\text{otherwise}.
\end{cases}}
\tag{15}
$$

Store $\mathbf h_i^{n+1}=\mathbf e_i^{0,n}$ for perception in the next step. If the environment has **no exits at all**, the code bypasses following and wandering and retains $\mathbf e_i^{0,n}=\mathbf h_i^n$.

The branches are exclusive: seeing an exit takes priority over following, which takes priority over wandering. There is no continuous herding weight or average of neighbors' directions. If a selected leader has the same position as the follower, safe normalization returns zero. The desired direction can therefore be zero in this degenerate case, despite the usual unit-vector convention. Subsequent wandering rotates this zero vector until another rule or repacking restores a direction. During initial packing and repacking after departures, near-zero stored headings are replaced by $(1,0)$.

The reactive movement direction uses the strictly nearest visible exit. The separate `_chosen_exit` diagnostic selects the first visible exit within $10^{-9}$ of the best distance; an almost exact tie can therefore produce a diagnostic exit index different from the target used for movement.

## 5. ToM0: exit memory and scored selection

This section specifies `tom_order >= 0`. ToM0 represents beliefs about the environment, without representing other agents' intentions or nested mental states.

### 5.1 Belief storage and observation updates

For each agent $i$ and exit $k$, store

$$
\mathcal B_{ik}^n=(b_{ik}^n,s_{ik}^n,\boldsymbol\mu_{ik}^n),
\qquad b_{ik}^n\in\{0,1\},
$$

where $b_{ik}^n$ indicates whether the exit has been observed while ToM0 was active, $s_{ik}^n$ is its last recorded observation time, and $\boldsymbol\mu_{ik}^n$ is the closest point on the exit at that observation. Initially all flags are zero; placeholder times and coordinates are also zero and are excluded from selection. Sightings made only in reactive mode do not create or refresh beliefs.

After evaluating current visibility, refresh the belief before scoring:

$$
(b_{ik}^{n,+},s_{ik}^{n,+},\boldsymbol\mu_{ik}^{n,+})=
\begin{cases}
(1,t_n,\mathbf q_{ik}^n),&V_{ik}^n=1,\\
(b_{ik}^n,s_{ik}^n,\boldsymbol\mu_{ik}^n),&V_{ik}^n=0.
\end{cases}
\tag{16}
$$

The superscript $+$ denotes the memory after observation but before movement. Surviving agents carry it into the next step. Beliefs are never forgotten because of age or uncertainty. A remembered target is an agent-specific last-seen point on a segment, not the exit midpoint or a newly computed closest point when the exit is invisible. Seeing an exit again refreshes both time and point.

### 5.2 Memory age and uncertainty

For a known exit, define

$$
a_{ik}^n=\max(t_n-s_{ik}^{n,+},0),\qquad
\sigma_{ik}^n=\sigma_0+\beta\sqrt{a_{ik}^n},\qquad
u_{ik}^n=1-\exp\!\left(-\frac{\sigma_{ik}^n}{\max(\sigma_{\mathrm{ref}},\epsilon)}\right).
\tag{17}
$$

Here $\beta=$ `sigma_alpha`; a distinct symbol avoids confusion with the assignment's future ToM influence parameter $\alpha$. A newly observed exit has $a_{ik}^n=0$ and $\sigma_{ik}^n=\sigma_0$. The bounded penalty $u_{ik}^n$ grows toward one as memory ages. Unknown exits receive $\sigma=\infty$ internally and are excluded by the belief mask.

Despite its name, $\sigma$ is used only to compute a score penalty. The code samples no positional noise, creates no covariance, and does not move $\boldsymbol\mu_{ik}$ between sightings. This implements age-dependent confidence, rather than random drift of the remembered exit location proposed in the earlier ToM0 note.

==Simplification opportunity — combine the uncertainty pipeline: write $s=\max(\sigma_{\mathrm{ref}},\epsilon)$, $\theta=\beta/s$, and a single age penalty $f(a)=1-\exp(-\theta\sqrt a)$. The current penalty is $u=(1-C)+Cf(a)$, where $C=\exp(-\sigma_0/s)>0$. Its common offset and positive scale cancel in the per-agent min-max normalization, so normalized $f$ gives the same scores algebraically whenever the criterion is not treated as constant. The behavioral shape then has one effective parameter, $\theta=0.0625$ at the defaults; $\sigma$ can remain a derived diagnostic. For exact agreement near the numerical cutoff, retain the original constant-criterion test as $C(f_{\max}-f_{\min})<\epsilon$, rather than testing the unscaled range. Retain the nonlinear age function: replacing it with raw age or a rank can change trade-offs between three or more exits.==

### 5.3 Congestion in a cone toward each believed exit

Let $R_o$ and $\psi$ be the occupancy range and full cone angle. Define

$$
\mathbf c_{ik}^n=
\frac{\boldsymbol\mu_{ik}^{n,+}-\mathbf r_i^n}
{\max(\|\boldsymbol\mu_{ik}^{n,+}-\mathbf r_i^n\|,\epsilon)},
\qquad
\mathbf p_{ij}^n=\frac{\mathbf r_j^n-\mathbf r_i^n}{d_{ij}^n}
\quad(d_{ij}^n>\epsilon).
$$

The measured cone density is

$$
o_{ik}^n=
\frac{
\displaystyle\sum_{j\in\mathcal A_n\setminus\{i\}}
\mathbf 1\{\epsilon<d_{ij}^n\le R_o\}
\mathbf 1\{\mathbf p_{ij}^n\cdot\mathbf c_{ik}^n\ge\cos(\psi/2)\}
}
{\max(\tfrac12 R_o^2\psi,\epsilon)}.
\tag{18}
$$

It is a number density of other agents in a sector anchored at the deciding agent, oriented toward the remembered target. The cone always uses its configured radius; it is not shortened at the exit or clipped against room boundaries. Density uses current positions, with no prediction of intentions, velocities, or future crowding. Obstacles do not occlude the counted agents, and the occupancy angle is independent of the exit-perception field of view.

==Simplification opportunity — use cone counts directly for scoring: the area divisor $A_o=\max(\tfrac12R_o^2\psi,\epsilon)$ is the same for every candidate exit. If $c_{ik}$ is the numerator of (18), min-max normalization gives $\widehat o_{ik}=(c_{ik}-c_{i,\min})/(c_{i,\max}-c_{i,\min})$ for a nonconstant criterion. Thus the decision rule does not need density or the area division at all. The cone range and angle still matter because they determine who is counted. To preserve the equality guard exactly, compare the count range with $A_o\epsilon$; at the defaults, any nonzero integer count difference exceeds this threshold. This preserves the normalized congestion contribution, apart from floating-point roundoff.==

### 5.4 Normalize the criteria and choose the highest score

Let $\mathcal K_i^n=\{k:b_{ik}^{n,+}=1\}$ be the known exits. The distance criterion is

$$
\widetilde d_{ik}^n=\|\boldsymbol\mu_{ik}^{n,+}-\mathbf r_i^n\|.
$$

For each criterion $x\in\{\widetilde d,o,u\}$, normalize across the known exits of the same agent:

$$
\widehat x_{ik}^n=
\begin{cases}
\displaystyle\frac{x_{ik}^n-x_{i,\min}^n}{x_{i,\max}^n-x_{i,\min}^n},
&x_{i,\max}^n-x_{i,\min}^n\ge\epsilon,\\
0,&\text{otherwise},
\end{cases}
\quad
x_{i,\min}^n=\min_{k\in\mathcal K_i^n}x_{ik}^n,
\quad
x_{i,\max}^n=\max_{k\in\mathcal K_i^n}x_{ik}^n.
\tag{19}
$$

Unknown entries are returned as zero by the normalization helper, then assigned an ineligible score. The utility is

$$
S_{ik}^n=
\begin{cases}
w_d\widehat d_{ik}^n+w_o\widehat o_{ik}^n+w_u\widehat u_{ik}^n,
&k\in\mathcal K_i^n,\\
-\infty,&\text{otherwise},
\end{cases}
\qquad
k_i^n=\underset{k\in\mathcal K_i^n}{\arg\max}\ S_{ik}^n.
\tag{20}
$$

All three default weights are $-1$, so shorter remembered distance, lower cone density, and lower uncertainty are preferred. Equal scores are resolved by exit-list order. Selection is recalculated every step, with no hysteresis or congestion-based desired-speed change.

With only one known exit, every normalized criterion is zero. Its score remains zero regardless of its absolute age, uncertainty, distance, or congestion, and it remains selected. With two known exits, each criterion whose range is at least $\epsilon$ maps its lower and higher values to zero and one: score differences depend on their relative ordering, rather than the size of the difference. Thus bounded absolute uncertainty is not an abandonment threshold.

### 5.5 Desired direction and fallback

Define the agents with any exit belief by $H_i^n=\mathbf 1\{\mathcal K_i^n\ne\varnothing\}$, and eligible leaders by $\mathcal F_i^{\mathrm{ToM0},n}=\{j\in\mathcal A_n\setminus\{i\}:H_j^n=1\}$. Then

$$
\boxed{
\mathbf e_i^{0,n}=
\begin{cases}
\mathcal N_\epsilon(\boldsymbol\mu_{ik_i^n}^{n,+}-\mathbf r_i^n),
&\mathcal K_i^n\ne\varnothing,\\
\mathcal N_\epsilon(\mathbf r_{j_i^n}^n-\mathbf r_i^n),
&\mathcal K_i^n=\varnothing,\ \mathcal F_i^{\mathrm{ToM0},n}\ne\varnothing,\\
\mathsf R(\eta_i^n)\mathbf h_i^n,&\text{otherwise},
\end{cases}}
\tag{21}
$$

where $j_i^n$ is the nearest eligible leader and $\eta_i^n$ has the distribution in (14). As in reactive mode, no follower-to-leader visibility test is applied. A leader can retain eligibility while no longer seeing any exit; followers do not acquire its beliefs merely by following it.

==Simplification opportunity — one shared movement rule: equations (15) and (21) have the same structure: move toward an available personal target; otherwise move toward the nearest eligible leader; otherwise wander. State this rule once, then define the two policy-specific inputs. Reactive agents use the nearest visible exit and leaders currently seeing an exit; ToM0 agents use the highest-scoring remembered exit and leaders holding any exit belief. This is an exact refactoring of the description and can also guide shared code. Keep those eligibility definitions distinct, since making both modes follow the same leaders would change their information assumptions and behavior.==

In ToM0 mode, seeing an exit refreshes memory but does not force immediate selection of that exit: an unseen remembered exit can receive the higher score. If the selected remembered point is reached exactly, safe normalization returns zero; there is no special arrival, forgetting, or exploration rule to resolve it. Stored heading is updated to the selected direction. As in reactive mode, an environment with no exits bypasses the policy and retains the previous heading.

Switching modes with **T** changes `tom_order` in place. It does not clear existing memories: reactive mode ignores them and leaves them unchanged, and switching back to ToM0 reuses them with their accumulated ages. Resetting or editing the experiment constructs a new environment and clears memory.

## 6. Desired speed and the proposed higher-order ToM connection

In the implementation,

$$
v_i^{0,n}=v_i^0(0),\qquad
\boxed{\mathbf u_i^{\mathrm{base},n}=v_i^0(0)\mathbf e_i^{0,n}}.
\tag{22}
$$

This applies to both modes, with the direction supplied by (15) or (21). No impatience, observed-speed adaptation, or ToM-dependent speed adjustment is performed. Forces still cause actual speed to differ from desired speed.

The assignment proposes the following interface for future reasoning at order $\ell$:

==Simplification opportunity — move the unimplemented extension out of the current model: equations (23)-(24) introduce nested beliefs, an adjustment vector, an influence parameter, and a desired-speed cap that neither implemented policy uses. Move them to a separate future-ToM section or design note. The current specification needs only (22), the selected direction, and the actual-speed cap in (26). This changes no simulated behavior and makes the implemented model substantially easier to distinguish from the research proposal.==

$$
\widetilde{\mathbf u}_i^{(\ell),n}
=\mathbf u_i^{\mathrm{base},n}
+\alpha\,\Delta\mathbf u_i^{(\ell)}(B_i^{(\ell),n}),
\tag{23}
$$

$$
\mathbf e_i^{0,n}
=\frac{\widetilde{\mathbf u}_i^{(\ell),n}}
{\|\widetilde{\mathbf u}_i^{(\ell),n}\|},\qquad
v_i^{0,n}=\min\!\left(v_i^{\max},
\|\widetilde{\mathbf u}_i^{(\ell),n}\|\right).
\tag{24}
$$

For a zero adjusted vector, the assignment specifies zero desired speed and retention of the previous unit direction. Equations (23)-(24) are included only to show where the implemented environmental-belief baseline fits the initial idea. The code now has ToM0 exit beliefs $\mathcal B_{ik}$, but no higher-order mental-state beliefs, $\Delta\mathbf u_i^{(\ell)}$, or influence parameter $\alpha$. It also does not implement the assignment's desired-speed cap $v_i^{\max}$. Its actual-speed cap below has a different role. Setting `tom_order` to 1 or 2 does not implement (23)-(24).

## 7. Numerical dynamics and departure

After determining direction with (15) or (21), compute all accelerations from the pre-movement state:

$$
\mathbf a_i^n=
\frac{v_i^0\mathbf e_i^{0,n}-\mathbf v_i^n}{\tau_i}
+\frac{1}{m_i}\left[
\sum_{j\in\mathcal A_n\setminus\{i\}}\mathbf f_{ij}^n
+\sum_{O\in\mathcal O}\mathbf f_{iO}^n
\right].
\tag{25}
$$

First update velocity, then cap its magnitude:

$$
\widehat{\mathbf v}_i^{n+1}=\mathbf v_i^n+\Delta t\,\mathbf a_i^n,
\qquad v_{i,\mathrm{cap}}=c\,v_i^0,
$$

$$
\mathbf v_i^{n+1}=
\begin{cases}
\widehat{\mathbf v}_i^{n+1},&
\|\widehat{\mathbf v}_i^{n+1}\|\le v_{i,\mathrm{cap}},\\
\displaystyle
\frac{v_{i,\mathrm{cap}}}
{\max(\|\widehat{\mathbf v}_i^{n+1}\|,\epsilon)}
\widehat{\mathbf v}_i^{n+1},&\text{otherwise}.
\end{cases}
\tag{26}
$$

Here $c=$ `speed_limit_factor`. The position update is semi-implicit Euler using the **capped new velocity**:

$$
\boxed{\mathbf r_i^{n+1}=\mathbf r_i^n+\Delta t\,\mathbf v_i^{n+1}}.
\tag{27}
$$

The cap changes physical trajectories whenever activated; it is part of the simulated model, rather than a display setting.

An agent leaves if its movement segment crosses any exit segment. To express the implemented test, define the two-dimensional cross product $\mathbf x\times\mathbf y=x_1y_2-x_2y_1$. For exit $k$, let

$$
\mathbf p_i=\mathbf r_i^{n+1}-\mathbf r_i^n,\qquad
\mathbf s_k=\mathbf b_k-\mathbf a_k,\qquad
\mathbf w_{ik}=\mathbf a_k-\mathbf r_i^n.
$$

When $|\mathbf p_i\times\mathbf s_k|\ge\epsilon$, define

$$
u_{ik}=\frac{\mathbf w_{ik}\times\mathbf s_k}{\mathbf p_i\times\mathbf s_k},
\qquad
z_{ik}=\frac{\mathbf w_{ik}\times\mathbf p_i}{\mathbf p_i\times\mathbf s_k}.
$$

Then

$$
C_i^n=\mathbf 1\left\{\exists k:
|\mathbf p_i\times\mathbf s_k|\ge\epsilon,
\ 0\le u_{ik}\le1,\ 0\le z_{ik}\le1\right\},
\qquad
\mathcal A_{n+1}=\{i\in\mathcal A_n:C_i^n=0\}.
\tag{28}
$$

Endpoint intersections count; parallel or collinear movement does not. Crossing is checked for every exit regardless of whether it was selected or visible, and in either crossing direction. Removal depends on the agent center's trajectory, not on its body radius or proximity to the opening. If one movement intersects multiple exits, it is counted only for the first intersected exit in list order. Surviving agents retain positions, velocities, memories, chosen-exit indices, and visibility rows when arrays are repacked. Visibility is recalculated on the next step.

Simulation time increments by $\Delta t$ after each nonempty tick, even if nobody departs. Once all agents have left, subsequent ticks return without advancing time.

## 8. Recorded model quantities

==Simplification opportunity — separate evaluation from dynamics: none of the quantities in this section feeds back into movement or belief updates. The main model description can give evacuation time, per-exit escape counts, and the memory-guided count as its primary outputs, while placing the remaining diagnostic formulas and pre-/post-step timing details in an evaluation appendix. They can all remain recorded in the implementation, so this simplification loses no results and changes no trajectories. Do not merge memory-guided and blind-committed counts: they measure different conditions, as (32) shows.==

The update counts escaped agents per exit. If $X_k^n$ is its cumulative count, then

$$
X_k^{n+1}=X_k^n+
\sum_{i\in\mathcal A_n}\mathbf 1\{i\text{ first crosses exit }k\text{ in list order during step }n\},
\qquad
X^{n+1}=\sum_k X_k^{n+1}=N_0-|\mathcal A_{n+1}|.
\tag{29}
$$

The evacuation time is stored at the first completed step leaving no agents:

$$
T_{\mathrm{evac}}=\min\{t_{n+1}:|\mathcal A_{n+1}|=0\}.
\tag{30}
$$

It is initially unset and is recorded on a tick that removes the final agent. The implementation does not assign a within-step intersection time.

For post-step diagnostics, let $\mathcal A'=\mathcal A_{n+1}$, retain $V_{ik}^n$ and the stored chosen exit $k_i^n$ from the decision snapshot, and define $H_i^+=\mathbf 1\{\exists k:b_{ik}^{n,+}=1\}$. Then

$$
N_{\mathrm{seeing}}^{n+1}=\sum_{i\in\mathcal A'}\mathbf 1\{\exists k:V_{ik}^n=1\},\qquad
N_{\mathrm{belief}}^{n+1}=\sum_{i\in\mathcal A'}H_i^+,\qquad
N_{\mathrm{no\ belief}}^{n+1}=|\mathcal A'|-N_{\mathrm{belief}}^{n+1},
\tag{31}
$$

$$
N_{\mathrm{memory}}^{n+1}=\sum_{\substack{i\in\mathcal A'\\k_i^n\ge0}}
\mathbf 1\{V_{ik_i^n}^n=0\},\qquad
N_{\mathrm{blind}}^{n+1}=\sum_{i\in\mathcal A'}
H_i^+\,\mathbf 1\{\nexists k:V_{ik}^n=1\}.
\tag{32}
$$

`memory_guided` counts agents selecting an unseen exit, including agents that can see a different exit. `blind_committed` counts belief holders seeing no exit at all. They are therefore different quantities. The code computes these from stored flags and does not check the active mode; after a mode toggle, belief-related statistics can include memory retained from earlier ToM0 steps.

It also records

$$
P_{\mathrm{memory}}=\max_n N_{\mathrm{memory}}^{n+1},\qquad
Q_{\mathrm{blind}}=\sum_n N_{\mathrm{blind}}^{n+1}\Delta t,
\tag{33}
$$

and the maximum distance from a surviving selecting agent to its stored target, plus the peak of that maximum over time. $Q_{\mathrm{blind}}$ is measured in agent-seconds. Mean memory age and mean $\sigma$ are calculated over agents with a stored selected-exit index, using the updated clock $t_{n+1}$; commit distances use the updated positions $\mathbf r_i^{n+1}$. Thus displayed visibility and choices describe pre-movement perception, while displayed ages and distances describe the post-movement state. In particular, a just-seen belief can already have diagnostic age $\Delta t$. In reactive mode these age/distance statistics still run for selected exits, even if their belief flags are false, in which case they use placeholder memory values and do not describe a real exit memory. Likewise, `no_belief` counts absent memory, rather than proving that an agent has never seen an exit in reactive mode.

`plots.History` records the time series `t,remaining,escaped,seeing,with_belief,memory_guided,no_belief` once per distinct simulation time. The other diagnostics are returned by `tom0_metrics` but are not columns in the current CSV. The UI saves these series to `experiments/last-metrics.csv` on request or at completion; resets clear the history.

## 9. Current default scene and parameters

Scene values now come from experiment YAML, loaded by `experiment.load_experiment` and built by `experiment.build_environment`. The table gives the bundled startup scene `two-north-one-south.yaml`, together with the physical and behavioral defaults. They are simulation settings rather than the original paper's calibration.

| Quantity | Symbol | Current value |
|---|---|---|
| Initial number of agents | $N_0$ | $400$ |
| Room dimensions | $L_x,L_y$ | $100\times100$ world units |
| Behavior mode | `tom_order` | $0$ (ToM0) |
| Agent mass | $m_i$ | $1.0$ |
| Body radius | $r_i$ | $0.3$ |
| Desired speed | $v_i^0$ | $2.0$ |
| Relaxation time | $\tau_i$ | $1.0$ |
| Social repulsion strength | $A_i$ | $2000$ |
| Social repulsion length | $B_i$ | $0.08$ |
| Compression coefficient | $k$ | $1.2\times10^5$ |
| Sliding coefficient | $\kappa$ | $2.4\times10^5$ |
| Agent-force cutoff | $\gamma$ | $1$ |
| Exit view range | $R$ | $40$ |
| Full field of view | $\phi$ | $2\pi$ ($360^\circ$) |
| Wander parameter | $\omega$ | $1.5$ |
| Actual-speed multiplier | $c$ | $1.5$; speed cap $3.0$ |
| Physics timestep | $\Delta t$ | $1/60$ |
| Initial uncertainty scale | $\sigma_0$ | $0.1$ |
| Age growth coefficient | $\beta$ (`sigma_alpha`) | $0.5$ |
| Uncertainty reference scale | $\sigma_{\mathrm{ref}}$ | $8.0$ |
| Score weights | $w_d,w_o,w_u$ | $-1,-1,-1$ |
| Occupancy cone range | $R_o$ | $12.0$ |
| Occupancy full cone angle | $\psi$ | $\pi/3$ ($60^\circ$) |

Under the original paper's SI convention, positions and radii are in meters, speeds in meters per second, mass in kilograms, $A_i$ and $\gamma$ in newtons, $B_i$ in meters, $k$ in $\mathrm{kg\,s^{-2}}$, and $\kappa$ in $\mathrm{kg\,m^{-1}\,s^{-1}}$. The code has no unit-conversion layer. Its mass $1.0$ and relaxation time $1.0$ differ from the paper's $80\ \mathrm{kg}$ and $0.5\ \mathrm s$, even though force coefficients match. These defaults should therefore be reported as simulation parameters rather than claimed to reproduce the paper's calibration.

The startup scene has exits from $(20,0)$ to $(23,0)$ (NW), $(77,0)$ to $(80,0)$ (NE), and $(50,100)$ to $(53,100)$ (south). Seven generated perimeter-wall segments enclose the room while leaving these openings. There are no interior obstacles in this scene. The previous two-exit scene is still available as `north-south.yaml`; `four-doors.yaml` adds west/east doors, and `pillar.yaml` adds an interior wall and a circle.

In general, an exit's YAML `center` is measured along its specified boundary; its endpoints are `center ± width/2`, clipped to that side's extent. Perimeter wall generation merges overlapping exit gaps before generating solid segments. Width, height, mode, agent count/radius/desired speed, exit-perception range/angle, exits, and interior geometry are configured in YAML. Force coefficients, mass, relaxation time, wandering, speed-limit multiplier, and ToM0 scoring parameters currently use code defaults; the YAML loader does not read overrides for these values. `Environment` accepts a `ToM0Params` object for programmatic overrides.

Initial velocities are zero and headings are independently sampled as $(\cos\theta_i,\sin\theta_i)$ with $\theta_i\sim U[0,2\pi)$. Positions are sampled by sequential rejection in $[M,L_x-M]\times[M,L_y-M]$, where $M=r_i+0.05+1.0$, requiring center separation at least $2r_i+0.05$ and obstacle-surface distance at least $r_i+0.05$. At the default radius this is $[1.35,98.65]^2$, separation $0.65$, and obstacle clearance $0.35$. Accepted positions are consequently not independent uniform samples. No initial body overlap is allowed. Initial belief flags are false, and selected-exit indices are $-1$.

To interpret $\sigma$ as a length scale, $\sigma_0$ and $\sigma_{\mathrm{ref}}$ share length units and $\beta$ has units of length per square root of time. Cone density has units of agents per area; all score inputs are dimensionless after min-max normalization.

At the default timestep, the wandering angular standard deviation is $1.5/60=0.025$ radians per step. The $360^\circ$ field of view disables angular restrictions, but range and obstacle occlusion still restrict exit visibility.

## 10. Equation-to-code references

| Model component | Active implementation |
|---|---|
| Forces and synchronous update, (1), (25)-(27) | [environment.py](/home/seb/Code/uni/crowd-tom/environment.py:76), `tick`, `_clip_speeds` |
| Pair forces and cutoff, (2)-(4) | [environment.py](/home/seb/Code/uni/crowd-tom/environment.py:354), `_social_forces` |
| Obstacle forces, (5) | [environment.py](/home/seb/Code/uni/crowd-tom/environment.py:389), `_obstacle_forces` |
| Obstacle geometry, (6)-(8) | [obstacle.py](/home/seb/Code/uni/crowd-tom/obstacle.py:67), `Wall.distances`, `Circle.distances` |
| Visibility, (9)-(11) | [perception.py](/home/seb/Code/uni/crowd-tom/perception.py:81), `visible_exit_mask`, intersection routines |
| Reactive policy, following, wandering, (12)-(15) | [environment.py](/home/seb/Code/uni/crowd-tom/environment.py:269), `_desired_directions`, `_follow_or_wander` |
| Belief refresh and age, (16)-(17) | [environment.py](/home/seb/Code/uni/crowd-tom/environment.py:269), `_desired_directions` |
| Cone density, (18) | [tom0.py](/home/seb/Code/uni/crowd-tom/tom0.py:39), `cone_density` |
| Normalization and score, (19)-(20) | [tom0.py](/home/seb/Code/uni/crowd-tom/tom0.py:22), `minmax_masked`, `uncertainty`, `utilities` |
| ToM0 heading, (21) | [tom0.py](/home/seb/Code/uni/crowd-tom/tom0.py:89), `headings_from_beliefs`, and environment fallback |
| Departure, (28) | [exit.py](/home/seb/Code/uni/crowd-tom/exit.py:14), `is_crossed_many` |
| Counts and diagnostics, (29)-(33) | [environment.py](/home/seb/Code/uni/crowd-tom/environment.py:141), `tom0_metrics`, `tick`, `_update_tom0_peaks` |
| Experiment loading and spawning | [experiment.py](/home/seb/Code/uni/crowd-tom/experiment.py:75), `load_experiment`, `spawn_positions`, `build_environment` |
| Exit and perimeter geometry | [layout.py](/home/seb/Code/uni/crowd-tom/layout.py:25), `exit_segment`, `perimeter_walls` |
| Startup scene | [two-north-one-south.yaml](/home/seb/Code/uni/crowd-tom/experiments/two-north-one-south.yaml) |
| Fixed timestep and controls | [main.py](/home/seb/Code/uni/crowd-tom/main.py:208), `run_simulation` |
| Time-series export | [plots.py](/home/seb/Code/uni/crowd-tom/plots.py:25), `History` |

`Agent.update_acceleration` and `Agent.update_desired_direction` retain an older nearest-exit implementation. The running loop calls `Environment.tick`, so those older methods do not specify the current behavior.

## References

1. Helbing, D., Farkas, I., Vicsek, T. (2000). *Simulating Dynamical Features of Escape Panic*. Supplied [panic-model.pdf](/home/seb/Vaults/University/26-27/DMAS/papers/panic-model.pdf), physical equations (1)-(3), pp. 3-4.
2. Pusch, S., Damian, M., Lin Yang, Y., Dinu, R. M. *Group 14 - Crowd Escape ToM*. Supplied [second DMAS assignment](/home/seb/Vaults/University/26-27/DMAS/assignments/2_DMAS___Crowd_cooperation_ToM.pdf), Section 3.1, pp. 3-6, and proposed ToM interface in Section 3.2, p. 9.


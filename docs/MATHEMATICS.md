# Mathematical Guarantees & Exact Invariants in `eproc`

Every kernel in `eproc` implements a non-asymptotic, finite-sample statistical or algebraic guarantee designed for **live, non-stationary, multi-agent, and distributed systems**. No heuristic asymptotic normal approximations are used where exact martingale or conformal bounds exist.

---

## 1. Anytime-Valid Betting E-Processes (`eproc_bet_t`)

### Setting
Let $X_1, X_2, \dots \in [a, b]$ be an adapted sequence of observations with conditional expectations $\mu_t = \mathbb{E}[X_t \mid \mathcal{F}_{t-1}]$. Rescale linearly to $Y_t = \frac{\operatorname{clip}(X_t, a, b) - a}{b - a} \in [0, 1]$ and $m_0 = \frac{\mu_0 - a}{b - a} \in (0, 1)$.

### Theorem 1 (Exact Ville Supermartingale Validity)
Let $\lambda_t$ be $\mathcal{F}_{t-1}$-measurable (strictly predictable) satisfying:
- **Upper-tail test ($H_0: \mu_t \le m_0 \;\forall t$):** $\lambda_t \in [0,\; c_{\max}/m_0]$ with $c_{\max} \in (0, 1)$.
- **Lower-tail test ($H_0: \mu_t \ge m_0 \;\forall t$):** $\lambda_t \in [-c_{\max}/(1 - m_0),\; 0]$.

Define the wealth process $M_0 = 1$ and
$$M_t = \prod_{s=1}^{t} \bigl(1 + \lambda_s (Y_s - m_0)\bigr).$$
Then for every $s \ge 1$, $1 + \lambda_s (Y_s - m_0) \ge 1 - c_{\max} > 0$, and under $H_0$,
$$\mathbb{E}\!\left[M_t \mid \mathcal{F}_{t-1}\right] = M_{t-1}\bigl(1 + \lambda_t(\mathbb{E}[Y_t \mid \mathcal{F}_{t-1}] - m_0)\bigr) \le M_{t-1}.$$
Hence $(M_t)_{t \ge 0}$ is a nonnegative supermartingale with $M_0 = 1$. By **Ville's inequality (1939)**, for *any* stopping time $\tau$ (including adversarial data-dependent continuous peeking at every tick):
$$\mathbb{P}_{H_0}\!\left(\exists\, t \ge 1 : M_t \ge \frac{1}{\alpha}\right) \le \alpha, \qquad p_t^{\text{anytime}} = \min\!\left(1,\; \frac{1}{\max_{1 \le s \le t} M_s}\right).$$

For **two-sided testing** ($H_0: \mu_t = m_0$), `eproc` splits initial capital 50/50 across an upper supermartingale $M_t^+$ and a lower supermartingale $M_t^-$:
$$M_t^{\pm} = \frac{1}{2} M_t^+ + \frac{1}{2} M_t^-,$$
which remains an exact nonnegative supermartingale with $\mathbb{E}_{H_0}[M_t^{\pm}] \le 1$.

### Online Newton Step (ONS) Bet Selection
To approach the log-optimal Kelly bet fraction without knowing the alternative mean in advance, `eproc` updates $\lambda_t$ using the predictable ONS strategy of Waudby-Smith & Ramdas (JRSS-B 2024):
$$z_t = \frac{Y_t - m_0}{1 + \lambda_t (Y_t - m_0)}, \qquad A_t = A_{t-1} + z_t^2 \;\;(A_0 = 1), \qquad \lambda_{t+1} = \Pi_{[\lambda_{\min}, \lambda_{\max}]}\!\left(\lambda_t + \frac{2}{2 - \ln 3} \cdot \frac{z_t}{A_t}\right).$$
Because $\lambda_{t+1}$ depends only on $\mathcal{F}_t$, strict predictability holds identically.

### Distributed E-Value Combination (Vovk & Wang 2021)
- **Independent shards/agents:** $\log E_{\text{prod}} = \sum_{k=1}^K \log E_k$ (exact product e-value).
- **Arbitrarily dependent / overlapping shards:** $E_{\text{avg}} = \frac{1}{K}\sum_{k=1}^K E_k$ (the unique admissible symmetric merging function under arbitrary dependence), computed stably via log-sum-exp minus $\ln K$.

---

## 2. Closed-Form Time-Uniform Empirical-Bernstein Confidence Sequences (`eproc_cs_t`)

### Theorem 2 (Variance-Adaptive Bounded Confidence Sequence — Waudby-Smith & Ramdas 2024)
For any adapted stream $Y_t \in [0, 1]$ with common conditional mean $\mu = \mathbb{E}[Y_t \mid \mathcal{F}_{t-1}]$, predictable plug-in mean $\hat{\mu}_{t-1} \in [0, 1]$, and predictable $\lambda_t \in [0, 1)$, let $\psi_E(\lambda) = -\ln(1 - \lambda) - \lambda$. By Fan–Grama–Liu (2015), $\exp(\lambda \xi - \psi_E(\lambda)\xi^2) \le 1 + \lambda \xi$ for all $\xi \ge -1$. Setting $\xi_t = Y_t - \hat{\mu}_{t-1} \ge -1$:
$$\mathbb{E}\!\left[\exp\!\bigl(\lambda_t (Y_t - \mu) - \psi_E(\lambda_t)(Y_t - \hat{\mu}_{t-1})^2\bigr) \;\middle|\; \mathcal{F}_{t-1}\right] \le e^{\lambda_t(\hat{\mu}_{t-1} - \mu)}\bigl(1 + \lambda_t(\mu - \hat{\mu}_{t-1})\bigr) \le 1.$$
Applying the same bound symmetrically to $1 - Y_t \in [0, 1]$ and applying Ville's inequality at $\alpha / 2$ per side yields the **exact $O(1)$-time, $O(1)$-memory closed-form confidence sequence**:
$$L_t = \frac{\sum_{s=1}^t \lambda_s Y_s - \ln(2/\alpha) - \sum_{s=1}^t \psi_E(\lambda_s)(Y_s - \hat{\mu}_{s-1})^2}{\sum_{s=1}^t \lambda_s}, \qquad U_t = \frac{\sum_{s=1}^t \lambda_s Y_s + \ln(2/\alpha) + \sum_{s=1}^t \psi_E(\lambda_s)(Y_s - \hat{\mu}_{s-1})^2}{\sum_{s=1}^t \lambda_s}.$$
Taking the running intersection $L_t^* = \max_{s \le t} L_s$ and $U_t^* = \min_{s \le t} U_s$ preserves exact simultaneous coverage:
$$\mathbb{P}\!\left(\forall\, t \ge 1 : \mu \in [L_t^*,\, U_t^*]\right) \ge 1 - \alpha.$$

### Multi-Arm Best-Agent Racing (`eproc_race_t`)
For $K$ competing agents/prompts/models, running $K$ Empirical-Bernstein confidence sequences each at level $\alpha / K$ gives simultaneous validity over all $K$ arms and all times $t \ge 1$ by the union bound:
$$\mathbb{P}\!\left(\exists\, k \in \{1,\dots,K\},\; \exists\, t \ge 1 : \mu_k \notin [L_{k,t}^*,\, U_{k,t}^*]\right) \le \alpha.$$
Eliminating any arm $j$ as soon as $U_{j,t}^* < \max_{i} L_{i,t}^* - \delta$ guarantees that the true best arm (up to indifference margin $\delta \ge 0$) is eliminated with probability at most $\alpha$, while saving $3\times$–$10\times$ evaluation calls on clearly dominated arms.

---

## 3. Multi-Rate Crossing-Free Online Conformal & `<1 KB` Abelian Monoid

### Multi-Rate `AgACI` + Conformal PID + PAVA Projection (`eproc_conformal_t`)
For $R \le 5$ simultaneous target miscoverage levels $\alpha_1 > \alpha_2 > \dots > \alpha_R$ (ascending coverage, e.g. $50\%, 80\%, 90\%, 95\%, 99\%$):
1. Each level $r$ runs 4 Adaptive Conformal Inference (ACI) experts with learning rates $\gamma \in \{0.005, 0.02, 0.08, 0.20\}$ aggregated online by pinball-loss exponential weights (Zaffran et al. 2022), plus a bounded Conformal PID saturation integrator (Angelopoulos et al. 2023):
   $$\alpha_{t+1}^{(r, e)} = \alpha_t^{(r, e)} + \gamma_e \bigl(\alpha_r - \mathbf{1}\{s_t > q_t^{(r, e)}\}\bigr).$$
2. Raw nonconformity scores live in an L1-resident circular buffer mirrored by an incrementally sorted array (`O(W)` binary search + `memmove`, zero `malloc`, zero `qsort`).
3. Every update finishes with **Pool-Adjacent-Violators (PAVA)** isotonic projection across $q_t^{(1)}, \dots, q_t^{(R)}$, mathematically guaranteeing **zero quantile crossing**: $q_t^{(1)} \le q_t^{(2)} \le \dots \le q_t^{(R)}$.

### Exact Abelian 640-Byte Distributed `ConformalMonoid` (`eproc_cmonoid_t`)
Nonconformity scores are mapped via a deterministic monotone signed log-transform into 64 bins with **exact `uint64_t` integer counts** `bin_counts[64]`:
- **Strict Commutativity & Associativity:** `(bin_counts_A + bin_counts_B) == (bin_counts_B + bin_counts_A)` and `((A + B) + C) == (A + (B + C))` with **zero floating-point rounding drift** across arbitrary tree-reduction topologies (`AVX2` `_mm256_add_epi64`).
- **Exact Leave-One-Out (`LOO`) Group Inverse:** For any agent $i$ contributing `local_i` to `global = merge_j(local_j)`, subtracting `local_i` via `_mm256_sub_epi64` recovers the exact peer-only distribution `global - local_i` with underflow guards preventing invalid subtraction.

---

## 4. Byzantine-Safe Multi-Agent Fusion (`SafeBOA` & `EchoPool`)

### Scale-Free BOA + 3-Gate Ville Non-Inferiority Guard (`eproc_boa_t`)
Given $M$ expert predictions $\hat{y}_{t,1}, \dots, \hat{y}_{t,M}$ and designated safe anchor $a$:
1. Losses are normalized by an $\mathcal{F}_{t-1}$-predictable EMA scale $S_{t-1}$ and Huber-clipped at $c_{\text{clip}} S_{t-1}$ (default $c_{\text{clip}} = 25$), bounding the per-tick influence of arbitrary Byzantine/NaN/$10^9$ spikes to zero catastrophic distortion.
2. Weights follow Wintenberger (2017) second-order Bernstein Online Aggregation with adaptive per-expert rates $\eta_{t,k} = \min\!\bigl(\frac{1}{2}, \sqrt{\frac{\ln M}{1 + \sum_{s < t} \tilde{\ell}_{s,k}^2}}\bigr)$.
3. Three parallel Ville e-processes compare the normalized loss difference between BOA and the anchor (`sup_e`, `deg_e`, `rec_e`). If `deg_e` crosses $1/\alpha_{\text{deg}}$, `SafeBOA` atomically falls back to the anchor and publishes the updated state via a **Boehm-2012 lock-free C11 seqlock** (`0` torn reads under concurrent inference).

### Echo-Deduplicated Log-Odds Pool (`eproc_echopool_t`)
Each submitted evidence item `(claim_id, evidence_id, source_id, llr)` is hashed into a fixed-memory open-addressing provenance table:
- **Zero Echo Double-Counting:** Reprints of any previously seen `(claim_id, evidence_id)` are dropped immediately (`accepted = 0`).
- **Zero Overwrite Churn:** Novel items accumulate commutatively in Kahan-compensated log-odds space rather than last-write-wins overwrites.
- **Online Peer Reliability Attenuation:** Each source $w$ tracks resolved outcomes via Beta-Bernoulli counts $(\alpha_w, \beta_w)$; its attenuation weight $r_w = \operatorname{clip}(2\frac{\alpha_w}{\alpha_w + \beta_w} - 1,\; 0,\; 1)$ smoothly shrinks uncalibrated or adversarial peers to zero weight.

---

## 5. Causal Horvitz-Thompson Subsampling (`VarOpt`, `OnlineRLS`, `SiftVOI`)

- **`VarOpt` (Cohen et al. 2011):** Maintains exact inclusion threshold $\tau$ and Horvitz-Thompson adjusted weights $\tilde{w}_i = \max(w_i, \tau)$ for all $k$ retained items, yielding $\mathbb{E}[\sum_{i \in S} \tilde{w}_i] = \sum_{i=1}^n w_i$ and $\mathbb{E}[\sum_{i \in S} \tilde{w}_i v_i] = \sum_{i=1}^n w_i v_i$ with provably minimal subset-sum variance.
- **`OnlineRLS` (Cohen–Musco–Pachocki):** Maintains $A_t^{-1} = (\lambda I + \sum_{s \le t} w_s x_s x_s^T)^{-1}$ via exact Sherman-Morrison rank-1 updates, sampling item $t$ with $p_t = \operatorname{clip}(c \cdot x_t^T A_{t-1}^{-1} x_t,\; p_{\min},\; 1)$ and unbiased weight $w_t = \mathbf{1}\{\text{keep}\} / p_t$.
- **`SiftVOI`:** Evaluates per-channel noise-normalized Gaussian likelihood ratios inside a hierarchical site × channel Bayesian posterior (eliminating loud-channel false-flag traps) and selects the query maximizing exact **Mutual Information (Expected Information Gain in nats)** per unit attention cost via 5-point Gauss-Hermite quadrature accelerated by exact Gaussian maximum-entropy branch-and-bound pruning:
  $$I(Z; Y) = h(Y) - h(Y \mid Z) \le \min\!\left(H(Z),\; \frac{1}{2}\ln\!\bigl(1 + p(1-p)\,\mathrm{SNR}^2\bigr)\right) \le \frac{1}{2}\,p(1-p)\,\mathrm{SNR}^2.$$

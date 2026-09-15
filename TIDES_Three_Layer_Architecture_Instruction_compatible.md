# TIDES: Three-Layer Inference Architecture
## Algorithm Instruction

### 1. Overview

TIDES organizes temporal network inference into three consecutive layers:

$$
\boxed{
\text{Temporal segmentation}
\;\longrightarrow\;
\text{Edge-space reconstruction}
\;\longrightarrow\;
\text{Physical compression}
}
$$

The three layers answer three distinct questions:

$$
\boxed{
\begin{aligned}
\textbf{Step 1:}\;&
\text{When does the vector field change?}
\$$1mm]
\textbf{Step 2:}\;&
\text{Which edge-space vector fields are compatible with the trajectory?}
\$$1mm]
\textbf{Step 3:}\;&
\text{Which physical topology--dynamics representation gives the shortest description?}
\end{aligned}
}
$$

The complete inference chain is

$$
\boxed{
x(t)
\longrightarrow
\tau_{1:K-1}
\longrightarrow
\mathcal B_\varepsilon
\longrightarrow
\mathcal H^\varepsilon
\longrightarrow
(W,\Theta)_H^\star .
}
$$

Step 1 extracts temporal structure from the observed trajectory.  
Step 2 reconstructs the full family of edge-space vector fields compatible with the observations.  
Step 3 introduces a physical hypothesis \(H\), factorizes the edge-space family into topology and dynamics, and uses minimum description length (MDL) to select the shortest admissible representation.

The core representation hierarchy is

$$
\boxed{
x(t)
\longleftarrow
B_{1:K}
\longleftarrow
(W,\Theta),
}
$$

and TIDES performs inference in the reverse direction:

$$
\boxed{
\text{trajectory}
\longrightarrow
\text{edge-space solution family}
\longrightarrow
\text{physical representation family}
\longrightarrow
\text{minimum-description representation}.
}
$$

---

### 2. Local interaction library and edge-space lifting

Consider a network with \(N\) nodes and \(M\) candidate edges. The local interaction law is represented in a prescribed library

$$
\boxed{
\Psi
=
\{\psi_1,\ldots,\psi_L\}.
}
$$

The corresponding interaction coefficients are collected as

$$
\boxed{
\Theta
=
\{\theta_1,\ldots,\theta_L\}.
}
$$

At temporal stage \(r\), both topology and interaction strengths may in general vary. The effective interaction law carried by edge \(m\) is written as

$$
\boxed{
u_m^{(r)}
=
w_m^{(r)}
\sum_{\ell=1}^{L}
\theta_\ell^{(r)}\psi_\ell .
}
$$

Here

$$
w_m^{(r)}
$$

is the structural amplitude of edge \(m\), while

$$
\theta_\ell^{(r)}
$$

is the coefficient of the \(\ell\)-th library function. The library functions \(\psi_\ell\) themselves remain fixed.

Moving the edge amplitude inside the sum gives

$$
u_m^{(r)}
=
\sum_{\ell=1}^{L}
w_m^{(r)}\theta_\ell^{(r)}\psi_\ell .
$$

This motivates the lifted coefficients

$$
\boxed{
B_{m\ell}^{(r)}
=
w_m^{(r)}\theta_\ell^{(r)} .
}
$$

Therefore

$$
\boxed{
u_m^{(r)}
=
\sum_{\ell=1}^{L}
B_{m\ell}^{(r)}\psi_\ell .
}
$$

Collecting all lifted coefficients gives the stage-wise edge-space matrix

$$
\boxed{
B^{(r)}
=
\left[B_{m\ell}^{(r)}\right]
\in\mathbb R^{M\times L}.
}
$$

The full temporal edge-space field is

$$
\boxed{
B_{1:K}
=
\left(
B^{(1)},\ldots,B^{(K)}
\right).
}
$$

The lifting

$$
\boxed{
(w^{(r)},\Theta^{(r)})
\longmapsto
B^{(r)}
}
$$

separates the observational reconstruction problem from the subsequent physical factorization problem. Once lifted, the trajectory is linear in the unknown coefficients \(B^{(r)}\).

---

### 3. Step 1 — Temporal segmentation

Given the observed trajectory

$$
x(t),
$$

Step 1 detects changepoints of the effective vector field,

$$
\boxed{
\tau_{1:K-1}
=
(\tau_1,\ldots,\tau_{K-1}).
}
$$

These changepoints partition the trajectory into \(K\) temporal stages,

$$
I_1,\ldots,I_K.
$$

The Step-1 map is therefore

$$
\boxed{
x(t)
\longrightarrow
\tau_{1:K-1}.
}
$$

Each stage subsequently provides observational constraints on one stage-wise edge-space field \(B^{(r)}\).

---

### 4. Step 2 — Edge-space vector-field family reconstruction

After lifting, the reconstruction problem is linear in \(B^{(r)}\).

For stage \(r\), let

$$
b_r
=
\operatorname{vec}\!\left(B^{(r)}\right).
$$

The observational equation is

$$
\boxed{
y_r
=
A_r b_r+\eta_r,
}
$$

where \(y_r\) contains the observed vector-field information and \(A_r\) is constructed from the observed states and the evaluations of the library functions \(\psi_\ell\).

Stacking all stages gives

$$
y
=
\begin{bmatrix}
y_1\\
\vdots\\
y_K
\end{bmatrix},
\qquad
b
=
\begin{bmatrix}
b_1\\
\vdots\\
b_K
\end{bmatrix},
$$

and hence

$$
\boxed{
y
=
Ab+\eta .
}
$$

Define the relative reconstruction residual

$$
\boxed{
\rho(B_{1:K})
=
\frac{
\left\|
y-A\,\operatorname{vec}(B_{1:K})
\right\|_2
}{
\|y\|_2
}.
}
$$

Given the observational uncertainty floor \(\varepsilon\), the complete feasible edge-space family is

$$
\boxed{
\mathcal B_\varepsilon
=
\left\{
B_{1:K}:
\rho(B_{1:K})\le\varepsilon
\right\}.
}
$$

Thus Step 2 is

$$
\boxed{
(x(t),\tau_{1:K-1},\Psi)
\longrightarrow
\mathcal B_\varepsilon .
}
$$

#### Geometry of the solution family

Let the effective singular-value decomposition of the stacked observation operator be

$$
A
=
U_q\Sigma_qV_q^\top,
$$

where \(q=\operatorname{rank}(A)\). Let

$$
b_0
=
A^+y
$$

be a minimum-residual reference solution, with residual

$$
r_0
=
y-Ab_0.
$$

Any coefficient vector can be decomposed as

$$
\boxed{
b
=
b_0+V_q a+V_0 z,
}
$$

where \(V_0\) spans \(\ker A\). Since \(r_0\) is orthogonal to \(\operatorname{col}(A)\),

$$
\boxed{
\|y-Ab\|_2^2
=
\|r_0\|_2^2
+
\|\Sigma_q a\|_2^2 .
}
$$

Hence the feasible family has the geometric form

$$
\boxed{
\text{identified-space ellipsoid}
\times
\text{observational nullspace}.
}
$$

The identifiable directions encode combinations of edge-space coefficients constrained by the trajectory, while the null directions encode observationally unresolved degrees of freedom.

#### Temporal-change view

For any feasible temporal field \(B_{1:K}\), define

$$
\boxed{
\Delta B^{(r)}
=
B^{(r+1)}-B^{(r)},
\qquad
r=1,\ldots,K-1.
}
$$

The temporal-change sequence is

$$
\boxed{
\Delta B_{1:K-1}
=
\left(
\Delta B^{(1)},\ldots,\Delta B^{(K-1)}
\right).
}
$$

Applying this difference operation to the full feasible family gives

$$
\boxed{
\Delta\mathcal B_\varepsilon
=
\left\{
\Delta B_{1:K-1}:
B_{1:K}\in\mathcal B_\varepsilon
\right\}.
}
$$

Thus Step 2 provides two complementary representations of the same observational information:

$$
\boxed{
\mathcal B_\varepsilon
\quad\text{and}\quad
\Delta\mathcal B_\varepsilon.
}
$$

The first is the absolute edge-space field family; the second is its temporal-change view.

---

### 5. Step 3 — Physical representation under hypothesis \(H\)

A physical hypothesis is denoted by

$$
\boxed{H}.
$$

The set of all topology--dynamics representations satisfying that hypothesis is denoted by

$$
\boxed{\mathcal H}.
$$

A candidate representation specifies topology coefficients and interaction coefficients and generates a temporal edge-space field

$$
B_{1:K}(W,\Theta).
$$

The representations compatible with the Step-2 field family are

$$
\boxed{
\mathcal H^\varepsilon
=
\left\{
(W,\Theta)\in\mathcal H:
B_{1:K}(W,\Theta)\in\mathcal B_\varepsilon
\right\}.
}
$$

Step 3 therefore searches the physical representation family induced by \(H\) inside the observational family reconstructed in Step 2.

Two central hypotheses are:

#### Fixed dynamics

$$
\boxed{
H_{\rm FD}:
\qquad
\Theta^{(1)}
=
\cdots
=
\Theta^{(K)}
=
\Theta.
}
$$

Then

$$
\boxed{
B^{(r)}
=
W^{(r)}\Theta^\top,
}
$$

and

$$
\boxed{
\Delta B^{(r)}
=
\Delta W^{(r)}\Theta^\top.
}
$$

The temporal changes share a common dynamical direction \(\Theta\), while the structural amplitudes vary with stage.

#### Fixed topology

$$
\boxed{
H_{\rm FT}:
\qquad
W^{(1)}
=
\cdots
=
W^{(K)}
=
W.
}
$$

Then

$$
\boxed{
B^{(r)}
=
W{\Theta^{(r)}}^\top,
}
$$

and

$$
\boxed{
\Delta B^{(r)}
=
W\,\Delta{\Theta^{(r)}}^\top.
}
$$

The temporal changes share a common structural direction \(W\), while the interaction coefficients vary with stage.

The same observational objects

$$
\mathcal B_\varepsilon,
\qquad
\Delta\mathcal B_\varepsilon
$$

therefore support different hypothesis-dependent factorization paths.

---

### 6. Conditional MDL

For a representation

$$
(W,\Theta)\in\mathcal H^\varepsilon,
$$

define the conditional description length

$$
\boxed{
L_{\rm MDL}(W,\Theta\mid H,\Psi)
=
L_{\rm struct}
+
L_{\rm expr}
+
L_{\rm prec}.
}
$$

The selected representation is

$$
\boxed{
(W,\Theta)_H^\star
=
\arg\min_{(W,\Theta)\in\mathcal H^\varepsilon}
L_{\rm MDL}(W,\Theta\mid H,\Psi).
}
$$

#### Structural code

\(L_{\rm struct}\) describes the structural degrees of freedom left unspecified by \(H\).

For \(H_{\rm FD}\), let the changed-edge support at transition \(k\) be

$$
S_k,
\qquad
E_k=|S_k|.
$$

With \(M\) candidate edges, an elementary support code is

$$
\boxed{
L_{\rm struct}^{(k)}
=
\log(M+1)
+
\log{M\choose E_k}.
}
$$

If the transition supports are encoded independently,

$$
\boxed{
L_{\rm struct}
=
\sum_{k=1}^{K-1}
\left[
\log(M+1)
+
\log{M\choose E_k}
\right].
}
$$

Additional structural blocks, such as a baseline support when required by \(H\), are encoded analogously.

#### Expression code

Assume the interaction library is polynomial and that the maximum available polynomial resolution is \(P\). Let

$$
\Psi_P
=
\{\psi_1,\ldots,\psi_{L_P}\}.
$$

For an independent interaction law

$$
f
=
\sum_{\ell\in J}
\theta_\ell\psi_\ell,
$$

define

$$
s=|J|,
$$

and let

$$
p
=
\max_{\ell\in J}\deg\psi_\ell
$$

be the largest active polynomial degree. Define

$$
L_p
=
\#\{\psi_\ell:\deg\psi_\ell\le p\}.
$$

A hierarchical code for the active polynomial expression is

$$
\boxed{
L_{\rm expr}(f)
=
\log P
+
\log L_p
+
\log
\left[
{L_p\choose s}
-
{L_{p-1}\choose s}
\right].
}
$$

Under \(H_{\rm FD}\), the interaction law is shared across all stages and is therefore encoded once.

More generally,

$$
\boxed{
L_{\rm expr}
=
\sum_a
L_{\rm expr}(f_a),
}
$$

where the sum runs over the independent interaction-law objects implied by \(H\).

#### Precision code

Let \(Q\) be the number of free numerical coefficients in a candidate physical representation. For a \(q\)-bit quantized representation, define

$$
\boxed{
q^\star
=
\min
\left\{
q:
\exists\,
(\widetilde W,\widetilde\Theta)_q
\text{ such that }
B_{1:K}(\widetilde W,\widetilde\Theta)
\in
\mathcal B_\varepsilon
\right\}.
}
$$

The numerical precision cost is

$$
\boxed{
L_{\rm prec}
=
L_{\mathbb N}(q^\star)
+
Qq^\star\log2.
}
$$

The total MDL therefore balances

$$
\boxed{
\text{structural complexity}
+
\text{functional complexity}
+
\text{numerical precision}.
}
$$

---

### 7. Example search strategy under \(H_{\rm FD}\)

Under the fixed-dynamics hypothesis,

$$
B^{(r)}
=
W^{(r)}\Theta^\top.
$$

The unknowns are the shared interaction law \(\Theta\) and the stage-dependent structural amplitudes \(W^{(r)}\).

For fixed \(\Theta\), the reconstruction in \(W_{1:K}\) is linear. Therefore define the profiled residual

$$
\boxed{
E(\Theta)
=
\min_{W_{1:K}}
\rho\!\left(
B_{1:K}(W_{1:K},\Theta)
\right).
}
$$

The nonlinear global search can then be carried out primarily in the lower-dimensional interaction-law space.

#### 7.1 Feasible-basin search

Starting from an initial interaction law \(\Theta_0\), use multiscale block basin hopping.

At iteration \(t\):

1. Select a coefficient block

$$
J\subseteq\{1,\ldots,L\}.
$$

2. Draw a multiscale perturbation

$$
\boxed{
\Theta'_{\rm raw}
=
\Theta_t+\delta,
\qquad
\delta_J
=
\sigma D_J z,
}
$$

where \(z\) is a random direction, \(\sigma\) is the jump scale, and \(D_J\) provides sensitivity scaling.

3. Starting from \(\Theta'_{\rm raw}\), locally minimize the profiled objective \(E(\Theta)\).

4. During each objective evaluation, solve

$$
\boxed{
W^\star_{1:K}(\Theta)
=
\arg\min_{W_{1:K}}
\rho\!\left(
B_{1:K}(W_{1:K},\Theta)
\right).
}
$$

5. Let the resulting local minimum be \(\Theta'\). Accept an improving basin according to

$$
E(\Theta')<E(\Theta_t).
$$

6. Repeat until

$$
\boxed{
E(\Theta)\le\varepsilon.
}
$$

This produces an observationally feasible representation

$$
\left(
W^\star_{1:K},\Theta
\right)
\in
\mathcal H^\varepsilon_{\rm FD}.
$$

Random block size and multiscale jump amplitude allow single-coordinate, small-block, and global moves within one proposal family.

#### 7.2 MDL compression inside the feasible family

Once a feasible physical basin has been located, search proceeds over representations that remain inside

$$
\mathcal H^\varepsilon_{\rm FD}.
$$

Candidate structural moves include

$$
\boxed{
\text{edge add},
\qquad
\text{edge drop},
\qquad
\text{edge swap},
}
$$

while interaction-law moves include

$$
\boxed{
\text{atom add},
\qquad
\text{atom drop},
\qquad
\text{atom swap}.
}
$$

For each candidate move:

1. modify the physical representation;
2. refit the associated continuous coefficients;
3. test whether

$$
B_{1:K}(W,\Theta)\in\mathcal B_\varepsilon;
$$

4. evaluate

$$
\Delta L_{\rm struct},
\qquad
\Delta L_{\rm expr},
\qquad
\Delta L_{\rm prec};
$$

5. combine them as

$$
\boxed{
\Delta L_{\rm MDL}
=
\Delta L_{\rm struct}
+
\Delta L_{\rm expr}
+
\Delta L_{\rm prec}.
}
$$

A candidate with

$$
\Delta L_{\rm MDL}<0
$$

gives a shorter admissible physical description.

#### 7.3 Precision profiling

For a fixed structural and expression model, first determine the continuous optimum. Let its coefficient vector be \(c^\star\), its minimum squared residual be \(R^\star\), and let

$$
T
=
\varepsilon^2\|y\|_2^2
$$

be the allowed squared-error threshold.

If the relevant local design matrix has Gram matrix

$$
G=A^\top A,
$$

then the remaining quantization budget is

$$
\boxed{
T-R^\star.
}
$$

A quantized coefficient vector \(\widetilde c\) remains feasible whenever

$$
\boxed{
(\widetilde c-c^\star)^\top
G
(\widetilde c-c^\star)
\le
T-R^\star.
}
$$

The smallest quantization depth satisfying this condition determines \(q^\star\) and therefore \(L_{\rm prec}\).

Search proposals may prioritize low-order atoms, large residual correlations, large conditional structural gains, weak active edges, or multiscale block moves. These proposal mechanisms guide exploration, while the final selection criterion remains

$$
\boxed{
L_{\rm MDL}
}
$$

subject to observational feasibility.

---

### 8. Complete TIDES algorithm

The complete three-layer architecture is

$$
\boxed{
\begin{aligned}
\textbf{Step 1: Temporal segmentation}
\qquad&
x(t)
\longrightarrow
\tau_{1:K-1},
\$$2mm]
\textbf{Step 2: Edge-space reconstruction}
\qquad&
(x(t),\tau_{1:K-1},\Psi)
\longrightarrow
\mathcal B_\varepsilon,
\\
&
\mathcal B_\varepsilon
\longrightarrow
\Delta\mathcal B_\varepsilon,
\$$2mm]
\textbf{Step 3: Physical compression}
\qquad&
(\mathcal B_\varepsilon,
\Delta\mathcal B_\varepsilon,
H)
\longrightarrow
\mathcal H^\varepsilon
\\
&\longrightarrow
(W,\Theta)_H^\star.
\end{aligned}
}
$$

The central lifting relation is

$$
\boxed{
B_{m\ell}^{(r)}
=
w_m^{(r)}\theta_\ell^{(r)},
}
$$

which converts topology--dynamics products into linearly reconstructable edge-space coefficients.

The resulting architecture separates three objects:

$$
\boxed{
\text{temporal structure}
\quad\longrightarrow\quad
\text{edge-space observational family}
\quad\longrightarrow\quad
\text{physical minimum-description representation}.
}
$$

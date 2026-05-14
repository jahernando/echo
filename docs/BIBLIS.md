# Bibliografía — comparación de muestras

Métodos relevantes para el problema central de `echo`: dadas dos muestras `{x_i}` y `{y_j}` con `x, y ∈ ℝ^d`, **(a)** decidir si provienen de la misma distribución, y **(b)** identificar regiones de `ℝ^d` donde difieren.

Cada entrada incluye: referencia canónica, idea central, formulación esencial, fortalezas/limitaciones, e implementación si la conozco.

---

## Índice

1. Tests clásicos univariantes
2. Tests multivariantes basados en distancia
3. Tests basados en grafos / vecinos más cercanos
4. Tests basados en clasificadores
5. Density ratio y detección local
6. Dataset shift en ML
7. Métodos específicos de física de partículas (HEP)
8. Enfoques bayesianos
9. Librerías y recursos
10. Roadmap orientativo para `echo`

---

## 1. Tests clásicos univariantes

Sólo aplicables marginal a marginal en multivariante. Sirven como baseline y diagnóstico rápido.

### 1.1. Kolmogorov–Smirnov (KS)

**Ref**: Kolmogorov 1933; Smirnov 1948. Cualquier texto de estadística no paramétrica.

**Idea**: Estadístico `D = sup_x |F_n(x) − G_m(x)|`, máxima diferencia absoluta entre las CDFs empíricas. Bajo H₀ (igualdad de distribuciones) sigue una distribución de Kolmogorov tabulada.

**Fortalezas**: Sencillo, paramétrico-libre, distribución del estadístico independiente de la forma de F bajo H₀ (test de distribución libre).

**Limitaciones**: Más sensible cerca del centro que en las colas. Sólo 1D — sus extensiones multivariantes (Peacock, Fasano–Franceschini) son costosas y poco usadas. Subóptimo cuando las diferencias son en las colas.

**Implementación**: `scipy.stats.ks_2samp`.

### 1.2. Anderson–Darling (2-sample)

**Ref**: Scholz & Stephens, *K-sample Anderson–Darling tests*, JASA 1987.

**Idea**: Variante ponderada del KS que enfatiza las colas vía un peso `1/(F(x)(1−F(x)))`. Más potente cuando la diferencia está en los extremos.

**Limitaciones**: 1D igualmente. Distribución del estadístico requiere tabulación o bootstrap.

**Implementación**: `scipy.stats.anderson_ksamp`.

### 1.3. Cramér–von Mises (2-sample)

**Ref**: Anderson, *On the distribution of the two-sample Cramér–von Mises criterion*, Ann. Math. Stat. 1962.

**Idea**: Integra la diferencia cuadrática entre CDFs en lugar de tomar el supremo. `T = ∫ (F_n − G_m)² dH_{n+m}`. Más robusto que KS frente a outliers, similar potencia global.

**Implementación**: `scipy.stats.cramervonmises_2samp`.

### 1.4. χ² (Pearson) / G-test

**Ref**: Pearson 1900; Wilks 1938 para el G-test.

**Idea**: Binar el espacio, contar eventos por bin, comparar contrastando contra una hipótesis nula vía `χ² = Σ (O_i − E_i)² / E_i`. El G-test es la versión basada en log-verosimilitud (`2 Σ O_i log(O_i/E_i)`), preferible asintóticamente.

**Limitaciones**: Depende fuertemente del binning. En multivariante el número de bins explota (curse of dimensionality). Requiere `E_i ≳ 5` por bin para validez asintótica.

**Implementación**: `scipy.stats.chisquare`; binning manual con NumPy.

---

## 2. Tests multivariantes basados en distancia

Familia central para nuestro problema. Construyen estadísticos sobre distancias entre puntos en `ℝ^d` directamente.

### 2.1. Maximum Mean Discrepancy (MMD)

**Ref**: Gretton, Borgwardt, Rasch, Schölkopf, Smola. *A Kernel Two-Sample Test*. JMLR 13:723–773 (2012). [arXiv:0805.2368](https://arxiv.org/abs/0805.2368).

**Idea**: Embed las distribuciones en un Reproducing Kernel Hilbert Space (RKHS). MMD es la distancia entre las medias en ese RKHS. Si el kernel es *characteristic* (e.g. RBF gaussiano), MMD = 0 ⟺ P = Q.

**Formulación**:
```
MMD²(P, Q) = E_P[k(x,x')] + E_Q[k(y,y')] − 2 E_{P,Q}[k(x,y)]
```
Estimador U-statistic insesgado. Test de permutaciones para p-value. La *witness function* `f(z) = E_P[k(x,z)] − E_Q[k(y,z)]` indica dónde difieren P y Q.

**Fortalezas**: No paramétrico, multivariante directo, fundamento teórico sólido, witness function da pista de localización.

**Limitaciones**: Sensible a la elección del bandwidth σ del kernel. Pierde potencia en dimensión alta. Coste O(n²) — escalado limitado a `n ≲ 10⁴`.

**Implementación**: `torch-two-sample`, manual con NumPy/JAX. Heurística σ: mediana de distancias.

### 2.2. Energy distance (Székely–Rizzo)

**Ref**: Székely & Rizzo. *Energy statistics: A class of statistics based on distances*. J. Stat. Plann. Inference 143(8):1249–1272 (2013). Trabajo original en *InterStat* 2004.

**Idea**: `E(P, Q) = 2 E‖X−Y‖ − E‖X−X'‖ − E‖Y−Y'‖` con `X, X' ~ P` y `Y, Y' ~ Q`. E = 0 ⟺ P = Q en `ℝ^d`. Caso especial de MMD con kernel `k(x,y) = −‖x−y‖`.

**Fortalezas**: Sin hiperparámetros (a diferencia de MMD-RBF). Interpretable como balance entre disparidades inter- e intra-muestra. Permutación da p-value exacto.

**Limitaciones**: Coste O(n²). En dimensiones muy altas la distancia euclídea pierde discriminación.

**Implementación**: `dcor`, `scipy` (parcial), o manual.

### 2.3. Wasserstein / Sinkhorn divergence

**Ref**: Cuturi. *Sinkhorn distances: Lightspeed computation of optimal transport*. NeurIPS 2013. [arXiv:1306.0895](https://arxiv.org/abs/1306.0895). Para revisión: Peyré & Cuturi, *Computational Optimal Transport*, 2019.

**Idea**: Wasserstein-p mide el coste mínimo de transportar masa de P a Q bajo una métrica base. `W_p(P, Q) = (inf_π ∫ ‖x−y‖^p dπ(x,y))^{1/p}`. Sinkhorn regulariza con entropía → cálculo O(n²) en lugar de O(n³).

**Fortalezas**: Métrica genuina, geometría intuitiva (mass-transport), gradiente diferenciable (útil para learning). El *transport plan* dice qué masa va de dónde a dónde — pista de localización.

**Limitaciones**: Coste alto sin regularización. Convergencia lenta en dim alta. Sinkhorn divergence introduce bias en función del parámetro ε.

**Implementación**: `POT` (Python Optimal Transport), `geomloss` (PyTorch).

### 2.4. MMD-D — Learning Deep Kernels for Two-Sample Tests

**Ref**: Liu, Xu, Lu, Zhang, Gretton, Sutherland. *Learning Deep Kernels for Non-Parametric Two-Sample Tests*. ICML 2020. [arXiv:2002.09116](https://arxiv.org/abs/2002.09116).

**Idea**: Aprende un kernel parametrizado por una red neuronal — `k_φ(x,y) = (1−ε) k_RBF(φ(x), φ(y)) + ε k_RBF(x,y)` — optimizando el ratio test-power / variance del MMD. Supera a C2ST y a MMD con kernel fijo en regímenes de muestra moderada.

**Fortalezas**: Adaptativo a la geometría del problema, mucho mejor en alta dimensión.

**Limitaciones**: Requiere split de datos (train/test del kernel) o data splitting con su coste de potencia. Riesgo de overfitting con muestras pequeñas.

**Implementación**: [github.com/fengliu90/DK-for-TST](https://github.com/fengliu90/DK-for-TST).

### 2.5. MMD-FUSE — agregación de kernels sin data splitting

**Ref**: Biggs, Schrab, Gretton. *MMD-FUSE: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting*. NeurIPS 2023. [arXiv:2306.08777](https://arxiv.org/abs/2306.08777).

**Idea**: Agrega MMDs calculados con muchos kernels (familia parametrizada) en un único estadístico válido sin sacrificar datos a un split de selección. Resuelve el dilema de "qué kernel elegir" combinándolos todos con control de tipo I.

**Fortalezas**: Robusto a la elección de kernel, válido uniformemente sobre la familia, no consume datos en validación.

**Implementación**: Código con el paper.

---

## 3. Tests basados en grafos / vecinos más cercanos

Construyen un grafo sobre la unión de las dos muestras y miden si las etiquetas (muestra A o B) están bien mezcladas.

### 3.1. Friedman–Rafsky (FR)

**Ref**: Friedman & Rafsky. *Multivariate generalizations of the Wald–Wolfowitz and Smirnov two-sample tests*. Ann. Stat. 7(4):697–717 (1979).

**Idea**: Construir el Minimum Spanning Tree (MST) sobre la unión `{x_i} ∪ {y_j}`. Contar `R` = número de aristas que conectan puntos de muestras distintas. Si P = Q, R debería ser alto (puntos bien mezclados). Bajo H₀ tiene media y varianza calculables.

**Fortalezas**: Multivariante directo, sin hiperparámetros, no paramétrico.

**Limitaciones**: Sensible a empates en distancias. Construcción del MST es O(n² log n). En dim muy alta el MST pierde discriminación.

**Implementación**: `scipy.spatial.minimum_spanning_tree` + lógica manual.

### 3.2. Henze–Penrose / k-NN tests

**Ref**: Henze. *A multivariate two-sample test based on the number of nearest neighbor type coincidences*. Ann. Stat. 16:772–783 (1988). Schilling, JASA 1986.

**Idea**: Para cada punto contar cuántos de sus k vecinos más cercanos son de la misma muestra. Estadístico = fracción de pares "mismo-mismo" entre k-NN. Si P = Q, esa fracción es predecible bajo H₀.

**Fortalezas**: Simple, multivariante, escala mejor que MST (kd-tree).

**Limitaciones**: Elección de k. Distribución asintótica conocida sólo bajo ciertos supuestos.

**Implementación**: `scipy.spatial.cKDTree` + cómputo manual.

---

## 4. Tests basados en clasificadores

Para nosotros una familia especialmente útil: el clasificador aprende **dónde** difieren las muestras, no sólo si difieren. Su score por punto es interpretable.

### 4.1. Classifier Two-Sample Test (C2ST)

**Ref**: Lopez-Paz & Oquab. *Revisiting Classifier Two-Sample Tests*. ICLR 2017. [arXiv:1610.06545](https://arxiv.org/abs/1610.06545).

**Idea**: Etiquetar `x_i → 0` y `y_j → 1`, mezclar y entrenar un clasificador binario. Si P ≠ Q, el clasificador alcanzará accuracy > 0.5 en test. Test: contrastar accuracy contra Binomial(0.5) bajo H₀.

**Fortalezas**: Multivariante por construcción. El clasificador da una *función* (probabilidad por punto) → interpretable, localizable. Funciona con cualquier estimador (BDT, NN, etc.).

**Limitaciones**: Pierde potencia respecto a tests basados en distancia cuando P ≈ Q. Requiere split train/test del clasificador. Calibración del score importa si se quiere likelihood ratio.

**Implementación**: Cualquier clasificador (sklearn, xgboost, torch) + lógica manual.

### 4.2. Witness Two-Sample Test

**Ref**: Kübler, Jitkrittum, Schölkopf, Muandet. *A Witness Two-Sample Test*. AISTATS 2022. [arXiv:2102.05573](https://arxiv.org/abs/2102.05573).

**Idea**: Aprender la *witness function* del MMD `f(z) = E_P k(x,z) − E_Q k(y,z)` en datos de train; evaluar la diferencia de su media en test → estadístico asintóticamente normal. La forma de la witness function es interpretable: dónde es positiva domina P, dónde negativa domina Q.

**Fortalezas**: Combina potencia kernel + interpretabilidad local. Distribución asintótica normal simplifica el p-value.

**Limitaciones**: Requiere split train/test. Elección del kernel.

### 4.3. AutoML Two-Sample Test

**Ref**: Kübler, Stimper, Buchholz, Muandet, Schölkopf. *AutoML Two-Sample Test*. NeurIPS 2022. [arXiv:2206.08843](https://arxiv.org/abs/2206.08843).

**Idea**: Usar regresión AutoML (e.g. AutoGluon) sobre target `±1` como witness function. Provablemente óptima bajo squared loss. Llave en mano para no expertos.

**Fortalezas**: Sin tuning, fácil de aplicar. Combina interpretabilidad de C2ST con poder de AutoML.

**Implementación**: Código público con el paper.

### 4.4. CARL — Calibrated likelihood ratio learning

**Ref**: Cranmer, Pavez, Louppe. *Approximating Likelihood Ratios with Calibrated Discriminative Classifiers*. 2015. [arXiv:1506.02169](https://arxiv.org/abs/1506.02169).

**Idea**: Entrenar un clasificador para distinguir muestras de P y Q; el score `s(x)` se relaciona con el likelihood ratio vía `r(x) = s(x)/(1−s(x))` *si* el clasificador está calibrado. Aproxima `r(x) = p(x)/q(x)` directamente. Muy usado en física para simulation-based inference.

**Fortalezas**: Da un density ratio que es justo el objeto de interés en localización. Trasero teórico riguroso.

**Limitaciones**: Calibración no trivial — el clasificador puede ser preciso pero mal calibrado. Sensible a class imbalance.

### 4.5. Learning Likelihood Ratios con clasificadores neuronales

**Ref**: Rizvi, Pettee, Nachman. *Learning Likelihood Ratios with Neural Network Classifiers*. JHEP 02 (2024) 136. [arXiv:2305.10500](https://arxiv.org/abs/2305.10500).

**Idea**: Comparación sistemática de funciones de pérdida (BCE, MLC, SQR, MSE) y parametrizaciones para obtener un likelihood ratio calibrado a partir de un clasificador. Identifica qué combinaciones son robustas y cuáles colapsan.

**Fortalezas**: Manual práctico actualizado de cómo extraer un LR fiable de un clasificador profundo.

---

## 5. Density ratio y detección local

Métodos que estiman `r(x) = p(x)/q(x)` directamente, sin pasar por densidades individuales. El ratio mismo localiza diferencias.

### 5.1. KLIEP — Kullback–Leibler Importance Estimation Procedure

**Ref**: Sugiyama, Suzuki, Nakajima, Kashima, von Bünau, Kawanabe. *Direct importance estimation for covariate shift adaptation*. Ann. Inst. Stat. Math. 60(4):699–746 (2008).

**Idea**: Modela `r(x) = Σ_l α_l φ_l(x)` con base de kernels gaussianos y optimiza los `α_l` maximizando la log-likelihood de `r(x) q(x)` sobre `p(x)` (i.e. KL[p||q·r]).

**Fortalezas**: Estima `r` directamente — sin estimar p y q por separado, lo que en alta dim es prohibitivo.

**Limitaciones**: Bandwidth del kernel y regularización deben ajustarse.

### 5.2. uLSIF — Unconstrained Least-Squares Importance Fitting

**Ref**: Kanamori, Hido, Sugiyama. *A least-squares approach to direct importance estimation*. JMLR 10:1391–1445 (2009).

**Idea**: Misma filosofía que KLIEP pero con pérdida cuadrática y forma cerrada. Mucho más rápido que KLIEP, validación cruzada eficiente.

**Implementación**: `densratio` (Python). Libro de referencia: Sugiyama, Suzuki, Kanamori, *Density Ratio Estimation in Machine Learning*, Cambridge UP 2012.

### 5.3. AUGUST — Resolution-based interpretable test

**Ref**: AUGUST: An Interpretable, Resolution-based Two-sample Test. NEJSDS 2024.

**Idea**: Descompone la diferencia distribucional en estadísticos ortogonales asociados a expansiones binarias (resoluciones). Cada componente rechazada apunta a una región/resolución específica del espacio. Diseñado para multivariante de dimensión baja-moderada.

**Fortalezas**: Salida interpretable y jerárquica — no sólo "difieren" sino "difieren en esta región y a esta resolución".

**Limitaciones**: Escalado limitado en dim muy alta.

### 5.4. Interpretable Model Drift Detection

**Ref**: *Interpretable Model Drift Detection*. arXiv:2503.06606 (2024–25).

**Idea**: Test de hipótesis consciente de interacciones entre features con garantías de potencia. Atribuye el shift a features individuales y a interacciones específicas.

**Fortalezas**: Combina detección + atribución. Garantías teóricas.

### 5.5. Additive Tree Models for Density Ratios

**Ref**: *Two-sample Comparison through Additive Tree Models for Density Ratios*. arXiv:2508.03059 (2025).

**Idea**: Gradient boosting de árboles para `r(x) = p(x)/q(x)` con una *balancing loss* nueva. La estructura del árbol particiona el espacio en regiones de ratio alto/bajo — partición natural e interpretable.

**Fortalezas**: Interpretable (árboles), localización explícita, escalable.

---

## 6. Dataset shift en ML

Visión ML del mismo problema cuando una muestra es "train" y otra "test/deployment".

### 6.1. Failing Loudly

**Ref**: Rabanser, Günnemann, Lipton. *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*. NeurIPS 2019. [arXiv:1810.11953](https://arxiv.org/abs/1810.11953).

**Idea**: Estudio comparativo masivo de métodos de detección de shift bajo distintos tipos (covariate, prior, concept) e intensidades. Encuentra que la combinación más robusta es reducción de dimensionalidad (PCA o BBSD — Black-Box Shift Detection con un clasificador pre-entrenado) + KS multivariante o MMD.

**Fortalezas**: Benchmark práctico, recomendaciones operativas claras.

**Implementación**: Conceptos integrados en `alibi-detect`, `evidently`, `frouros`.

---

## 7. Métodos específicos de física de partículas

La comunidad HEP ha desarrollado herramientas propias para comparar datos vs simulación o señal vs sideband. Aplicables a `echo` con cambios mínimos.

### 7.1. Aslan–Zech energy test

**Ref**: Aslan & Zech. *Statistical energy as a tool for binning-free, multivariate goodness-of-fit tests, two-sample comparison and unfolding*. Nucl. Instrum. Methods A 537:626–636 (2005).

**Idea**: Versión HEP de energy distance, usando `Φ(r) = −log r` o `1/r` como potencial (analogía electrostática). Estadístico:
```
T = (1/n²) Σ_{i<j} Φ(|x_i−x_j|) + (1/m²) Σ_{i<j} Φ(|y_i−y_j|) − (1/nm) Σ_{i,j} Φ(|x_i−y_j|)
```
P-value por permutación.

**Fortalezas**: Sin binning, multivariante, fácil de implementar. Muy usado en HEP.

**Limitaciones**: Coste O((n+m)²). Sensible al kernel `Φ`.

### 7.2. CWoLa — Classification Without Labels

**Ref**: Metodiev, Nachman, Thaler. *Classification without labels: Learning from mixed samples in high energy physics*. JHEP 10 (2017) 174. [arXiv:1708.02949](https://arxiv.org/abs/1708.02949).

**Idea**: Si dos muestras son mezclas distintas de las mismas dos clases subyacentes, un clasificador entrenado a distinguirlas aprende implícitamente la frontera entre las clases subyacentes. Base del "CWoLa hunting" para anomalías en sidebands.

**Fortalezas**: Aprovecha estructura de los datos, sin necesidad de simulación de la señal.

### 7.3. CATHODE — Classifying Anomalies Through Outer Density Estimation

**Ref**: Hallin et al. *Classifying anomalies through outer density estimation*. Phys. Rev. D 106:055006 (2022). [arXiv:2109.00546](https://arxiv.org/abs/2109.00546).

**Idea**: Entrena un *normalizing flow* condicional en una variable de resonancia sobre las sidebands; muestrea background sintético en la región de señal; clasifica datos vs background sintético. Sucesor moderno de CWoLa hunting.

**Implementación**: Código público.

### 7.4. CURTAINs / LaCathode

**Ref**: Raine et al. *CURTAINs: Constructing Unobserved Regions by Transforming Adjacent Intervals*. [arXiv:2203.09470](https://arxiv.org/abs/2203.09470) (2022). LaCathode: *Resonant anomaly detection without background sculpting*. Phys. Rev. D 2023. [arXiv:2210.14924](https://arxiv.org/abs/2210.14924).

**Idea**: Flow-based para transportar entre sidebands y región de señal. LaCathode construye el score en el espacio latente del flow para evitar sculpting de la variable resonante.

### 7.5. FETA / FlowSALAD

**Ref**: *FETA: Flow-Enhanced Transportation for Anomaly Detection*. [arXiv:2212.11285](https://arxiv.org/abs/2212.11285) (2022–23).

**Idea**: Aprende un reweighting/transport por *normalizing flow* desde Monte Carlo a datos en sidebands; aplica en la región de señal. Heredero directo del enfoque SALAD para comparación data-vs-MC.

### 7.6. NPLM — New Physics Learning Machine

**Ref**: Grosso, Letizia, Pierini, Wulzer. *Goodness of fit by Neyman–Pearson testing*. SciPost Phys. 16:123 (2024). *Robust resonant anomaly detection with NPLM*. [arXiv:2501.01778](https://arxiv.org/abs/2501.01778) (2025).

**Idea**: Goodness-of-fit multivariante como estimación de log-likelihood-ratio in-sample con un clasificador regularizado. Da estadístico global + score por evento. Menor variancia de hiperparámetros que BDT-CWoLa.

**Implementación**: Basada en Falkon (kernel methods escalables). Código público.

---

## 8. Enfoques bayesianos

### 8.1. Posterior Predictive Checks (PPC)

**Ref**: Gelman, Meng, Stern. *Posterior predictive assessment of model fitness via realized discrepancies*. Statistica Sinica 6:733–760 (1996). Texto: Gelman et al., *Bayesian Data Analysis*, 3ª ed.

**Idea**: Si un modelo bayesiano genera réplicas `y_rep ~ p(y|y_obs)` similares a los datos observados `y_obs` bajo alguna función discrepante `T(y, θ)`, el modelo es compatible con los datos. P-value bayesiano: `P(T(y_rep) > T(y_obs) | y_obs)`.

**Fortalezas**: Marco bayesiano unificado, flexible en la elección de `T`. Bien establecido.

**Limitaciones**: Conservador (los datos se usan dos veces). Elección de `T` subjetiva.

### 8.2. Holdout Predictive Checks

**Ref**: Moran, Blei, Ranganath. *Holdout predictive checks for Bayesian model criticism*. J. R. Stat. Soc. B 86(1):194–214 (2024).

**Idea**: Arregla el "data used twice" del PPC clásico evaluando la posterior predictive sobre datos held-out → p-values calibrados.

**Fortalezas**: Misma flexibilidad que PPC con garantías frecuentistas.

### 8.3. Posterior Predictive Null (PPN)

**Ref**: Moran, Blei, Ranganath. *The posterior predictive null*. Bayesian Analysis 18(4) (2023). [arXiv:2112.03333](https://arxiv.org/abs/2112.03333).

**Idea**: PPC comparativo — comprueba si la predictiva de un modelo pasa los checks de otro modelo. Útil para decidir si dos modelos generativos son efectivamente equivalentes sobre las funciones de prueba elegidas.

### 8.4. Bayesian Kernel Two-Sample Testing

**Ref**: Zhang, Cremer, Wager. *Bayesian kernel two-sample testing*. J. Comput. Graph. Stat. (2022).

**Idea**: Análogo bayesiano del MMD: prior sobre la witness function en RKHS → posterior sobre "¿son iguales P y Q?". Da Bayes factor en lugar de p-value.

**Fortalezas**: Interpretación bayesiana directa (probabilidad de hipótesis), no sólo rechazo/no-rechazo.

### 8.5. Sequential Kernelized Stein Discrepancy

**Ref**: Martinez-Taboada & Ramdas. *Sequential kernelized Stein discrepancy*. [arXiv:2409.17505](https://arxiv.org/abs/2409.17505) (2024).

**Idea**: Test secuencial *anytime-valid* usando KSD. Permite seguir recolectando datos y parar adaptativamente sin inflar el error tipo I. Pareja con Sequential MMD (Shekhar & Ramdas).

**Aplicación a `echo`**: Monitorización continua train-vs-test en producción.

---

## 9. Librerías y recursos

| Librería | Métodos cubiertos | Notas |
|---|---|---|
| `scipy.stats` | KS, Anderson, Cramér-von Mises, χ², t-test | Sólido para 1D. |
| `torch-two-sample` | MMD, energy, FR, k-NN | PyTorch, mantenimiento variable. |
| `POT` (Python Optimal Transport) | Wasserstein, Sinkhorn, transport plans | Maduro y bien documentado. |
| `geomloss` | Sinkhorn / energy / MMD en GPU | Para tensors grandes. |
| `dcor` | Energy distance, distance correlation | Limpio y rápido. |
| `densratio` | KLIEP, uLSIF | Suficiente para POC. |
| `alibi-detect` | Drift detection (KS multivariante, MMD, classifier) | Pensada para producción ML. |
| `evidently` | Drift dashboards, varios tests | Más orientada a reporting. |
| `frouros` | Drift detection methods | Mantenida activamente. |
| `nflows`, `zuko` | Normalizing flows | Útil para CATHODE/CURTAINs/FETA. |

**Libros de referencia**:
- Gretton et al. *A Kernel Two-Sample Test*, JMLR 2012 (review).
- Sugiyama, Suzuki, Kanamori. *Density Ratio Estimation in Machine Learning*. Cambridge UP, 2012.
- Peyré & Cuturi. *Computational Optimal Transport*. Foundations and Trends in ML, 2019.
- Gelman et al. *Bayesian Data Analysis*, 3ª ed. CRC, 2013.

---

## 10. Roadmap orientativo para `echo`

Una posible jerarquía de funcionalidad a implementar:

**Nivel 1 — diagnóstico rápido** (baseline imprescindible)
- Tests univariantes marginal por marginal (KS, Anderson, χ²).
- Reporte por feature: estadístico, p-value, intensidad relativa.

**Nivel 2 — test global multivariante**
- Energy distance (sin hiperparámetros, buena baseline).
- MMD-RBF con bandwidth por mediana (más sensible).
- Wasserstein/Sinkhorn (geometría intuitiva).
- C2ST con un clasificador ligero (RandomForest / gradient boosting).

**Nivel 3 — localización**
- Score por punto del C2ST → ranking de eventos "más diferentes".
- Density ratio vía uLSIF o gradient boosting → mapas locales.
- Particionamiento (árbol del C2ST) → regiones interpretables.

**Nivel 4 — métodos avanzados** (opcionales)
- MMD-D / kernel aprendido para alta dimensión.
- NPLM-style: clasificador regularizado como GoF multivariante con score por evento.
- Sequential KSD/MMD para monitoreo continuo.

Sugerencia: empezar por niveles 1+2 (cubren el 80% del valor práctico) y construir una API uniforme antes de pasar a 3 y 4.

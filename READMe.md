# Алгоритм регуляризованного уравнивания геодезических наблюдений

## 1. Формирование системы уравнений поправок и нормальных уравнений

### 1.1. Построение матрицы коэффициентов и вектора измерений

**Вход:** список из $m$ измерений $\ell_1, \dots, \ell_m$ с весами $p_i > 0$; список из $n$ неизвестных $x_1, \dots, x_n$; начальное приближение $x_0$.

**Выход:** матрица коэффициентов $A \in \mathbb{R}^{m \times n}$, вектор измерений $y \in \mathbb{R}^m$, весовая матрица $P = \mathrm{diag}(p_1, \dots, p_m)$.

**Формальные шаги:**

1. Для каждого измерения $i = 1, \dots, m$:
   - вычислить частные производные $\partial f_i / \partial x_j$ в точке $x_0$ по каждому неизвестному $x_j$, $j = 1, \dots, n$;
   - заполнить строку: $A[i][j] = \partial f_i / \partial x_j$;
   - положить свободный член: $y[i] = \ell_i - f_i(x_0)$.

2. Весовая матрица: $P[i][i] = p_i$, $P[i][j] = 0$ при $i \neq j$.

3. Начальное значение вектора поправок: $x = 0$.

### 1.2. Формирование нормальных уравнений Гаусса

**Вход:** матрица $A$, вектор $y$, весовая матрица $P$.

**Выход:** матрица нормальных уравнений $N = A^T P A \in \mathbb{R}^{n \times n}$ и правая часть $b = A^T P y \in \mathbb{R}^n$.

**Формальные шаги:**

1. Взвешенная матрица: $\tilde{A}[i][j] = \sqrt{p_i} \cdot A[i][j]$.

2. Взвешенный вектор: $\tilde{y}[i] = \sqrt{p_i} \cdot y[i]$.

3. Вычисление нормальной матрицы:

$$
N[j][k] = \sum_{i=1}^{m} \tilde{A}[i][j] \cdot \tilde{A}[i][k], \quad j, k = 1, \dots, n.
$$

(OpenMP: параллельный цикл по $j$ с приватным накоплением.)

4. Вычисление правой части:

$$
b[j] = \sum_{i=1}^{m} \tilde{A}[i][j] \cdot \tilde{y}[i], \quad j = 1, \dots, n.
$$

5. Оценка числа обусловленности $\kappa(N) = \lambda_{\max} / \lambda_{\min}$ через диагональ или через SVD (см. п. 2).

**Замечание.** Прямое решение $\hat{x} = N^{-1} b$ неустойчиво при $\kappa(N) \gg 1$, поэтому переходим к регуляризации.

---

## 2. Сингулярное разложение матрицы $\tilde{A}$

### 2.1. Односторонний SVD (метод Якоби)

**Вход:** взвешенная матрица $\tilde{A} \in \mathbb{R}^{m \times n}$, $m \ge n$; порог сходимости $\varepsilon_{\text{SVD}} > 0$; максимальное число итераций $K_{\max}$.

**Выход:** матрицы $U \in \mathbb{R}^{m \times n}$, $V \in \mathbb{R}^{n \times n}$, вектор сингулярных чисел $\sigma \in \mathbb{R}^n$ (в порядке убывания).

**Формальные шаги:**

1. Инициализация: $U \leftarrow \tilde{A}$, $V \leftarrow I_n$, счётчик итераций $k = 0$.

2. Пока $k < K_{\max}$ и не выполнено условие сходимости:

   а) Для всех пар столбцов $(p, q)$, $p < q$, вычислить:

$$
\alpha = \sum_{i=1}^{m} U[i][p]^2, \quad \beta = \sum_{i=1}^{m} U[i][q]^2, \quad \gamma = \sum_{i=1}^{m} U[i][p] \, U[i][q].
$$

   б) Если $|\gamma| / \sqrt{\alpha \beta} > \varepsilon_{\text{SVD}}$, выполнить вращение Якоби:

$$
\zeta = (\beta - \alpha) / (2\gamma), \quad t = \mathrm{sign}(\zeta) / (|\zeta| + \sqrt{1 + \zeta^2}),
$$

$$
c = 1 / \sqrt{1 + t^2}, \quad s = c \, t.
$$

   в) Обновить столбцы $p, q$ матриц $U$ и $V$:

$$
U[:, p]' = c \cdot U[:, p] - s \cdot U[:, q],
$$

$$
U[:, q]' = s \cdot U[:, p] + c \cdot U[:, q],
$$

$$
V[:, p]' = c \cdot V[:, p] - s \cdot V[:, q],
$$

$$
V[:, q]' = s \cdot V[:, p] + c \cdot V[:, q].
$$

   г) $k \leftarrow k + 1$.

3. Сингулярные числа: $\sigma_j = \|U[:, j]\|_2$, $j = 1, \dots, n$.

4. Нормировка: $U[:, j] \leftarrow U[:, j] / \sigma_j$, $j = 1, \dots, n$.

5. Сортировка: переставить столбцы $U, V$ и элементы $\sigma$ так, чтобы $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_n$.

**Замечание.** Условие сходимости: $\max_{p < q} |\gamma| / \sqrt{\alpha \beta} < \varepsilon_{\text{SVD}}$. OpenMP: параллельный цикл по парам $(p, q)$.

---

## 3. Тихоновская регуляризация через SVD

### 3.1. Вычисление решения $x_\alpha$

**Вход:** матрицы $U, V$, вектор $\sigma$, вектор $\tilde{y}$, параметр $\alpha > 0$.

**Выход:** регуляризованное решение $x_\alpha \in \mathbb{R}^n$.

**Формальные шаги:**

1. Для каждого $j = 1, \dots, n$:
   - вычислить проекцию: $c_j = \sum_{i=1}^{m} U[i][j] \cdot \tilde{y}[i]$;
   - коэффициент фильтра: $f_j = \sigma_j / (\sigma_j^2 + \alpha)$;
   - накопить: $x_\alpha \leftarrow x_\alpha + f_j \cdot c_j \cdot V[:, j]$.

2. Вернуть $x_\alpha$.

**Замечание.** Решение

$$
x_\alpha = \sum_{j} \frac{\sigma_j}{\sigma_j^2 + \alpha} \langle \tilde{y}, u_j \rangle \, v_j
$$

соответствует разделу 3.3 «Tikhonov's regularization» в главе Freeden & Nashed (с. 229–232).

### 3.2. Вычисление невязки и нормы решения

**Вход:** $x_\alpha$, $\tilde{A}$, $\tilde{y}$.

**Выход:** невязка $\rho(\alpha) = \|\tilde{A} x_\alpha - \tilde{y}\|_2$, норма $\eta(\alpha) = \|x_\alpha\|_2$.

**Формальные шаги:**

1. Вычислить вектор невязки: $r = \tilde{A} x_\alpha - \tilde{y}$.

2. $\rho(\alpha) = \sqrt{\sum_{i=1}^{m} r[i]^2}$.

3. $\eta(\alpha) = \sqrt{\sum_{j=1}^{n} x_\alpha[j]^2}$.

**Альтернатива через SVD:**

$$
\rho(\alpha)^2 = \sum_{j=1}^{n} \left(\frac{\alpha}{\sigma_j^2 + \alpha}\right)^2 c_j^2 + \|\tilde{y}\|^2 - \sum_{j=1}^{n} c_j^2,
$$

$$
\eta(\alpha)^2 = \sum_{j=1}^{n} \left(\frac{\sigma_j}{\sigma_j^2 + \alpha}\right)^2 c_j^2.
$$

---

## 4. Выбор параметра регуляризации по принципу невязки

### 4.1. Принцип невязки (исправленная версия)

**Вход:** $\tilde{A}$, $\tilde{y}$, оценка уровня шума $\varepsilon > 0$; параметр $\tau > 1$; интервал поиска $[\alpha_{\min}, \alpha_{\max}]$, гарантирующий $d(\alpha_{\min}) < 0 < d(\alpha_{\max})$; точность $\delta_\alpha$.

**Выход:** $\alpha^*$.

**Формальные шаги:**

1. Определить $d(\alpha) = \rho(\alpha) - \tau \varepsilon$.

2. Проверить корректность интервала:
   - вычислить $d(\alpha_{\min})$ и $d(\alpha_{\max})$;
   - если $d(\alpha_{\min}) > 0$ или $d(\alpha_{\max}) < 0$ — расширить интервал (например, $\alpha_{\min} \leftarrow \alpha_{\min}/10$, $\alpha_{\max} \leftarrow \alpha_{\max} \cdot 10$) до выполнения условия.

3. Бинарный поиск:
   - $\alpha_L \leftarrow \alpha_{\min}$, $\alpha_R \leftarrow \alpha_{\max}$;
   - пока $(\alpha_R - \alpha_L) > \delta_\alpha$:
     - $\alpha_M \leftarrow (\alpha_L + \alpha_R) / 2$;
     - вычислить $x_{\alpha_M}$ (п. 3.1) и $d(\alpha_M)$;
     - **если $d(\alpha_M) > 0$** (невязка слишком большая, $\alpha$ велико): $\alpha_R \leftarrow \alpha_M$;
     - **иначе** (невязка слишком мала, $\alpha$ мало): $\alpha_L \leftarrow \alpha_M$.
   - $\alpha^* \leftarrow \alpha_M$.

4. Вернуть $\alpha^*$ и $x_{\alpha^*}$.

---

## 5. Альтернативный выбор параметра методом L-кривой

### 5.1. Построение L-кривой

**Вход:** $\tilde{A}$, $\tilde{y}$; сетка $\{\alpha_k\}_{k=1}^{N_\alpha}$ (лог-равномерная); функция $\rho(\alpha), \eta(\alpha)$.

**Выход:** точки $(\eta(\alpha_k), \rho(\alpha_k))$, $k = 1, \dots, N_\alpha$.

**Формальные шаги:**

1. Для каждого $k = 1, \dots, N_\alpha$:
   - вычислить $x_{\alpha_k}$ (п. 3.1);
   - вычислить $\rho_k = \rho(\alpha_k)$, $\eta_k = \eta(\alpha_k)$;
   - сохранить точку $(\eta_k, \rho_k)$.

2. Перейти к логарифмическим координатам:

$$
\hat{\eta}_k = \log \eta_k, \quad \hat{\rho}_k = \log \rho_k.
$$

### 5.2. Поиск угловой точки

**Вход:** логарифмические точки $(\hat{\eta}_k, \hat{\rho}_k)$, $k = 1, \dots, N_\alpha$.

**Выход:** индекс $k^*$ угловой точки.

**Формальные шаги:**

1. Для каждого $k = 2, \dots, N_\alpha - 1$:
   - вычислить конечные разности:

$$
\Delta \hat{\eta}_k = \hat{\eta}_{k+1} - \hat{\eta}_{k-1}, \quad \Delta \hat{\rho}_k = \hat{\rho}_{k+1} - \hat{\rho}_{k-1};
$$

   - вторая разность:

$$
\Delta^2 \hat{\eta}_k = \hat{\eta}_{k+1} - 2 \hat{\eta}_k + \hat{\eta}_{k-1}, \quad \Delta^2 \hat{\rho}_k = \hat{\rho}_{k+1} - 2 \hat{\rho}_k + \hat{\rho}_{k-1};
$$

   - кривизна:

$$
\kappa_k = \frac{\Delta \hat{\eta}_k \cdot \Delta^2 \hat{\rho}_k - \Delta \hat{\rho}_k \cdot \Delta^2 \hat{\eta}_k}{\left((\Delta \hat{\eta}_k)^2 + (\Delta \hat{\rho}_k)^2\right)^{3/2}}.
$$

2. $k^* = \arg\max_k |\kappa_k|$.

3. Вернуть $\alpha^* = \alpha_{k^*}$.

**Замечание.** Описание — раздел 4.8 «Numerical dilemmas and methodologies» в главе Freeden & Nashed (с. 251).

---

## 6. Оценка точности и визуализация

### 6.1. Сравнение с эталоном

**Вход:** $x_\alpha$ (решение), $x_{\text{true}}$ (эталон), $A$, $y$.

**Выход:** относительная ошибка $\epsilon_{\text{rel}}$, невязка $\rho$, норма решения $\eta$.

**Формальные шаги:**

1. $\epsilon_{\text{rel}} = \|x_\alpha - x_{\text{true}}\|_2 / \|x_{\text{true}}\|_2$.

2. $\rho = \|\tilde{A} x_\alpha - \tilde{y}\|_2$.

3. $\eta = \|x_\alpha\|_2$.

4. Сохранить результаты в CSV.

### 6.2. Визуализация

**Вход:** сетка $\{\alpha_k\}$, $\{\rho_k\}$, $\{\eta_k\}$, $\{\epsilon_{\text{rel},k}\}$.

**Выход:** графики.

**Формальные шаги:**

1. График L-кривой: $\hat{\rho}$ от $\hat{\eta}$.

2. График $\epsilon_{\text{rel}}$ от $\log \alpha$.

3. График $\rho(\alpha)$ от $\log \alpha$.

4. График $\eta(\alpha)$ от $\log \alpha$.

5. Сравнительный график $x_\alpha$ и $x_{\text{true}}$ (для $n \le 3$).

---

## 7. Полный алгоритм (сводка)

**Вход:** $A$, $y$, $P$, $\varepsilon$, $\tau$, сетка $\{\alpha_k\}$.

**Выход:** $x_{\alpha^*}$, $\alpha^*$, метрики.

**Формальные шаги:**

1. Сформировать $\tilde{A} = \sqrt{P} A$, $\tilde{y} = \sqrt{P} y$ (п. 1.2).

2. Вычислить SVD: $\tilde{A} = U \Sigma V^T$ (п. 2).

3. Для каждого $\alpha_k$ из сетки вычислить $x_{\alpha_k}$, $\rho_k$, $\eta_k$ (п. 3).

4. Выбрать $\alpha^*$:
   - через принцип невязки (п. 4), или
   - через L-кривую (п. 5).

5. Вычислить $x_{\alpha^*}$ (п. 3.1) и метрики (п. 6).

6. Вернуть результат.

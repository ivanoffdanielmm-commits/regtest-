# Регуляризованное уравнивание геодезических наблюдений

## Постановка

Дана сеть из $n$ пунктов с неизвестными поправками $x \in \mathbb{R}^n$. Выполнено $m$ измерений $\ell_i$ с весами $p_i > 0$. Требуется решить систему
$$A x \approx y,$$
где $A \in \mathbb{R}^{m \times n}$ — матрица коэффициентов, $y \in \mathbb{R}^m$ — вектор свободных членов.

---

## Шаги

**1. Формирование уравнений поправок.**  
Для каждого измерения $i = 1, \dots, m$:
* вычислить $A_{i,j} = \partial f_i / \partial x_j$ в точке $x_0$;
* задать $y_i = \ell_i - f_i(x_0)$;
* задать вес $p_i > 0$.

**2. Взвешивание.**  
$$\tilde{A} = P^{1/2} A, \qquad \tilde{y} = P^{1/2} y,$$
где $P = \mathrm{diag}(p_1, \dots, p_m)$.

**3. Сингулярное разложение.**  
Вычислить $\tilde{A} = U \Sigma V^T$, где $\sigma_1 \geq \dots \geq \sigma_n \geq 0$ — сингулярные числа, $u_j, v_j$ — столбцы $U, V$.

**4. Оценка уровня шума.**  
$$\sigma_0 = \frac{\|\tilde{A} x_{\text{LS}} - \tilde{y}\|_2}{\sqrt{m - n}}, \qquad \varepsilon = \sigma_0 \sqrt{m}.$$

**5. Выбор параметра $\alpha$ по принципу невязки.**  
Найти $\alpha^* > 0$ такое, что
$$\rho(\alpha^*) = \tau \varepsilon, \qquad \rho(\alpha) = \|\tilde{A} x_{\alpha} - \tilde{y}\|_2, \quad \tau > 1.$$
Функция $d(\alpha) = \rho(\alpha) - \tau \varepsilon$ монотонна; корень ищется бисекцией.

**6. Регуляризованное решение.**  
$$\hat{x} = x_{\alpha^{\ast}} = \sum_{j=1}^n \frac{\sigma_j}{\sigma_j^2 + \alpha^{\ast}} \, \langle \tilde{y}, u_j \rangle \, v_j.$$


**7. Уравненные значения.**  
$$h_i = x_{0,i} + \hat{x}_i, \qquad i = 1, \dots, n.$$

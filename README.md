# Scoring crediticio con Árboles y Random Forest: German Credit, bien hecho

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USUARIO/credit-scoring-trees-german-credit/blob/main/notebooks/german_credit_trees_rf.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)

Análisis de riesgo crediticio sobre el dataset **Statlog German Credit** (UCI). El objetivo va más allá de entrenar un clasificador: se usa un caso real para **demostrar empíricamente** cómo funcionan los árboles de decisión y el Random Forest, y para mostrar cómo cambian las conclusiones cuando se incorporan **costos asimétricos, sobremuestreo, sesgo de selección y equidad**.

> **La mayoría de los análisis públicos de este dataset usan una codificación incorrecta.** Grömping (2019) demostró que el codebook de UCI tiene los niveles mezclados en 8 atributos. Este proyecto usa la codificación corregida y muestra el impacto de la corrección.

---

## Hallazgos principales

| # | Hallazgo | Evidencia |
|---|---|---|
| 1 | **El codebook de UCI invierte la interpretación.** Con la etiqueta de UCI, "sin cuenta corriente" es el grupo más seguro. Con la corregida, es el más riesgoso. | 11.7 % vs. **49.3 %** de no pago |
| 2 | **El umbral 0.5 es la decisión equivocada.** Con la matriz de costos oficial (5:1), el umbral óptimo es p* = 5/6. Con 0.5, el Random Forest cuesta más que rechazar a todos. | costo 0.785 → **0.520** |
| 3 | **El sobremuestreo cambia la política óptima.** La muestra tiene 30 % de malos y la población real ~5 %. Corregir el prior por Bayes transforma la política de crédito. | aprobación 42 % → **94 %** |
| 4 | **La fórmula de varianza del Random Forest, medida.** Bajar `max_features` reduce la correlación entre árboles (ρ) y mejora el bosque aunque cada árbol empeore. | ρ 0.30 → 0.12 |
| 5 | **La importancia MDI es engañosa.** Una columna de ruido puro queda 6.ª de 19 en MDI y ≈ 0 en importancia por permutación. | ver figura |
| 6 | **Eliminar `edad` no elimina el sesgo.** Las demás variables predicen si alguien tiene ≤ 25 años. Los jóvenes que sí pagan son aprobados 3 veces menos. | AUC proxy **0.80**; TPR 0.16 vs 0.47 |

<p align="center">
  <img src="figures/cost_vs_threshold.png" width="48%">
  <img src="figures/rf_decorrelation.png" width="48%">
</p>

## Benchmark (CV estratificada 10×3, modelos calibrados, umbral 5/6)

| Modelo | ROC-AUC | Costo por solicitud ↓ |
|---|---|---|
| Random Forest | 0.794 ± 0.042 | **0.537** |
| Gradient Boosting | 0.794 ± 0.045 | 0.542 |
| Regresión logística | 0.783 ± 0.043 | 0.564 |
| Árbol podado | 0.728 ± 0.047 | 0.695 |
| *Referencia: rechazar a todos* | — | *0.700* |

Los ensambles y la logística son estadísticamente equivalentes. El árbol individual **no aporta valor económico** con esta matriz de costos.

## Contenido del notebook

1. **Datos:** decodificación corregida, comparación contra el codebook de UCI, EDA de tasas de no pago con intervalos de Wilson, asociación (V de Cramér / Mann-Whitney).
2. **Árbol de decisión:** impureza de Gini calculada a mano y verificada contra sklearn, invariancia ante transformaciones monótonas, **inestabilidad** medida con bootstrap, postpoda por *cost-complexity* con α elegido por CV.
3. **Random Forest:** verificación empírica de $\mathrm{Var}=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$, fracción OOB teórica vs. real, búsqueda de hiperparámetros.
4. **Decisión:** umbral óptimo derivado de costos, equivalencia entre `class_weight` y mover el umbral, **corrección por prior**, calibración (Brier, curvas de confiabilidad).
5. **Interpretabilidad:** MDI vs. permutación con variable de ruido de control, dependencia parcial + ICE, SHAP, explicación contrafactual de un rechazo.
6. **Equidad:** por qué el sexo **no** es medible en este dataset, impacto dispar y TPR por edad, test de variables proxy.

## Cómo ejecutarlo

**Opción 1 (Colab):** clic en el badge de arriba. El dataset se descarga automáticamente.

**Opción 2 (local):**
```bash
git clone https://github.com/USUARIO/credit-scoring-trees-german-credit.git
cd credit-scoring-trees-german-credit
pip install -r requirements.txt
jupyter notebook notebooks/german_credit_trees_rf.ipynb
```

## Estructura

```
├── notebooks/
│   └── german_credit_trees_rf.ipynb   # análisis completo, con outputs
├── figures/                           # figuras usadas en este README
├── requirements.txt
└── README.md
```

## Limitaciones

- Los datos son de **1973–1975**, de un solo banco alemán. Las relaciones no son extrapolables al crédito actual.
- **Sesgo de selección:** solo se observan préstamos aprobados. Algunos patrones (por ejemplo, que tener un inmueble se asocie a más mora) son compatibles con este sesgo y no deben leerse como causales.
- Con 1.000 observaciones, las diferencias pequeñas entre modelos no son estadísticamente significativas. Por eso se reportan desvíos e intervalos bootstrap.

## Datos y referencias

- **Dataset:** Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77 (licencia CC BY 4.0).
- **Codificación corregida:** Grömping, U. (2019). *South German Credit Data: Correcting a Widely Used Data Set*. Reports in Mathematics, Physics and Chemistry 4/2019, Beuth Hochschule für Technik Berlin.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
- Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984). *Classification and Regression Trees*.

## Autor

**Juan Andrés Fernandez**: estudiante de Ciencia de Datos (UNLP).
[LinkedIn](https://www.linkedin.com/in/fernándezjuan) · [GitHub](https://github.com/montesquieu891)

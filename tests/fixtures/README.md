`legacy_v012_demo.npz` contains four control-cell mass matrices and balance
vectors produced by unmodified `src/fuelfield/core.py` at commit `8631c6f`,
using `examples/demo_project_140.json`. Generated with Python 3.12/NumPy 2.5.4.
The test compares v0.2 outputs against these independent v0.1.2 reference
results with rtol=1e-12, atol=1e-15. It requires no Git checkout at test runtime.

To regenerate intentionally: extract that core with
`git show 8631c6f:src/fuelfield/core.py`, import it under a separate module name,
load the demo using its `project_from_json`, call `calculate_plane` for each z,
and store `mass_0`…`mass_3` plus `balance_0`…`balance_3` containing
[injected, inside, inner, outer] kg/s with `numpy.savez_compressed`.
Do not regenerate to make a changed physical model pass the compatibility test.
# Эталоны Дитякина

`dityakin_reference.json` создан независимым генератором
`dityakin_reference.py` по уравнениям монографии 1977 г. Генератор не импортирует
приложение: используется RK4 для уравнений момента/работы и Decimal для поиска
максимума расхода. Это численная проверка, не измерения реального распылителя.
Источник, геометрия, все коэффициенты и ограничения описаны в
`docs/DITYAKIN_METHOD.md`. Перегенерация: `python tests/fixtures/dityakin_reference.py`.

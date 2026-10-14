# mai-numerical-methods
Лабораторные работы по курсу "Численные методы" студента Ивченко Матвей Сергеевич группы М8О-307Б-23

## Структура проекта

```
numerical-methods/
│
├── pyproject.toml
├── README.md
│
├── src/
│   └── nm/
│       ├── __init__.py
│       │
│       ├── linalg/            # Линейная алгебра (Лаб 1)
│       │   ├── __init__.py
│       │   ├── lu.py
│       │   ├── tridiagonal.py
│       │   ├── iterative.py
│       │   ├── qr.py
│       │   └── eigen.py
│       │
│       ├── nonlinear/         # Решение нелинейных уравнений (Лаб 2)
│       │   ├── __init__.py
│       │   ├── solvers.py     # Метод Ньютона и простой итерации для f(x) = 0
│       │   └── systems.py     # Те же методы для систем F(X) = 0
│       │
│       ├── approximation/     # Приближение функций (Лаб 3)
│       │   ├── __init__.py
│       │   ├── interpolation.py
│       │   └── splines.py
│       │
│       ├── ode/               # Задачи Коши для ОДУ (Лаб 4)
│       │   ├── __init__.py
│       │   └── ivp.py
│       │
│       ├── parabolic.py       # Параболические уравнения, три схемы и граничные условия (Лаб 5)
│       │
│       └── utils.py           # Общие утилиты (графики, замеры погрешности)
│
├── labs/
│   ├── lab1.ipynb
│   ├── lab2.ipynb
│   ├── lab3.ipynb
│   ├── lab4.ipynb
│   └── lab5.ipynb             # Вариант 8: решение, выводы формул, графики и погрешности
│
└── tests/
    ├── test_linalg.py
    ├── test_nonlinear.py
    ├── test_approximation.py
    └── test_parabolic.py      # Сходимость схем и порядки аппроксимации границ (Лаб 5)
```

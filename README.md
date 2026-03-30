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
│       └── utils.py           # Общие утилиты (графики, замеры погрешности)
│
├── labs/
│   ├── lab1/
│   │   └── main.ipynb
│   ├── lab2/
│   │   ├── lab2_1.ipynb       # Исследование уравнения
│   │   └── lab2_2.ipynb       # Исследование системы
│   └── lab3/
│
└── tests/
    ├── test_linalg.py
    └── test_nonlinear.py
```


import random
import math
from copy import deepcopy


class Matrix:
    def __init__(self, values: list[list[float]], shape: tuple[int, int] = None):
        self.values = deepcopy(values)
        if shape is None:
            if not isinstance(values, (list, tuple)):
                raise TypeError('Значения должны быть списком списков')
            if not all(isinstance(i, (list, tuple)) for i in values):
                raise TypeError('Значения матрицы должны быть упакованы в список списков')
            if len(values) == 0:
                raise ValueError('Список значений не может быть пустым')

            shape = (len(values), len(values[0]))
            if not all(len(row) == shape[1] for row in values):
                raise ValueError('Вложенные списки должны быть одного размера')
        self.shape = shape

    def __repr__(self):
        rows = []
        for r in self.values:
            # Целая часть добивается до 11 символов, дробная - до 4 символов
            # Все числа объединяются в формате [x1, x2, ...]
            rows.append("[" + ", ".join(f"{x:11.4g}" for x in r) + "]")

        # Разделитель строк - перенос строки и небольшой отступ
        delimiter = ',\n  '
        return f"Matrix(\n  {delimiter.join(rows)}\n)"
    
    def copy(self):
        return Matrix(self.values, self.shape)

    @classmethod
    def zeros(cls, shape: tuple[int, int]):
        values = [[0.0] * shape[1] for _ in range(shape[0])]
        return cls(values, shape)
    
    @classmethod
    def ones(cls, shape: tuple[int, int]):
        values = [[1.0] * shape[1] for _ in range(shape[0])]
        return cls(values, shape)
    
    @classmethod
    def normal(cls, shape: tuple[int, int], mean=0.0, std=1.0):
        values = [[random.gauss(mu=mean, sigma=std) for _ in range(shape[1])] for _ in range(shape[0])]
        return cls(values, shape)
    
    @classmethod
    def uniform(cls, shape: tuple[int, int], a=0.0, b=1.0):
        values = [[random.uniform(a, b) for _ in range(shape[1])] for _ in range(shape[0])]
        return cls(values, shape)
    
    @classmethod
    def eye(cls, size: int):
        values = [[0.0] * size for _ in range(size)]
        for i in range(size):
            values[i][i] = 1.0
        return cls(values, (size, size))

    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ ИНДЕКСАЦИИ

    def _normalize_key(self, key):
        """
        Приводит ключ к виду (row_spec, col_spec).
        Поддерживает: int, slice, tuple, list, Ellipsis (...)
        """
        # Когда к квадратных скобках только один элемент (число или срез),
        # то нужно для унификации преобразовать в кортеж
        if not isinstance(key, tuple):
            key = (key,)
        
        # Обработка троеточия. Оно эквивалентно некоторому количеству полных срезов
        if Ellipsis in key:
            idx = key.index(Ellipsis)
            missing = 2 - len(key) + 1
            key = key[:idx] + (slice(None),) * missing + key[idx+1:]

        # Если ключ короче 2 измерений, дополняем slice(None) (полный срез)
        if len(key) == 1:
            key = (key[0], slice(None))
        # такой случай редко встречается на практике (чтобы это условие сработало
        # необходимо в квадратные скобки передать пустой кортеж), но для полноты
        # картины пусть тоже будет
        elif len(key) == 0:
            key = (slice(None), slice(None))
            
        if len(key) != 2:
            raise IndexError(f"Invalid index for 2D Matrix: {key}")
            
        return key

    def _get_indices(self, spec, limit):
        """
        Превращает int, slice, list или tuple в список индексов.
        """
        # 1. Одиночный индекс
        if isinstance(spec, int):
            # Поддержка отрицательных индексов
            if spec < 0:
                spec += limit
            if not (0 <= spec < limit):
                raise IndexError(f"Index {spec} out of bounds for dimension {limit}")
            return [spec]

        # 2. Срез
        elif isinstance(spec, slice):
            return list(range(*spec.indices(limit)))
        
        # 3. Список или кортеж индексов (Fancy Indexing)
        elif isinstance(spec, (list, tuple)):
            indices = []
            for s in spec:
                if not isinstance(s, int):
                    raise TypeError(f"Index list must contain integers, got {type(s)}")
                if s < 0:
                    s += limit
                if not (0 <= s < limit):
                    raise IndexError(f"Index {s} out of bounds for dimension {limit}")
                indices.append(s)
            return indices
        
        else:
            raise TypeError(f"Invalid index type: {type(spec)}")

    def __getitem__(self, key) -> 'Matrix':
        row_spec, col_spec = self._normalize_key(key)
        
        row_indices = self._get_indices(row_spec, self.shape[0])
        col_indices = self._get_indices(col_spec, self.shape[1])
        
        # Если выбран ровно один элемент, возвращаем скаляр
        if isinstance(row_spec, int) and isinstance(col_spec, int):
            return self.values[row_indices[0]][col_indices[0]]
        
        # Иначе возвращаем новую матрицу (срез)
        new_values = []
        for r in row_indices:
            new_row = []
            for c in col_indices:
                new_row.append(self.values[r][c])
            new_values.append(new_row)
            
        new_shape = (len(row_indices), len(col_indices))
        return Matrix(new_values, new_shape)

    def __setitem__(self, key, value):
        row_spec, col_spec = self._normalize_key(key)
        
        row_indices = self._get_indices(row_spec, self.shape[0])
        col_indices = self._get_indices(col_spec, self.shape[1])
        
        target_shape = (len(row_indices), len(col_indices))
        
        # 1. Если присваиваем скаляр, то заполняем всю выделенную область
        if isinstance(value, (int, float)):
            for r in row_indices:
                for c in col_indices:
                    self.values[r][c] = float(value)
            return

        # 2. Если присваиваем Matrix
        if isinstance(value, Matrix):
            if value.shape != target_shape:
                raise ValueError(f"Shape mismatch: cannot assign Matrix {value.shape} to slice {target_shape}")
            for i, r in enumerate(row_indices):
                for j, c in enumerate(col_indices):
                    self.values[r][c] = value.values[i][j]
            return

        # 3. Если присваиваем список списков
        if isinstance(value, list):
            # Попытка определить форму
            try:
                h = len(value)
                w = len(value[0]) if h > 0 else 0
                # Проверка прямоугольности
                if any(len(row) != w for row in value):
                    raise ValueError("Rows in value list have different lengths")
            except (TypeError, IndexError):
                raise ValueError("Invalid list structure for assignment")
            
            if (h, w) != target_shape:
                raise ValueError(f"Shape mismatch: cannot assign list shape {(h, w)} to slice {target_shape}")
            
            for i, r in enumerate(row_indices):
                for j, c in enumerate(col_indices):
                    self.values[r][c] = float(value[i][j])
            return

        raise TypeError(f"Cannot assign type {type(value)} to Matrix")

    # АРИФМЕТИЧЕСКИЕ ОПЕРАЦИИ

    def _check_shape(self, other: 'Matrix', op_name: str):
        if not isinstance(other, Matrix):
            raise TypeError(f"{op_name} requires Matrix, got {type(other)}")
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch for {op_name}: {self.shape} vs {other.shape}")

    def __add__(self, other):
        if isinstance(other, (int, float)):
            new_values = [[x + other for x in row] for row in self.values]
            return Matrix(new_values, self.shape)
        
        self._check_shape(other, "addition")
        new_values = [
            [self.values[i][j] + other.values[i][j] for j in range(self.shape[1])]
            for i in range(self.shape[0])
        ]
        return Matrix(new_values, self.shape)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self.values[i][j] += other
            return self
        
        self._check_shape(other, "in-place addition")
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.values[i][j] += other.values[i][j]
        return self

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            new_values = [[x - other for x in row] for row in self.values]
            return Matrix(new_values, self.shape)
        
        self._check_shape(other, "subtraction")
        new_values = [
            [self.values[i][j] - other.values[i][j] for j in range(self.shape[1])]
            for i in range(self.shape[0])
        ]
        return Matrix(new_values, self.shape)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            new_values = [[other - x for x in row] for row in self.values]
            return Matrix(new_values, self.shape)
        return NotImplemented

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self.values[i][j] -= other
            return self
        
        self._check_shape(other, "in-place subtraction")
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.values[i][j] -= other.values[i][j]
        return self

    def __mul__(self, other):
        """Поэлементное умножение (Hadamard product)"""
        if isinstance(other, (int, float)):
            new_values = [[x * other for x in row] for row in self.values]
            return Matrix(new_values, self.shape)
        
        self._check_shape(other, "element-wise multiplication")
        new_values = [
            [self.values[i][j] * other.values[i][j] for j in range(self.shape[1])]
            for i in range(self.shape[0])
        ]
        return Matrix(new_values, self.shape)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Деление матрицы на число или поэлементное деление на другую матрицу.
        Пример: matrix / 2.0  или  matrix_a / matrix_b
        """
        # Деление на скаляр (число)
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            new_values = [[x / other for x in row] for row in self.values]
            return Matrix(new_values, self.shape)
        
        self._check_shape(other, "division")
        new_values = [
            [self.values[i][j] / other.values[i][j] for j in range(self.shape[1])]
            for i in range(self.shape[0])
        ]
        return Matrix(new_values, self.shape)

    def __rtruediv__(self, other):
        """
        Деление числа на матрицу (поэлементное).
        Пример: 2.0 / matrix
        """
        if isinstance(other, (int, float)):
            new_values = [[other / x for x in row] for row in self.values]
            return Matrix(new_values, self.shape)
        return NotImplemented

    def __matmul__(self, other):
        """Матричное умножение (A @ B)"""
        if not isinstance(other, Matrix):
            return NotImplemented
        
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Matrix dimensions mismatch for multiplication: {self.shape} and {other.shape}")
        
        # Результат shape: (rows_A, cols_B)
        new_shape = (self.shape[0], other.shape[1])
        new_values = [[0.0] * new_shape[1] for _ in range(new_shape[0])]
        
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                sum_val = 0.0
                for k in range(self.shape[1]):
                    sum_val += self.values[i][k] * other.values[k][j]
                new_values[i][j] = sum_val
                
        return Matrix(new_values, new_shape)

    def dot(self, other: 'Matrix') -> 'Matrix':
        """Явный метод для матричного умножения"""
        return self.__matmul__(other)

    def __neg__(self):
        return self * -1.0

    # УТИЛИТЫ И СРАВНЕНИЕ

    @property
    def T(self) -> 'Matrix':
        """Транспонирование матрицы"""
        new_values = [[self.values[r][c] for r in range(self.shape[0])] for c in range(self.shape[1])]
        return Matrix(new_values, (self.shape[1], self.shape[0]))

    def sum(self) -> float:
        """Сумма всех элементов"""
        return sum(sum(row) for row in self.values)

    def mean(self) -> float:
        """Среднее значение всех элементов"""
        return self.sum() / (self.shape[0] * self.shape[1])

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        if self.shape != other.shape:
            return False
        # Сравнение с допуском для float
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if not math.isclose(self.values[i][j], other.values[i][j], rel_tol=1e-9):
                    return False
        return True

    def __len__(self):
        # Возвращаем количество строк, как принято в Python для 2D структур
        return self.shape[0]


def transpose_permutation(p):
    p_new = [0] * len(p)
    for i, k in enumerate(p):
        p_new[k] = i
    return p_new

def count_permutation(p):
    p = p[:]
    result = 0
    for i in range(len(p)):
        if p[i] == -1: continue

        cnt = 0
        while p[i] != -1:
            i_new = p[i]
            p[i] = -1

            cnt += 1
            i = i_new

        result += cnt - 1

    return result

# ==========================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ==========================================

if __name__ == " __main__":
    print("=== 1. БАЗОВЫЕ ОПЕРАЦИИ ===")
    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]], (2, 2))
    m2 = Matrix([[5.0, 6.0], [7.0, 8.0]], (2, 2))
    
    print("M1:\n", m1)
    print("M2:\n", m2)
    
    print("\nСложение (M1 + M2):\n", m1 + m2)
    print("Вычитание (M2 - M1):\n", m2 - m1)
    print("Поэлементное умножение (M1 * M2):\n", m1 * m2)
    
    print("\n=== 2. МАТРИЧНОЕ УМНОЖЕНИЕ (DOT) ===")
    m3 = m1 @ m2  # Оператор @
    print("M1 @ M2:\n", m3)
    
    m4 = m1.dot(m2) # Метод dot
    print("M1.dot(M2) (должно совпадать):\n", m4)
    
    print("\n=== 3. ТРАНСПОНИРОВАНИЕ ===")
    print("M1.T:\n", m1.T)
    
    print("\n=== 4. ПРОДВИНУТАЯ ИНДЕКСАЦИЯ (СПИСКИ) ===")
    m_big = Matrix.normal((5, 5), mean=0, std=10)
    print("Исходная матрица 5x5:\n", m_big)
    
    # Выбор конкретных строк (0-й и 4-й)
    print("\nВыбор строк 0 и 4 (m_big[[0, 4], :]):")
    sub_rows = m_big[[0, 4], :]
    print(sub_rows)
    
    # Выбор конкретных столбцов
    print("\nВыбор столбцов 1 и 3 у всех строк (m_big[:, [1, 3]]):")
    sub_cols = m_big[:, [1, 3]]
    print(sub_cols)
    
    # Выбор пересечения (строки 0, 2 и столбцы 1, 4)
    print("\nСложный срез (m_big[[0, 2], [1, 4]]):")
    sub_mix = m_big[[0, 2], [1, 4]]
    print(sub_mix)
    
    print("\n=== 5. ПРИСВАИВАНИЕ ЧЕРЕЗ СПИСКИ ===")
    m_test = Matrix.zeros((4, 4))
    print("Нулевая матрица:\n", m_test)
    
    # Записываем 100 в строки 1 и 3, во все столбцы
    m_test[[1, 3], :] = 100.0
    print("\nПосле m_test[[1, 3], :] = 100.0:\n", m_test)
    
    print("\n=== 6. СРАВНЕНИЕ И СТАТИСТИКА ===")
    print(f"M1 == M1.copy(): {m1 == m1.copy()}")
    print(f"M1 == M2: {m1 == m2}")
    print(f"Сумма элементов M1: {m1.sum()}")
    print(f"Среднее M1: {m1.mean():.4f}")
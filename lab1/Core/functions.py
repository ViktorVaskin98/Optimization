from typing import List
from .constructive_numbers import ConstructiveNumber, CNConstant, CNVariable

def get_rosenbrock(
    variables: List[CNVariable]
) -> ConstructiveNumber:
    """
    Генерирует Дерево для N-мерной функции Розенброка
    f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2) для i от 0 до N-2
    """

    n = len(variables)
    if n < 2:
        raise ValueError("Для функции Розенброка требуется как минимум 2 переменные")

    c_100 = CNConstant(100)
    c_1 = CNConstant(1)
    c_2 = CNConstant(2)

    result_tree = None

    for i in range(n - 1):
        x_i = variables[i]
        x_next = variables[i + 1]

        left_term = c_100 * ((x_next - (x_i ** c_2)) ** c_2)
        right_term = (c_1 - x_i) ** c_2
        current_sum = left_term + right_term

        if result_tree is None:
            result_tree = current_sum
        else:
            result_tree = result_tree + current_sum

    return result_tree


def get_dixon_price_nd(
    variables: List[CNVariable]
) -> ConstructiveNumber:
    """
    Дерево для N-мерной функции Диксона-Прайса
    f(x) = (x_0 - 1)^2 + sum( (i+1) * (2 * x_i^2 - x_{i-1})^2 ) для i от 1 до N-1
    """
    n = len(variables)
    if n < 2:
        raise ValueError("Для функции Диксона-Прайса требуется как минимум 2 переменные")

    c_1 = CNConstant(1)
    c_2 = CNConstant(2)

    result_tree = (variables[0] - c_1) ** c_2


    for i in range(1, n):
        x_i = variables[i]
        x_prev = variables[i - 1]

        i_coeff = CNConstant(i + 1)

        inner_term = (c_2 * (x_i ** c_2)) - x_prev
        current_sum = i_coeff * (inner_term ** c_2)
        result_tree = result_tree + current_sum
        
    return result_tree


def create_variables(
    n: int,
    initial_val: float = 0.0
) -> List[CNVariable]:
    """
    Вспомогательная функция для быстрой генерации списка переменных
    """
    if n < 1:
        raise ValueError("Размерность пространства должна быть >= 1")
    
    return [CNVariable(name=f"x{i}", initial_val=initial_val) for i in range(n)]
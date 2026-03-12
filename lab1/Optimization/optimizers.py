import decimal
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

from Core.constructive_numbers import ConstructiveNumber, CNVariable
from Core.math_tools import derivative, simplify

class BaseOptimizer(ABC):
    def __init__(
        self,
        learning_rate: float = 0.001,
        max_iter: int = 2000, 
        tolerance: float = 1e-5,
        precision_digits: int = 25
    ) -> None:
        self.lr = decimal.Decimal(str(learning_rate))
        self.max_iter = max_iter
        self.tolerance = decimal.Decimal(str(tolerance))
        self.precision_digits = precision_digits

    def optimize(
        self, 
        target_func: ConstructiveNumber, 
        variables: List[CNVariable],
        verbose: bool = False
    ) -> Dict:
        self._initialize(variables)

        ast_data = self._build_gradients(target_func, variables)
        
        point_history = [[float(var.val) for var in variables]]
        loss_history = []

        for t in range(1, self.max_iter + 1):
            current_loss = float(target_func.evaluate(self.precision_digits).middle)
            loss_history.append(current_loss)

            current_grads, max_grad_norm = self._compute_gradients(ast_data, variables, t)
            
            if verbose and t % 100 == 0:
                print(f"Итерация {t}/{self.max_iter} | Значение функции: {current_loss:.6f} | Макс. градиент: {float(max_grad_norm):.6f}")

            if max_grad_norm < self.tolerance:
                if verbose:
                    print(f"Сходится на итерации {t}!")
                break
            
            steps = self._calculate_steps(current_grads, t)
            new_point = self._take_step(variables, steps)
            point_history.append(new_point)

        final_loss = float(target_func.evaluate(self.precision_digits).middle)
        loss_history.append(final_loss)

        return {
            "final_point": [var.val for var in variables],
            "point_history": point_history,
            "loss_history": loss_history
        }

    @abstractmethod
    def _initialize(
        self,
        variables: List[CNVariable]
    ) -> None:
        """Метод для инициализации специфичных для алгоритма переменных"""
        pass
    
    @abstractmethod
    def _calculate_steps(
        self,
        gradients: List[decimal.Decimal],
        t: int) -> List[decimal.Decimal]:
        """Основной метод, где реализуется логика вычисления шага каждого алгоритма"""
        pass

    def _build_gradients(
        self,
        target_func: ConstructiveNumber,
        variables: List[CNVariable]
    ) -> List[ConstructiveNumber]:
        return [simplify(derivative(target_func, var)) for var in variables]

    def _compute_gradients(
        self,
        ast_data,
        variables: List[CNVariable],
        t: int
    ) -> Tuple[List[decimal.Decimal], decimal.Decimal]:
        current_grads = [grad.evaluate(self.precision_digits).middle for grad in ast_data]
        max_grad_norm = max(abs(g) for g in current_grads) if current_grads else decimal.Decimal(0)
        return current_grads, max_grad_norm

    def _take_step(
        self,
        variables: List[CNVariable],
        steps: List[decimal.Decimal]
    ) -> List[float]:
        new_point = []
        for i, var in enumerate(variables):
            new_val = var.val - steps[i]
            var.set_val(str(new_val))
            new_point.append(float(new_val))
        return new_point


class KieferWolfowitzOptimizer(BaseOptimizer):
    def __init__(
        self,
        c: float = 0.01, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.c = decimal.Decimal(str(c))

    def _initialize(
        self, 
        variables: List[CNVariable]
    ) -> None:
        pass

    def _build_gradients(
        self,
        target_func: ConstructiveNumber,
        variables: List[CNVariable]
    ) -> ConstructiveNumber:
        return target_func

    def _compute_gradients(
        self, 
        target_func: ConstructiveNumber, 
        variables: List[CNVariable], 
        t: int
    ) -> Tuple[List[decimal.Decimal], decimal.Decimal]:
        n_vars = len(variables)
        estimated_grads = []
        
        c_t = self.c / (decimal.Decimal(t) ** decimal.Decimal('0.25'))

        for i in range(n_vars):
            original_val = variables[i].val

            variables[i].set_val(original_val + c_t)
            f_plus = target_func.evaluate(self.precision_digits).middle

            variables[i].set_val(original_val - c_t)
            f_minus = target_func.evaluate(self.precision_digits).middle

            variables[i].set_val(original_val)

            grad_est = (f_plus - f_minus) / (decimal.Decimal(2) * c_t)
            estimated_grads.append(grad_est)

        max_grad_norm = max(abs(g) for g in estimated_grads) if estimated_grads else decimal.Decimal(0)
        return estimated_grads, max_grad_norm

    def _calculate_steps(
        self, 
        gradients: List[decimal.Decimal], 
        t: int
    ) -> List[decimal.Decimal]:
        a_t = self.lr / decimal.Decimal(t)
        return [a_t * g for g in gradients]


class GradientDescent(BaseOptimizer):
    def _initialize(
        self, 
        variables: List[CNVariable]
    ) -> None:
        pass
    
    def _calculate_steps(
        self, gradients: List[decimal.Decimal], t: int
    ) -> List[decimal.Decimal]:
        return [self.lr * g for g in gradients]


class MomentumGradientDescent(BaseOptimizer):
    def __init__(
        self, 
        momentum: float = 0.9, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.beta = decimal.Decimal(str(momentum))
        self.velocities = []

    def _initialize(
        self, 
        variables: List[CNVariable]
    ) -> None:
        self.velocities = [decimal.Decimal(0) for _ in variables]
    
    def _calculate_steps(
        self, 
        gradients: List[decimal.Decimal], 
        t: int
    ) -> List[decimal.Decimal]:
        steps = []
        for i, g in enumerate(gradients):
            self.velocities[i] = self.beta * self.velocities[i] + (decimal.Decimal(1) - self.beta) * g
            steps.append(self.lr * self.velocities[i])
        return steps


class AdamOptimizer(BaseOptimizer):
    def __init__(
        self, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        epsilon: float = 1e-8, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.beta1 = decimal.Decimal(str(beta1))
        self.beta2 = decimal.Decimal(str(beta2))
        self.epsilon = decimal.Decimal(str(epsilon))
        self.m = []
        self.v = []

    def _initialize(
        self, 
        variables: List[CNVariable]
    ) -> None:
        self.m = [decimal.Decimal(0) for _ in variables]
        self.v = [decimal.Decimal(0) for _ in variables]

    def _calculate_steps(
        self, 
        gradients: List[decimal.Decimal], 
        t: int
    ) -> List[decimal.Decimal]:
        steps = []
        bias_correction1 = decimal.Decimal(1) - (self.beta1 ** decimal.Decimal(t))
        bias_correction2 = decimal.Decimal(1) - (self.beta2 ** decimal.Decimal(t))
        
        for i, g in enumerate(gradients):
            self.m[i] = self.beta1 * self.m[i] + (decimal.Decimal(1) - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (decimal.Decimal(1) - self.beta2) * (g ** 2)
            
            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2
            
            denominator = v_hat.sqrt() + self.epsilon
            step_size = self.lr * m_hat / denominator
            steps.append(step_size)
            
        return steps
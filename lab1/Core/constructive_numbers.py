import decimal
from abc import ABC, abstractmethod
import warnings

from .interval import Interval

class ConstructiveNumber(ABC):
    """Базовый класс графа ленивых вычислений"""

    def __init__(self) -> None:
        self._cached_prec: int = -1
        self._cached_interval: Interval | None = None

    def evaluate(self, precision_digits: int) -> Interval:
        """Метод, возвращающий интервал с точностью `precision_digits`"""
        if self._cached_prec >= precision_digits and self._cached_interval:
            return self._cached_interval
        
        result = self._do_evaluate(precision_digits)

        self._cached_prec = precision_digits
        self._cached_interval = result
        return result

    @abstractmethod
    def _do_evaluate(self, precision_digits: int) -> Interval:
        """Метод, высчитывающий интервалы с точностью `precision_digits`"""
        pass

    def _compare(
        self,
        other: 'ConstructiveNumber',
        op_type: str
    ) -> bool:
        """Вспомогательный метод для сравнений двух `ConstructiveNumber`"""
        prec = 10
        max_prec = 2000

        while prec <= max_prec:
            i1 = self.evaluate(prec)
            i2 = other.evaluate(prec)

            if i1.high < i2.low:
                return op_type == '<'
            if i1.low > i2.high:
                return op_type == '>'
            
            prec *= 2

        if op_type == '==':
            return True
        
        warnings.warn(f'Числа неразличимы с точностью {max_prec}')
        return False
    
    def __lt__(self, other: 'ConstructiveNumber') -> bool:
        return self._compare(other, '<')
    
    def __gt__(self, other: 'ConstructiveNumber') -> bool:
        return self._compare(other, '>')
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConstructiveNumber):
            return NotImplemented
        return self._compare(other, '==')
    
    def _ensure_cn(
        self,
        val: 'ConstructiveNumber | int | float'
    ) -> 'ConstructiveNumber':
        """Метод, который проверяет, либо заменяет на `ConstructiveNumber`"""
        if isinstance(val, ConstructiveNumber):
            return val
        return CNConstant(val)
    
    def __add__(
        self,
        other: 'ConstructiveNumber | int | float'
    ) -> 'CNAdd':
        return CNAdd(self, self._ensure_cn(other))
    
    def __radd__(
        self,
        other: 'ConstructiveNumber | int | float'
    ) -> 'CNAdd':
        return CNAdd(self._ensure_cn(other), self)
    
    def __sub__(
        self,
        other: 'ConstructiveNumber | int | float'
    ) -> 'CNSub':
        return CNSub(self, self._ensure_cn(other))
    
    def __rsub__(
        self,
        other: 'ConstructiveNumber | int | float'
    ) -> 'CNSub':
        return CNSub(self._ensure_cn(other), self)
    
    def __mul__(
        self,
        other: 'ConstructiveNumber | int | float'
    ) -> 'CNMul':
        return CNMul(self, self._ensure_cn(other))
    
    def __rmul__(
        self,
        other: 'ConstructiveNumber | int | float'
    ) -> 'CNMul':
        return CNMul(self._ensure_cn(other), self)
    
    def __truediv__(
        self,
        other: 'ConstructiveNumber | int | float'
    ) -> 'CNDiv':
        return CNDiv(self, self._ensure_cn(other))
    
    def __rtruediv__(
        self,
        other: 'ConstructiveNumber | int | float'
    ) -> 'CNDiv':
        return CNDiv(self._ensure_cn(other), self)
    
    def __pow__(
        self,
        power: 'ConstructiveNumber | int | float'
    ) -> 'CNPow':
        return CNPow(self, self._ensure_cn(power))
    

class CNConstant(ConstructiveNumber):
    """Класс константы"""

    def __init__(
        self,
        val: str | int | float
    ) -> None:
        super().__init__()
        self.val = decimal.Decimal(str(val))

    
    def _do_evaluate(self, precision_digits: int) -> Interval:
        return Interval(self.val, self.val)


class CNAdd(ConstructiveNumber):
    """Класс сложения"""

    def __init__(
        self,
        left: ConstructiveNumber,
        right: ConstructiveNumber
    ) -> None:
        super().__init__()
        self.left = left
        self.right = right

    def _do_evaluate(self, precision_digits: int) -> Interval:
        extra_prec = precision_digits + 5
        decimal.getcontext().prec = extra_prec
        i1 = self.left.evaluate(extra_prec)
        i2 = self.right.evaluate(extra_prec)
        return Interval(i1.low + i2.low, i1.high + i2.high)


class CNSub(ConstructiveNumber):
    """Класс вычитания"""
    
    def __init__(
        self,
        left: ConstructiveNumber,
        right: ConstructiveNumber
    ) -> None:
        super().__init__()
        self.left = left
        self.right = right

    def _do_evaluate(self, precision_digits: int) -> Interval:
        extra_prec = precision_digits + 5
        decimal.getcontext().prec = extra_prec
        i1 = self.left.evaluate(extra_prec)
        i2 = self.right.evaluate(extra_prec)
        return Interval(i1.low - i2.high, i1.high - i2.low)


class CNMul(ConstructiveNumber):
    """Класс умножения"""
    
    def __init__(
        self,
        left: ConstructiveNumber,
        right: ConstructiveNumber
    ) -> None:
        super().__init__()
        self.left = left
        self.right = right

    def _do_evaluate(self, precision_digits: int) -> Interval:
        extra_prec = precision_digits + 5
        decimal.getcontext().prec = extra_prec
        i1 = self.left.evaluate(extra_prec)
        i2 = self.right.evaluate(extra_prec)

        coords = [
            i1.low * i2.low,
            i1.low * i2.high,
            i1.high * i2.low,
            i1.high * i2.high
        ]
        return Interval(min(coords), max(coords))


class CNDiv(ConstructiveNumber):
    """Класс деления"""
    
    def __init__(
        self,
        left: ConstructiveNumber,
        right: ConstructiveNumber
    ) -> None:
        super().__init__()
        self.left = left
        self.right = right

    def _do_evaluate(self, precision_digits: int) -> Interval:
        extra_prec = precision_digits + 5
        decimal.getcontext().prec = extra_prec
        i1 = self.left.evaluate(extra_prec)
        i2 = self.right.evaluate(extra_prec)

        if i2.low <= 0 <= i2.high:
            raise ValueError('Деление на интервал, содержащий ноль')
        
        coords = [
            i1.low / i2.low,
            i1.low / i2.high,
            i1.high / i2.low,
            i1.high / i2.high
        ]
        return Interval(min(coords), max(coords))


class CNPow(ConstructiveNumber):
    """Класс возведения в степень"""
    
    def __init__(
        self,
        base: ConstructiveNumber,
        power: ConstructiveNumber
    ) -> None:
        super().__init__()
        self.base = base
        self.power = power

    def _do_evaluate(self, precision_digits: int) -> Interval:
        extra_prec = precision_digits + 5
        decimal.getcontext().prec = extra_prec
        
        if isinstance(self.power, CNConstant) and self.power.val == self.power.val.to_integral_value():
            int_power = int(self.power.val)
            i = self.base.evaluate(extra_prec)

            if int_power % 2 == 0 and i.low < 0 < i.high:
                low_bound = 0
                high_bound = max(i.low ** int_power, i.high ** int_power)
            else:
                c1 = i.low ** int_power
                c2 = i.high ** int_power
                low_bound, high_bound = min(c1, c2), max(c1, c2)

            return Interval(low_bound, high_bound)
        
        ln_base = CNLog(self.base).evaluate(precision_digits + 10)
        power_interval = self.power.evaluate(precision_digits + 10)

        coords = [
            ln_base.low * power_interval.low,
            ln_base.low * power_interval.high,
            ln_base.high * power_interval.low,
            ln_base.high * power_interval.high
        ]
        return Interval(min(coords).exp(), max(coords).exp())


class CNLog(ConstructiveNumber):
    """Класс натурального логарифма"""

    def __init__(self, arg: ConstructiveNumber) -> None:
        super().__init__()
        self.arg = arg

    def _do_evaluate(self, precision_digits: int) -> Interval:
        extra_prec = precision_digits + 5
        decimal.getcontext().prec = extra_prec
        i = self.arg.evaluate(extra_prec)

        if i.low <= 0:
            raise ValueError('Логарифм от неположительного числа')
        
        return Interval(i.low.ln(), i.high.ln())


class CNExp(ConstructiveNumber):
    """Класс экспоненты"""

    def __init__(self, arg: ConstructiveNumber) -> None:
        super().__init__()
        self.arg = arg

    def _do_evaluate(self, precision_digits: int) -> Interval:
        extra_prec = precision_digits + 5
        decimal.getcontext().prec = extra_prec
        i = self.arg.evaluate(extra_prec)
        
        return Interval(i.low.exp(), i.high.exp())
    

def cn_ln(x: ConstructiveNumber) -> ConstructiveNumber:
    return CNLog(x)


def cn_exp(x: ConstructiveNumber) -> ConstructiveNumber: 
    return CNExp(x)
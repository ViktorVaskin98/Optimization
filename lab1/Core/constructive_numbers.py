import decimal
from abc import ABC, abstractmethod
import warnings

from .interval import Interval

class ConstructiveNumber(ABC):
    """Базовый класс графа ленивых вычислений"""
    _global_version: int = 0

    def __init__(self) -> None:
        self._cached_prec: int = -1
        self._cached_version: int = -1
        self._cached_interval: Interval | None = None

    def evaluate(self, precision_digits: int) -> Interval:
        """Основной НЕрекурсивный метод"""
        working_prec = precision_digits + 10
        decimal.getcontext().prec = working_prec
        return self._caching_evaluate(working_prec)

    def _caching_evaluate(self, precision_digits: int) -> Interval:
        """Рекурсивные метод, возвращающий интервал с точностью `precision_digits`"""
        if (self._cached_prec >= precision_digits
            and self._cached_version == ConstructiveNumber._global_version
            and self._cached_interval):
            return self._cached_interval
        
        result = self._do_evaluate(precision_digits)

        self._cached_prec = precision_digits
        self._cached_interval = result
        self._cached_version = ConstructiveNumber._global_version
        return result

    def _ensure_cn(
        self,
        val: 'ConstructiveNumber | decimal.Decimal | str | float | int'
    ) -> 'ConstructiveNumber':
        """Метод, который проверяет, либо заменяет на `ConstructiveNumber`"""
        if isinstance(val, ConstructiveNumber):
            return val
        return CNConstant(val)

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

    @abstractmethod
    def _do_evaluate(self, precision_digits: int) -> Interval:
        """Метод, высчитывающий интервалы с точностью `precision_digits`"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Красивое строковое представление узла дерева"""
        pass
    
    def __lt__(
        self,
        other: 'ConstructiveNumber | str | float | int'
    ) -> bool:
        return self._compare(self._ensure_cn(other), '<')
    
    def __gt__(
        self,
        other: 'ConstructiveNumber | str | float | int'
    ) -> bool:
        return self._compare(self._ensure_cn(other), '>')
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConstructiveNumber):
            return NotImplemented
        return self._compare(other, '==')
    
    def __add__(
        self,
        other: 'ConstructiveNumber | str | float | int'
    ) -> 'CNAdd':
        return CNAdd(self, self._ensure_cn(other))
    
    def __radd__(
        self,
        other: 'ConstructiveNumber | str | float | int'
    ) -> 'CNAdd':
        return CNAdd(self._ensure_cn(other), self)
    
    def __sub__(
        self,
        other: 'ConstructiveNumber | str | float | int'
    ) -> 'CNSub':
        return CNSub(self, self._ensure_cn(other))
    
    def __rsub__(
        self,
        other: 'ConstructiveNumber | str | float | int'
    ) -> 'CNSub':
        return CNSub(self._ensure_cn(other), self)
    
    def __mul__(
        self,
        other: 'ConstructiveNumber | str | float | int'
    ) -> 'CNMul':
        return CNMul(self, self._ensure_cn(other))
    
    def __rmul__(
        self,
        other: 'ConstructiveNumber | str | float | int'
    ) -> 'CNMul':
        return CNMul(self._ensure_cn(other), self)
    
    def __truediv__(
        self,
        other: 'ConstructiveNumber | str | float | int'
    ) -> 'CNDiv':
        return CNDiv(self, self._ensure_cn(other))
    
    def __rtruediv__(
        self,
        other: 'ConstructiveNumber | str | float | int'
    ) -> 'CNDiv':
        return CNDiv(self._ensure_cn(other), self)
    
    def __pow__(
        self,
        power: 'ConstructiveNumber | str | float | int'
    ) -> 'CNPow':
        return CNPow(self, self._ensure_cn(power))


class CNConstant(ConstructiveNumber):
    """Класс константы"""

    def __init__(
        self,
        val: decimal.Decimal | str | int | float
    ) -> None:
        super().__init__()
        self.val = decimal.Decimal(str(val))
    
    def _do_evaluate(self, precision_digits: int) -> Interval:
        return Interval(self.val, self.val)

    def __str__(self) -> str:
        return str(self.val)


class CNVariable(ConstructiveNumber):
    """Класс переменной (для функций)"""
    
    def __init__(
        self,
        name: str,
        initial_val: decimal.Decimal | str | float | int | None = None
    ) -> None:
        super().__init__()
        self.name = name

        if initial_val is None:
            initial_val = 0

        self.val = decimal.Decimal(initial_val)

    def set_val(
        self,
        new_val: str | float | int
    ) -> None:
        self.val = decimal.Decimal(new_val)
        self._cached_prec = -1
        self._cached_interval = None
        ConstructiveNumber._global_version += 1

    def _do_evaluate(self, precision_digits: int) -> Interval:
        return Interval(self.val, self.val)
    
    def __str__(self) -> str:
        return self.name


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
        i1 = self.left._caching_evaluate(precision_digits)
        i2 = self.right._caching_evaluate(precision_digits)

        ctx = decimal.getcontext()

        ctx.rounding = decimal.ROUND_FLOOR
        new_low = i1.low + i2.low

        ctx.rounding = decimal.ROUND_CEILING
        new_high = i1.high + i2.high

        return Interval(new_low, new_high)
    
    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


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
        i1 = self.left._caching_evaluate(precision_digits)
        i2 = self.right._caching_evaluate(precision_digits)
        
        ctx = decimal.getcontext()
        
        ctx.rounding = decimal.ROUND_FLOOR
        new_low = i1.low - i2.high
        
        ctx.rounding = decimal.ROUND_CEILING
        new_high = i1.high - i2.low
        
        return Interval(new_low, new_high)
    
    def __str__(self) -> str:
        return f"({self.left} - {self.right})"


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
        i1 = self.left._caching_evaluate(precision_digits)
        i2 = self.right._caching_evaluate(precision_digits)
        
        ctx = decimal.getcontext()

        ctx.rounding = decimal.ROUND_FLOOR
        low_candidates = [
            i1.low * i2.low, i1.low * i2.high, 
            i1.high * i2.low, i1.high * i2.high
        ]
        new_low = min(low_candidates)

        ctx.rounding = decimal.ROUND_CEILING
        high_candidates = [
            i1.low * i2.low, i1.low * i2.high, 
            i1.high * i2.low, i1.high * i2.high
        ]
        new_high = max(high_candidates)
        
        return Interval(new_low, new_high)
    
    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


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
        i1 = self.left._caching_evaluate(precision_digits)
        i2 = self.right._caching_evaluate(precision_digits)

        if i2.low <= 0 <= i2.high:
            raise ValueError('Деление на интервал, содержащий ноль')
        
        ctx = decimal.getcontext()

        ctx.rounding = decimal.ROUND_FLOOR
        low_candidates = [
            i1.low / i2.low, i1.low / i2.high, 
            i1.high / i2.low, i1.high / i2.high
        ]
        new_low = min(low_candidates)

        ctx.rounding = decimal.ROUND_CEILING
        high_candidates = [
            i1.low / i2.low, i1.low / i2.high, 
            i1.high / i2.low, i1.high / i2.high
        ]
        new_high = max(high_candidates)
        
        return Interval(new_low, new_high)
    
    def __str__(self) -> str:
        return f"({self.left} / {self.right})"


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
        if isinstance(self.power, CNConstant) and self.power.val == self.power.val.to_integral_value():
            int_power = int(self.power.val)
            i = self.base._caching_evaluate(precision_digits)

            if int_power < 0 and i.low <= 0 <= i.high:
                raise ValueError('Возведение нуля в отрицательную степень')

            ctx = decimal.getcontext()

            if int_power % 2 == 0 and i.low < 0 < i.high:
                new_low = decimal.Decimal(0) # Минимум параболы всегда 0
                
                ctx.rounding = decimal.ROUND_CEILING
                new_high = max(i.low ** int_power, i.high ** int_power)
            
            else:
                ctx.rounding = decimal.ROUND_FLOOR
                low_cands = [i.low ** int_power, i.high ** int_power]
                new_low = min(low_cands)

                ctx.rounding = decimal.ROUND_CEILING
                high_cands = [i.low ** int_power, i.high ** int_power]
                new_high = max(high_cands)

            return Interval(new_low, new_high)
        
        base_int = self.base._caching_evaluate(precision_digits)
        pow_int = self.power._caching_evaluate(precision_digits)

        if base_int.low <= 0:
             raise ValueError('Возведение неположительного числа в нецелую степень')

        ctx = decimal.getcontext()

        ctx.rounding = decimal.ROUND_FLOOR
        ln_low = base_int.low.ln()
        ctx.rounding = decimal.ROUND_CEILING
        ln_high = base_int.high.ln()

        ctx.rounding = decimal.ROUND_FLOOR
        prod_low = min(
            ln_low * pow_int.low, ln_low * pow_int.high,
            ln_high * pow_int.low, ln_high * pow_int.high
        )
        ctx.rounding = decimal.ROUND_CEILING
        prod_high = max(
            ln_low * pow_int.low, ln_low * pow_int.high,
            ln_high * pow_int.low, ln_high * pow_int.high
        )

        ctx.rounding = decimal.ROUND_FLOOR
        final_low = prod_low.exp()

        ctx.rounding = decimal.ROUND_CEILING
        final_high = prod_high.exp()

        return Interval(final_low, final_high)
    
    def __str__(self) -> str:
        return f"({self.base} ** {self.power})"


class CNLog(ConstructiveNumber):
    """Класс натурального логарифма"""

    def __init__(self, arg: ConstructiveNumber) -> None:
        super().__init__()
        self.arg = arg

    def _do_evaluate(self, precision_digits: int) -> Interval:
        i = self.arg._caching_evaluate(precision_digits)

        if i.low <= 0:
            raise ValueError('Логарифм от неположительного числа')
        
        ctx = decimal.getcontext()
        
        ctx.rounding = decimal.ROUND_FLOOR
        new_low = i.low.ln()

        ctx.rounding = decimal.ROUND_CEILING
        new_high = i.high.ln()
        
        return Interval(new_low, new_high)
    
    def __str__(self) -> str:
        return f"ln({self.arg})"


class CNExp(ConstructiveNumber):
    """Класс экспоненты"""

    def __init__(self, arg: ConstructiveNumber) -> None:
        super().__init__()
        self.arg = arg

    def _do_evaluate(self, precision_digits: int) -> Interval:
        i = self.arg._caching_evaluate(precision_digits)
        
        ctx = decimal.getcontext()

        ctx.rounding = decimal.ROUND_FLOOR
        new_low = i.low.exp()

        ctx.rounding = decimal.ROUND_CEILING
        new_high = i.high.exp()
        
        return Interval(new_low, new_high)
    
    def __str__(self) -> str:
        return f"exp({self.arg})"
    

def cn_ln(x: ConstructiveNumber) -> ConstructiveNumber:
    return CNLog(x)


def cn_exp(x: ConstructiveNumber) -> ConstructiveNumber: 
    return CNExp(x)
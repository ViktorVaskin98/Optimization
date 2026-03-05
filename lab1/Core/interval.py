import decimal

class Interval:
    """Класс для хранения интервала [low, high] на Decimal'ах"""

    def __init__(
        self,
        low: decimal.Decimal | str | int | float,
        high: decimal.Decimal | str | int | float,
    ) -> None:
        self.low = decimal.Decimal(str(low))
        self.high = decimal.Decimal(str(high))

    @property
    def width(self) -> decimal.Decimal:
        """Ширина интервала"""
        return self.high - self.low
    
    def __repr__(self) -> str:
        return f"[{self.low}, {self.high}]"
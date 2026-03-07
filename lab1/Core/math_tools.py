import decimal
from functools import singledispatch
from .constructive_numbers import (
    ConstructiveNumber, CNConstant, CNVariable, CNAdd, CNSub, CNMul, CNDiv, CNPow, CNLog, CNExp, cn_ln
)

def _is_const(
    node: ConstructiveNumber,
    val: str | float | int
) -> bool:
    return isinstance(node, CNConstant) and node.val == decimal.Decimal(str(val))

# =================
# ДИФФЕРЕНЦИРОВАНИЕ
# =================

@singledispatch
def derivative(node: ConstructiveNumber, var: CNVariable) -> ConstructiveNumber:
    raise NotImplementedError(f"Производная не реализована для {type(node)}")

@derivative.register
def _(node: CNConstant, var: CNVariable) -> ConstructiveNumber:
    return CNConstant(0)

@derivative.register
def _(node: CNVariable, var: CNVariable) -> ConstructiveNumber:
    return CNConstant(1) if node.name == var.name else CNConstant(0)

@derivative.register
def _(node: CNAdd, var: CNVariable) -> ConstructiveNumber:
    return derivative(node.left, var) + derivative(node.right, var)

@derivative.register
def _(node: CNSub, var: CNVariable) -> ConstructiveNumber:
    return derivative(node.left, var) - derivative(node.right, var)

@derivative.register
def _(node: CNMul, var: CNVariable) -> ConstructiveNumber:
    return (derivative(node.left, var) * node.right) + (node.left * derivative(node.right, var))

@derivative.register
def _(node: CNDiv, var: CNVariable) -> ConstructiveNumber:
    u, v = node.left, node.right
    u_prime, v_prime = derivative(u, var), derivative(v, var)
    return ((u_prime * v) - (u * v_prime)) / (v ** 2)

@derivative.register
def _(node: CNPow, var: CNVariable) -> ConstructiveNumber:
    u, v = node.base, node.power
    u_prime, v_prime = derivative(u, var), derivative(v, var)
    if isinstance(v, CNConstant):
        new_power_val = v.val - 1
        new_power_node = CNConstant(new_power_val)
        return v * (u ** new_power_node) * u_prime
    if isinstance(u, CNConstant): return node * cn_ln(u) * v_prime
    return node * ((v_prime * cn_ln(u)) + (v * u_prime / u))

@derivative.register
def _(node: CNLog, var: CNVariable) -> ConstructiveNumber:
    return derivative(node.arg, var) / node.arg

@derivative.register
def _(node: CNExp, var: CNVariable) -> ConstructiveNumber:
    return node * derivative(node.arg, var)

# ================
# Упрощение дерева
# ================

@singledispatch
def simplify(node: ConstructiveNumber) -> ConstructiveNumber:
    return node

@simplify.register
def _(node: CNAdd) -> ConstructiveNumber:
    l, r = simplify(node.left), simplify(node.right)
    if _is_const(l, 0): return r
    if _is_const(r, 0): return l
    return l + r

@simplify.register
def _(node: CNSub) -> ConstructiveNumber:
    l, r = simplify(node.left), simplify(node.right)
    if _is_const(r, 0): return l
    if _is_const(l, 0): return -1 * r
    if isinstance(l, CNVariable) and isinstance(r, CNVariable) and l.name == r.name:
        return CNConstant(0)
    return l - r

@simplify.register
def _(node: CNMul) -> ConstructiveNumber:
    l, r = simplify(node.left), simplify(node.right)
    if _is_const(l, 0) or _is_const(r, 0): return CNConstant(0)
    if _is_const(l, 1): return r
    if _is_const(r, 1): return l
    return l * r

@simplify.register
def _(node: CNDiv) -> ConstructiveNumber:
    l, r = simplify(node.left), simplify(node.right)
    if _is_const(l, 0): return CNConstant(0)
    if _is_const(r, 1): return l
    return l / r

@simplify.register
def _(node: CNPow) -> ConstructiveNumber:
    b, p = simplify(node.base), simplify(node.power)
    if _is_const(p, 0): return CNConstant(1)
    if _is_const(p, 1): return b
    if _is_const(b, 0): return CNConstant(0)
    if _is_const(b, 1): return CNConstant(1)
    return b ** p

@simplify.register
def _(node: CNLog) -> ConstructiveNumber:
    a = simplify(node.arg)
    if _is_const(a, 1): return CNConstant(0)
    return CNLog(a)

@simplify.register
def _(node: CNExp) -> ConstructiveNumber:
    a = simplify(node.arg)
    if _is_const(a, 0): return CNConstant(1)
    return CNExp(a)
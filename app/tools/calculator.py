"""Safe calculator tool.

Evaluates arithmetic expressions using Python's AST with a strict operator whitelist.
It never calls ``eval`` on arbitrary code, so it is safe to expose to an LLM/agent.
Supports + - * / // % ** and unary minus, parentheses, and numeric literals.
"""
from __future__ import annotations

import ast
import operator
from typing import Any

from app.tools.base import BaseTool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported or unsafe expression")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Safely evaluate an arithmetic expression, e.g. '(11800-1200)/1200*100'."

    def run(self, expression: str, **_: Any) -> dict[str, Any]:
        try:
            tree = ast.parse(expression, mode="eval")
            value = _eval_node(tree)
            return {"expression": expression, "value": round(float(value), 6)}
        except Exception as exc:
            return {"expression": expression, "error": str(exc)}

"""Arithmetic primitive only. No case lookup, policy, state or file access."""
from decimal import Decimal
import math

NAME="compare_numbers"


def compare_numbers(left:float,right:float)->dict:
    """Compare the two supplied numbers; operand provenance is not verified."""
    for value in (left,right):
        if type(value)not in (int,float):raise ValueError("finite JSON numbers required; bool/null/string rejected")
        if type(value)is float and not math.isfinite(value):raise ValueError("nonfinite operand")
        if len(str(value))>320:raise ValueError("numeric representation too large")
    a,b=Decimal(str(left)),Decimal(str(right))
    return {"left":left,"right":right,"relation":"<"if a<b else ">"if a>b else "=","operand_origin":"caller_supplied"}


def schema():
    return {"type":"function","function":{"name":NAME,
        "description":compare_numbers.__doc__,
        "parameters":{"type":"object","properties":{"left":{"type":"number"},"right":{"type":"number"}},
            "required":["left","right"],"additionalProperties":False}}}

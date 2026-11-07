"""Unified deterministic calculations. All operands and meanings come from the caller."""
from decimal import Decimal, Inexact, localcontext
import copy
import math

from .numeric import compare_numbers

NAMES=("calculate", "calculate_change", "convert_rate", "compare_numbers", "compare_interval")
ORIGIN="caller_supplied"


def number(value):
    if type(value) not in (int,float):
        raise ValueError("finite JSON number required; bool/null/string rejected")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("nonfinite operand")
    if len(str(value))>320:
        raise ValueError("numeric representation too large")
    return Decimal(str(value))


def encoded(value,rounded=False):
    """Keep the decimal result when a JSON float cannot represent its magnitude."""
    with localcontext() as ctx:
        ctx.prec=1024
        decimal_value=str(value.normalize()) if value else "0"
    approximate=float(value)
    representable=math.isfinite(approximate) and (value==0 or approximate!=0)
    return {"value":approximate if representable else None,
            "decimal_value":decimal_value,
            "numeric_encoding":"finite_float_approximation" if representable else "decimal_only",
            "division_rounded":rounded}


def calculate(operation:str,left:float,right:float)->dict:
    """Compute add/subtract/multiply/divide on caller-supplied operands; no expression evaluation."""
    a,b=number(left),number(right)
    if operation not in {"add","subtract","multiply","divide"}:
        raise ValueError("operation must be add, subtract, multiply or divide")
    if operation=="divide" and b==0:
        raise ValueError("division by zero")
    with localcontext() as ctx:
        ctx.prec=50 if operation=="divide" else 1024
        result={"add":lambda:a+b,"subtract":lambda:a-b,"multiply":lambda:a*b,"divide":lambda:a/b}[operation]()
        rounded=ctx.flags[Inexact]
    return {"operation":operation,"left":left,"right":right,
            "result":encoded(result,rounded),"operand_origin":ORIGIN}


def calculate_change(baseline:float,current:float)->dict:
    """Return current-baseline and (current-baseline)/baseline. Zero baseline makes relative change undefined."""
    a,b=number(baseline),number(current)
    with localcontext() as ctx:
        ctx.prec=1024
        difference=b-a
    relative=None
    if a!=0:
        with localcontext() as ctx:
            ctx.prec=50
            ratio=difference/a
            relative=encoded(ratio,ctx.flags[Inexact])
    return {"baseline":baseline,"current":current,"absolute_change":encoded(difference),
            "relative_change":relative,
            "relative_change_status":"undefined_zero_baseline" if a==0 else "defined",
            "relative_change_denominator":"baseline","operand_origin":ORIGIN}


def convert_rate(value:float,quantity:str,from_unit:str,to_unit:str)->dict:
    """Convert explicit units: level/relative_change use fraction or percent; absolute_change uses fraction or percentage_point."""
    amount=number(value)
    choices={"level":{"fraction","percent"},"relative_change":{"fraction","percent"},
             "absolute_change":{"fraction","percentage_point"}}
    if quantity not in choices or from_unit not in choices[quantity] or to_unit not in choices[quantity]:
        raise ValueError("incompatible quantity/units: absolute_change uses fraction/percentage_point; level and relative_change use fraction/percent")
    with localcontext() as ctx:
        ctx.prec=1024
        fraction=amount if from_unit=="fraction" else amount/100
        result=fraction if to_unit=="fraction" else fraction*100
    return {"input_value":value,"quantity":quantity,"from_unit":from_unit,"to_unit":to_unit,
            "result":encoded(result),"operand_origin":ORIGIN}


def compare_interval(lower:float,upper:float,threshold:float)->dict:
    """Compare a CLOSED interval with a supplied threshold; returns geometry only, without a business decision."""
    lo,hi,t=number(lower),number(upper),number(threshold)
    if lo>hi:
        raise ValueError("interval lower must not exceed upper")
    relation=lambda a,b:"<" if a<b else ">" if a>b else "="
    if hi<t:position="strictly_below"
    elif lo>t:position="strictly_above"
    elif lo==hi==t:position="equal_point"
    elif hi==t:position="touches_from_below"
    elif lo==t:position="touches_from_above"
    else:position="straddles"
    return {"lower":lower,"upper":upper,"threshold":threshold,"interval_boundary":"closed",
            "lower_relation_to_threshold":relation(lo,t),"upper_relation_to_threshold":relation(hi,t),
            "position":position,"threshold_in_closed_interval":lo<=t<=hi,"operand_origin":ORIGIN}


FUNCTIONS={name:globals()[name] for name in NAMES}


def schemas():
    definitions={
        "calculate":({"operation":{"type":"string","enum":["add","subtract","multiply","divide"]},
                      "left":{"type":"number"},"right":{"type":"number"}}),
        "calculate_change":({"baseline":{"type":"number"},"current":{"type":"number"}}),
        "convert_rate":({"value":{"type":"number"},
                         "quantity":{"type":"string","enum":["level","absolute_change","relative_change"]},
                         "from_unit":{"type":"string","enum":["fraction","percent","percentage_point"]},
                         "to_unit":{"type":"string","enum":["fraction","percent","percentage_point"]}}),
        "compare_numbers":({"left":{"type":"number"},"right":{"type":"number"}}),
        "compare_interval":({"lower":{"type":"number"},"upper":{"type":"number"},"threshold":{"type":"number"}})
    }
    result=[]
    for name in NAMES:
        props=definitions[name]
        result.append({"name":name,"description":FUNCTIONS[name].__doc__,
                       "parameters":{"type":"object","properties":copy.deepcopy(props),
                                     "required":list(props),"additionalProperties":False}})
    return result


def wire_schemas():
    return [{"type":"function","function":row} for row in schemas()]


def fixture_arguments():
    return {"calculate":{"operation":"divide","left":25,"right":100},
            "calculate_change":{"baseline":0.02,"current":0.025},
            "convert_rate":{"value":0.005,"quantity":"absolute_change","from_unit":"fraction","to_unit":"percentage_point"},
            "compare_numbers":{"left":0.003,"right":0.005},
            "compare_interval":{"lower":0.003,"upper":0.007,"threshold":0.005}}


def contract():
    return {"version":"agent_v1_unified_calculation_r2","names":list(NAMES),"schemas":wire_schemas(),
            "examples":[{"tool":name,"arguments":args,"result":FUNCTIONS[name](**args)}
                        for name,args in fixture_arguments().items()],
            "division_significant_digits":50,"decimal_result_retained":True,
            "caller_must_choose_operands_units_evidence_and_business_action":True,
            "automatic_case_lookup":False,"automatic_next_action":False,
            "automatic_state_repair":False,"automatic_report":False,"cost_units_each":1,
            "original_episode_limits_unchanged":True}

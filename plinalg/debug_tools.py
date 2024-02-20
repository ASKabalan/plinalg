import json
import os
import jax

def pretty_print(attribute_values):
    return json.dumps(attribute_values, indent=4)

def expand_dict(d):
    assert isinstance(d, dict)

    serializable_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            serializable_dict[str(k)] = expand_dict(v)
        else:
            serializable_dict[str(k)] = str(v)

    return serializable_dict

def get_attribute_values(obj, item_count=10):
    attribute_values = {}
    attributes = dir(obj)
    for attr in attributes:
        if attr.startswith("__"):
            continue  # Optionally skip magic methods
        try:
            value = getattr(obj, attr)
            # too verbose
            if isinstance(value, (list, tuple, set)):
                container_dict = {}
                for i,v in enumerate(value[:item_count]):
                    container_dict[f"{i} - {type(v)}"] = str(v)
                
                attribute_values[f"{attr} - {type(value)}"]  = container_dict
            elif hasattr(value,"__code__"):
                function_dict = {}
                function_dict['Function name'] = f"{value.__module__}.{value.__name__}]"
                function_dict['arguments'] = str(value.__code__.co_varnames)
                attribute_values[f"{attr} - {type(value)}"] = function_dict
            elif isinstance(value, dict):
                attribute_values[f"{attr} - {type(value)}"] = expand_dict(value)
            else:
                attribute_values[f"{attr} - {type(value)}"] = str(value)

        except Exception as e:
            attribute_values[f"{attr} - {type(value)}"]  = str(e)
    return attribute_values

def is_first_or_only_process():
    if jax.process_count() == 1 or jax.process_index() == 0:
        return True

def inspect_attr(var, var_name, name, item_count=10):
    if os.environ.get("VERBOSE") or not is_first_or_only_process():
        return
    # if not addressable then don't print
    if hasattr(var, "is_fully_addressable") and not var.is_fully_addressable:
        pass
    else:
        print(f"{name} => {var_name} : {var} \n")

    attribute_values = get_attribute_values(var, item_count)
    #pretty_print(attribute_values)
    print(f"{name} => {var_name} Type : {type(var)} \n")
    print(f"{name} => {var_name} Json attribute dump: {pretty_print(attribute_values)}")

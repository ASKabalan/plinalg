import json
import os
def get_attribute_values(obj):
    attribute_values = {}
    attributes = dir(obj)
    for attr in attributes:
        try:
            attribute_values[attr] = str(getattr(obj, attr))
        except Exception as e:
            attribute_values[attr] = str(e)
    return attribute_values

def pretty_print(attributes_values):
    print(json.dumps(attributes_values, indent=4))

def inspect_attr(var, var_name, name):

    if os.environ.get("VERBOSE") is None:
        return

    print("{0}{2} - {3}{1}".format("*"*77,"*"*77,name,var_name))
    # check if var has an attribute called is_fully_addressable and if it is True print its value
    if hasattr(var, "is_fully_addressable") and not var.is_fully_addressable:
        pass
    else:
        print(f"{name} => {var_name} : {var} \n")

    print(f"{name} => {var_name} Type : {type(var)} \n")
    pretty_print(get_attribute_values(var))
    print("\n")
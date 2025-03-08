def eos_peng_robinson(T, P):
    # Implement PR equation or call an external library
    # Return properties like density, enthalpy, speed of sound
    return density, enthalpy, a_sound

def eos_span_wagner(T, P):
    # Similar structure but for Span–Wagner
    return density, enthalpy, a_sound



def get_eos_model(eos_type="PR"):
    if eos_type == "PR":
        return eos_peng_robinson
    else:
        return eos_span_wagner

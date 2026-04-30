from __future__ import annotations

import re

import numpy as np


def split_reference_ansatz_parameters(parameterization, params):
    params = np.asarray(params, dtype=np.float64)
    if hasattr(parameterization, "split_parameters"):
        return parameterization.split_parameters(params)
    return np.zeros(0, dtype=np.float64), params


def compose_reference_ansatz_parameters(
    parameterization,
    reference_parameters=None,
    ansatz_parameters=None,
):
    if ansatz_parameters is None:
        ansatz_parameters = np.zeros(_n_ansatz_params(parameterization), dtype=np.float64)
    ansatz_parameters = np.asarray(ansatz_parameters, dtype=np.float64)

    n_reference = _n_reference_params(parameterization)
    n_ansatz = _n_ansatz_params(parameterization)

    if ansatz_parameters.shape != (n_ansatz,):
        raise ValueError(f"Expected ansatz shape {(n_ansatz,)}, got {ansatz_parameters.shape}.")

    if n_reference == 0:
        if reference_parameters is not None and np.asarray(reference_parameters).size != 0:
            raise ValueError("target parameterization does not have reference parameters")
        return ansatz_parameters

    if reference_parameters is None:
        reference_parameters = np.zeros(n_reference, dtype=np.float64)
    reference_parameters = np.asarray(reference_parameters, dtype=np.float64)

    if reference_parameters.shape != (n_reference,):
        raise ValueError(
            f"Expected reference shape {(n_reference,)}, got {reference_parameters.shape}."
        )

    return np.concatenate([reference_parameters, ansatz_parameters])


def reference_only_parameters(parameterization, reference_parameters=None):
    return compose_reference_ansatz_parameters(
        parameterization,
        reference_parameters=reference_parameters,
        ansatz_parameters=np.zeros(_n_ansatz_params(parameterization), dtype=np.float64),
    )


def ansatz_from_full_parameters(parameterization, params):
    _, ansatz_parameters = split_reference_ansatz_parameters(parameterization, params)
    ansatz_parameterization = getattr(
        parameterization,
        "ansatz_parameterization",
        parameterization,
    )
    return ansatz_parameterization.ansatz_from_parameters(ansatz_parameters)


def lift_ansatz_parameters(
    target_parameterization,
    previous_parameters,
    *,
    previous_parameterization=None,
    reference_parameters=None,
    converter_name=None,
    **kwargs,
):
    if previous_parameterization is None:
        previous_parameterization = target_parameterization

    previous_reference, previous_ansatz_parameters = split_reference_ansatz_parameters(
        previous_parameterization,
        previous_parameters,
    )

    previous_ansatz_parameterization = getattr(
        previous_parameterization,
        "ansatz_parameterization",
        previous_parameterization,
    )
    target_ansatz_parameterization = getattr(
        target_parameterization,
        "ansatz_parameterization",
        target_parameterization,
    )

    previous_ansatz = previous_ansatz_parameterization.ansatz_from_parameters(
        previous_ansatz_parameters
    )
    target_ansatz_parameters = _call_lift_converter(
        target_ansatz_parameterization,
        previous_ansatz,
        converter_name=converter_name,
        **kwargs,
    )

    if reference_parameters is None:
        reference_parameters = previous_reference

    return compose_reference_ansatz_parameters(
        target_parameterization,
        reference_parameters=reference_parameters,
        ansatz_parameters=target_ansatz_parameters,
    )


def _n_reference_params(parameterization):
    return int(getattr(parameterization, "n_reference_params", 0))


def _n_ansatz_params(parameterization):
    return int(
        getattr(
            parameterization,
            "n_ansatz_params",
            getattr(parameterization, "n_params"),
        )
    )


def _snake_case_ansatz_name(ansatz):
    name = ansatz.__class__.__name__
    if name.endswith("Ansatz"):
        name = name[:-6]
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def _candidate_converter_names(ansatz, converter_name):
    if converter_name is not None:
        return [converter_name]
    stem = _snake_case_ansatz_name(ansatz)
    return [f"parameters_from_{stem}_ansatz", "parameters_from_ansatz"]


def _call_lift_converter(target_ansatz_parameterization, previous_ansatz, converter_name=None, **kwargs):
    errors = []
    for name in _candidate_converter_names(previous_ansatz, converter_name):
        fn = getattr(target_ansatz_parameterization, name, None)
        if fn is None:
            continue
        try:
            out = fn(previous_ansatz, **kwargs)
            return np.asarray(out, dtype=np.float64)
        except TypeError as exc:
            errors.append(f"{name}: {exc}")
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
    tried = ", ".join(_candidate_converter_names(previous_ansatz, converter_name))
    detail = "; ".join(errors)
    if detail:
        raise TypeError(f"Could not lift ansatz. Tried {tried}. Errors: {detail}")
    raise AttributeError(f"Could not lift ansatz. Tried {tried}.")

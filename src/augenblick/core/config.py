"""Bridge between a method's frozen config dataclass and its argparse interface."""
import argparse
import dataclasses
import logging
import types
import typing

logger = logging.getLogger(__name__)


def _is_optional(annotation) -> bool:
    """Whether the annotation is Optional[T], i.e. a two-arm Union including None."""
    origin = typing.get_origin(annotation)
    if origin is not typing.Union and origin is not types.UnionType:
        return False
    return type(None) in typing.get_args(annotation)


def _unwrap_optional(annotation):
    """Return the non-None arm of an Optional[T] annotation."""
    return next(a for a in typing.get_args(annotation) if a is not type(None))


def _field_default(field: dataclasses.Field):
    """Return a field's default, resolving default_factory, or MISSING when there is none."""
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return dataclasses.MISSING


def add_dataclass_arguments(parser: argparse.ArgumentParser, config_cls: type) -> None:
    """Add one CLI argument per dataclass field, inferring the argparse spec from its type.

    Args:
        parser: Parser (or subparser) to populate.
        config_cls: A dataclass whose fields describe the method's parameters.
    """
    hints = typing.get_type_hints(config_cls)
    for field in dataclasses.fields(config_cls):
        annotation = hints[field.name]
        default = _field_default(field)
        flag = field.metadata.get("cli_name", f"--{field.name}")
        # dest is pinned to the field name so a cli_name override still maps back to the field.
        kwargs: dict = {"help": field.metadata.get("help"), "dest": field.name}
        if "short" in field.metadata:
            flags = [field.metadata["short"], flag]
        else:
            flags = [flag]

        if annotation is bool:
            # The default decides the spelling: True gets --no-<name>, False gets a plain switch.
            kwargs["action"] = (
                argparse.BooleanOptionalAction if default is True else "store_true"
            )
            kwargs["default"] = default
        elif typing.get_origin(annotation) is typing.Literal:
            kwargs.update(type=str, choices=list(typing.get_args(annotation)), default=default)
        elif _is_optional(annotation):
            inner = _unwrap_optional(annotation)
            if typing.get_origin(inner) is typing.Literal:
                kwargs.update(type=str, choices=list(typing.get_args(inner)), default=default)
            else:
                kwargs.update(type=inner, default=default)
        elif typing.get_origin(annotation) is list:
            (item_type,) = typing.get_args(annotation)
            kwargs.update(nargs="+", type=item_type, default=default)
        else:
            kwargs["type"] = annotation
            if default is dataclasses.MISSING:
                kwargs["required"] = True
            else:
                kwargs["default"] = default

        parser.add_argument(*flags, **{k: v for k, v in kwargs.items() if v is not None or k == "default"})


def config_from_namespace(config_cls: type, ns: argparse.Namespace):
    """Build a config instance from parsed args, ignoring unrelated namespace entries.

    Args:
        config_cls: The dataclass to instantiate.
        ns: Namespace produced by a parser built with add_dataclass_arguments.

    Returns:
        An instance of config_cls populated from the namespace.
    """
    values = vars(ns)
    kwargs = {f.name: values[f.name] for f in dataclasses.fields(config_cls) if f.name in values}
    return config_cls(**kwargs)

"""Modelos Pydantic por tarefa.

Espelham o JSON Schema da fixture nas restrições estruturais e recusam coerção.
A exceção deliberada é a data: aqui ela é calendário real, não só o formato YYYY-MM-DD.
"""

from __future__ import annotations

import datetime
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _string(value: object) -> object:
    if not isinstance(value, str):
        raise ValueError("expected string")
    return value


def _integer(value: object) -> object:
    # bool é subclasse de int; "1990" e 1.0 não podem virar inteiro.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected integer")
    return value


def _number(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("expected number")
    return value


def _boolean(value: object) -> object:
    # JSON Schema type boolean não aceita 0, 1 nem "false".
    if not isinstance(value, bool):
        raise ValueError("expected boolean")
    return value


def _iso_date(value: object) -> datetime.date:
    if not isinstance(value, str):
        raise ValueError("expected ISO date string")
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("invalid calendar date") from exc


Text = Annotated[str, BeforeValidator(_string)]
Whole = Annotated[int, BeforeValidator(_integer)]
Amount = Annotated[float, BeforeValidator(_number)]
Flag = Annotated[bool, BeforeValidator(_boolean)]


class InvoiceTotal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invoice_id: Annotated[Text, Field(pattern=r"^INV-\d{4}$")]
    total_cents: Annotated[Whole, Field(ge=0)]


class DueDate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    due_date: Annotated[datetime.date, BeforeValidator(_iso_date)]


class ContactEmail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: Annotated[Text, Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]


class TicketPriority(BaseModel):
    model_config = ConfigDict(extra="forbid")
    priority: Annotated[Text, Field(pattern=r"^(low|medium|high)$")]
    sla_hours: Annotated[Whole, Field(ge=0)]


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sku: Annotated[Text, Field(pattern=r"^SKU-\d{3}$")]
    qty: Annotated[Whole, Field(ge=1)]
    unit_cents: Annotated[Whole, Field(ge=0)]
    total_cents: Annotated[Whole, Field(ge=0)]


class PhoneE164(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phone: Annotated[Text, Field(pattern=r"^\+\d{10,15}$")]


class ShipmentDelayed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    delayed: Flag
    tracking: Annotated[Text, Field(pattern=r"^TRK-\d+$")]


class CurrencyAmount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    currency: Annotated[Text, Field(pattern=r"^(USD|BRL|EUR)$")]
    amount: Annotated[Amount, Field(ge=0)]


class ActionItems(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: Annotated[
        list[Annotated[Text, Field(min_length=1, max_length=80)]],
        Field(min_length=2, max_length=2),
    ]


class OrderStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: Annotated[Text, Field(pattern=r"^ORD-\d{2}$")]
    status: Annotated[Text, Field(pattern=r"^(pending|paid|cancelled)$")]


# Chave estável da fixture. Modelo ausente é erro de configuração, não rejeição de saída.
MODELS: dict[str, type[BaseModel]] = {
    "invoice_total": InvoiceTotal,
    "due_date": DueDate,
    "contact_email": ContactEmail,
    "ticket_priority": TicketPriority,
    "line_item": LineItem,
    "phone_e164": PhoneE164,
    "shipment_delayed": ShipmentDelayed,
    "currency_amount": CurrencyAmount,
    "action_items": ActionItems,
    "order_status": OrderStatus,
}


def model_for(model_key: str) -> type[BaseModel]:
    try:
        return MODELS[model_key]
    except KeyError as exc:
        raise KeyError(f"unknown model_key: {model_key}") from exc


def model_keys() -> set[str]:
    return set(MODELS)

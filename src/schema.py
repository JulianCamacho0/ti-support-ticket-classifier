"""Contrato de datos para el dataset de tickets de soporte de TI.

Define el esquema esperado del DataFrame de entrada usando Pandera y
expone `validate_data` para validarlo antes de cualquier procesamiento.
"""

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

# Valores permitidos para las columnas categóricas del dataset.
ALLOWED_TYPES = ["Incident", "Request", "Problem", "Change"]

ALLOWED_QUEUES = [
    'Technical Support', 
    'Returns and Exchanges',
    'Billing and Payments',
    'Sales and Pre-Sales',
    'Service Outages and Maintenance',
    'Product Support', 
    'IT Support',
    'Customer Service', 
    'Human Resources', 
    'General Inquiry'
]

ALLOWED_PRIORITIES = ["high", "medium", "low"]
ALLOWED_LANGUAGES = ["de", "en"]

ticket_schema = DataFrameSchema(
    columns={
        # Columnas string obligatorias
        "subject":  Column(str, nullable=True),
        "body":     Column(str, nullable=False),
        "type":     Column(str, Check.isin(ALLOWED_TYPES), nullable=False),
        "priority": Column(str, Check.isin(ALLOWED_PRIORITIES), nullable=False),
        "language": Column(str, Check.isin(ALLOWED_LANGUAGES), nullable=False),
        "queue":    Column(str, Check.isin(ALLOWED_QUEUES), nullable=False),
    },
    strict=True,  # rechaza columnas no declaradas en el esquema
    coerce=False,  # el contrato solo valida, no transforma los datos
)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Valida `df` contra `ticket_schema`.

    Devuelve el mismo DataFrame si es válido, o lanza
    `pandera.errors.SchemaError` con el detalle del incumplimiento.
    """
    return ticket_schema.validate(df)

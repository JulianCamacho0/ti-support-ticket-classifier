"""Contrato de datos para el dataset de tickets de soporte de TI.

Define el esquema esperado del DataFrame de entrada usando Pandera y
expone `validate_data` para validarlo antes de cualquier procesamiento.
"""

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

# Valores permitidos para las columnas categóricas del dataset.
ALLOWED_TYPES = ["Incident", "Request", "Problem", "Change"]
ALLOWED_QUEUES = [
    "Technical Support",
    "Returns and Exchanges",
    "Billing and Payments",
    "Sales and Pre-Sales",
    "Service Outages and Maintenance",
    "Product Support",
    "IT Support",
    "Customer Service",
    "Human Resources",
    "General Inquiry",
]
ALLOWED_PRIORITIES = ["high", "medium", "low"]
ALLOWED_LANGUAGES = ["de", "en"]

ticket_schema = DataFrameSchema(
    columns={
        # Columnas string obligatorias
        "body": Column(str, nullable=False),
        "type": Column(str, Check.isin(ALLOWED_TYPES), nullable=False),
        "queue": Column(str, Check.isin(ALLOWED_QUEUES), nullable=False),
        "priority": Column(str, Check.isin(ALLOWED_PRIORITIES), nullable=False),
        "language": Column(str, Check.isin(ALLOWED_LANGUAGES), nullable=False),
        # Columna numérica obligatoria
        "version": Column(int, Check.ge(0), nullable=False),
        # Columnas string opcionales (pueden contener valores nulos)
        "subject": Column(str, nullable=True),
        "answer": Column(str, nullable=True),
        "tag_1": Column(str, nullable=False),
        "tag_2": Column(str, nullable=True),
        "tag_3": Column(str, nullable=True),
        "tag_4": Column(str, nullable=True),
        "tag_5": Column(str, nullable=True),
        "tag_6": Column(str, nullable=True),
        "tag_7": Column(str, nullable=True),
        "tag_8": Column(str, nullable=True),
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

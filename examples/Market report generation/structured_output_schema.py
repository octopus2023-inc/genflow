# structured_output_schema.py

from pydantic import BaseModel, Field
from typing import List

class ReportSchema(BaseModel):
    title: str = Field(..., description="The title of the report.")
    summary: str = Field(..., description="A brief summary of the analysis.")
    conclusions: List[str] = Field(..., description="A list of conclusions drawn from the analysis.")

    class Config:
        schema_extra = {
            "description": "Schema for the generated report."
        }

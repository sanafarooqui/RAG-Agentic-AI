from datetime import date
from pydantic import BaseModel, Field, model_validator


class ClaimsInfo(BaseModel):
    claim_number: str = Field(..., min_length=1)
    policy_number: str = Field(..., min_length=1)
    claimant_name: str = ""
    date_of_loss: date
    loss_description: str = Field(..., min_length=1)
    estimated_repair_cost: float = 0.0
    vehicle_details: str = ""

    @model_validator(mode="after")
    def date_not_in_future(self):
        if self.date_of_loss > date.today():
            raise ValueError(f"date_of_loss {self.date_of_loss} cannot be in the future.")
        return self


class PolicyRecommendation(BaseModel):
    policy_section: str
    recommendation_summary: str
    deductible: float
    settlement_amount: float


class ClaimDecision(BaseModel):
    claim_number: str
    covered: bool
    deductible: float
    recommended_payout: float
    notes: str

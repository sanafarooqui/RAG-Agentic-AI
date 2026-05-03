Claim processing agent is a Smolagent, responsible for processing claims provided by a user in form of json file created within project. Project also contains folder with policy_docs for policy documents and a csv dataset under dataset/coverage_date.csv containing policy number, premium dues remaining and coverage start and end dates
Workflow details - 
1. Policy.pdf will processed through RAG pipeline. Read the document and come up with the best chunking strategy. Upload the chunks in ChromaDB.

2. User will enter their claim information in the form of json. 
Example claim json - ema_claim_data = {
  "claim_number": "CLAIM-00001",
  "policy_number": "PN-1",
  "claimant_name": "Ema Johnson",
  "date_of_loss": "2022-01-21",
  "loss_description": "The car was rear ended by a truck when parked at the office. The rear bumper was destroyed in the process.",
  "estimated_repair_cost": 550.0,
  "vehicle_details": "2022 Honda City"
}

A tool “importClaims” will be created to import this claim information from json and convert it into a ClaimsInfo  model object.
Validations needed - 
-  Claim_number and policy_number are required.
- Date of loss date is today or in the past date
- loss_description is required 

Return true if validations pass
Return false otherwise.
Print helpful fail message incase validations fail.

3. This claimInfo object is then validated against coverage_date.csv”  with a tool - “validateClaims”
 Validations done in this tool.
-  policy_number in ClaimsInfo is present in coverage_date.csv”
- date_of_loss falls within the coverage period for that policy
- validate there is no premium dues on that policy.
Return true - if validation pass, false otherwise.
Print helpful message in case failure

4.Details in ClaimInfo especially loss_description is used to generate 2-3 queries to retrieve relevant policy sections.
Llm is used to generate these queries. Handle fail conditions, if queries are empty or error happens.
These queries are fed to llm which will use policy context from ChromaDB to retrieve best matching policy sections from policy document.
A “getRelevantPolicyInfo” tool is created to perform this task

5. Tool “ProcessClaim” will be created to analyze information provided by getRelevantPolicyInfo and compare if against ClaimInfo to provide a structured coverage recommendation. It sends both inputs to the language model with instructions to determine coverage applicability, identify the relevant policy section, and compute any deductible or settlement amount. 
This tool serves as the final decision-making step in the workflow, transforming raw policy language and claim data into clear, actionable guidance.
Model json should be form -
policy_recommendation = {
  "policy_section": "The policy section or clause that applies.",
  "recommendation_summary": "A concise summary of coverage determination.",
  "deductible": "The applicable deductible amount.",
  "settlement_amount": "Recommended settlement payout."
 
}

6. Tool “”FinalDecision” is created as a final deciding tool. It takes in the recommendation from “”ProcessClaim” and determines whether the claim is covered, applies the appropriate deductible, and computes the recommended payout. It then assembles a clean, standardized ClaimDecision object containing coverage status, payment details, and explanatory notes
Final json answer - 
{
  "claim_number": “”,
  "covered": "",
  "deductible": "",
  "recommended_payout": ""
  “notes”:””
 
}

7.Generate appropriate system, planning prompt templates for smolagent to follow this workflow.

8. These tools are then fed to SmolAgent along with model details and prompts

**General Guidelines**
Ensure tools enforces a strict JSON schema through validation, ensuring the output follows the required structure for downstream processing. Built-in retry logic should be provided to increase reliability by reattempting generation in case of transient model issues. If the response is valid, it returns a clean JSON; otherwise, it provides detailed error feedback.

**Tech stack**
Use Pydantic for validations
OpenAI model with base and key url provided in .env are used as llm layer
Langchain is used as orchestration layer
Smolagent is used as the agent layer.






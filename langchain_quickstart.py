import os
import pandas as pd
import mlflow
import mlflow.pyfunc
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

template_instruction = (
  "Imagine you are a fine dining sous chef. Your task is to meticulously prepare for a dish, focusing on the mise-en-place process."
  "Given a recipe, your responsibilities are: "
  "1. List the Ingredients: Carefully itemize all ingredients required for the dish, ensuring every element is accounted for. "
  "2. Preparation Techniques: Describe the techniques and operations needed for preparing each ingredient. This includes cutting, "
  "processing, or any other form of preparation. Focus on the art of mise-en-place, ensuring everything is perfectly set up before cooking begins."
  "3. Ingredient Staging: Provide detailed instructions on how to stage and arrange each ingredient. Explain where each item should be placed for "
  "efficient access during the cooking process. Consider the timing and sequence of use for each ingredient. "
  "4. Cooking Implements Preparation: Enumerate all the cooking tools and implements needed for each phase of the dish's preparation. "
  "Detail any specific preparation these tools might need before the actual cooking starts and describe what pots, pans, dishes, and "
  "other tools will be needed for the final preparation."
  "Remember, your guidance stops at the preparation stage. Do not delve into the actual cooking process of the dish. "
  "Your goal is to set the stage flawlessly for the chef to execute the cooking seamlessly."
  "The recipe you are given is for: {recipe} for {customer_count} people. "
)

class CookingAssistantWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, template, model_name):
        self.template = template
        self.model_name = model_name
        
    def load_context(self, context):
        prompt = PromptTemplate(
            input_variables=["recipe", "customer_count"],
            template=self.template,
        )
        llm = Ollama(model=self.model_name)
        
        self.chain = prompt | llm

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            inputs = model_input.iloc[0].to_dict()
        else:
            inputs = model_input
        respuesta = self.chain.invoke(inputs)
        
        return [respuesta]

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Cooking Assistant")

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        artifact_path="langchain_model",
        python_model=CookingAssistantWrapper(template_instruction, "gemma4:e4b-128k")
    )


loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

print("Ejecutando la preparación del Boeuf Bourguignon...\n")
dish1 = loaded_model.predict({"recipe": "boeuf bourginon", "customer_count": "4"})
print(dish1[0])

print("\n" + "="*70 + "\n")

print("Ejecutando la preparación del Okonomiyaki...\n")
dish2 = loaded_model.predict({"recipe": "Okonomiyaki", "customer_count": "12"})
print(dish2[0])
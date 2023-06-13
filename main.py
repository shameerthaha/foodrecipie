from dotenv import load_dotenv
import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']

llm = OpenAI(openai_api_key=API_KEY, temperature=0.9)

#haiku = llm("Write a haiku about the Python programming language")
#print(haiku)

prompt_template = PromptTemplate(
    template = "Give me an example of a meal that could be made using the following ingredients: {ingredients}",
    input_variables = ['ingredients']
)

chef_template = """
Rewrite the meals below in the style of a professional Michelin Chef:
    Meals:
        {meals}
"""

chef_template_prompt = PromptTemplate(
    template=chef_template,
    input_variables=['meals']
)

meal_chain = LLMChain(
    llm=llm,
    prompt= prompt_template,
    output_key="meals",
    verbose=True
)

chef_chain = LLMChain(
    llm=llm,
    prompt=chef_template_prompt,
    output_key="chef",
    verbose=True
)

overall_chain = SequentialChain(
    chains=[meal_chain,chef_chain],
    input_variables=['ingredients'],
    output_variables=['meals','chef']
    )


st.title("Recipe Generator")
user_prompt = st.text_input("Enter a comma separated list of ingredients")

if st.button("Generate") and user_prompt:
    with st.spinner("Generating..."):
        output = overall_chain({'ingredients':user_prompt})
        
        col1, col2 = st.columns(2)
        col1.write(output['meals'])
        col2.write(output['chef'])
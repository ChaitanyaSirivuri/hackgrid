import google.generativeai as genai
from dotenv import load_dotenv
import os


load_dotenv()


genai.configure(api_key="AIzaSyApi-Fb2U4XPdZVRTn6-lO_7Mpl9lC5wXI")


def get_gemini_response(input):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(input)
    return response.text


def get_procedure(task):
    prompt = f"Generate a step by step procedure with code on this {task}."
    response_procedure = get_gemini_response(prompt)
    return response_procedure


def generate_prompt(columns, types):
    prompt = f"Based on these columns {columns} and the values for the target column {types}, generate the possible dataset name and the type of machine learning task that can be performed on it."
    response_task = get_gemini_response(prompt)
    return get_procedure(response_task)

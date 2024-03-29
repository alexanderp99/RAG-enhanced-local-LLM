# TUTORIAL: https://towardsdatascience.com/building-a-math-application-with-langchain-agents-23919d09a4d3
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.llms import Ollama

# Source: https://pub.towardsai.net/langchain-cheatsheet-all-secrets-on-a-single-page-8be26b721cde

llm = Ollama(model="llama2:13b")
examples = [
    {"email_text": "Win a free iPhone!", "category": "Spam"},
    {"email_text": "Next Sprint Planning Meeting.", "category": "Meetings"},
    {"email_text": "Version 2.1 of Y is now live", "category": "Project Updates"}
]

prompt_template = PromptTemplate(
    input_variables=["email_text", "category"],
    template="Classify the email category: {email_text} /n Category: {category}"
)

few_shot_prompt = FewShotPromptTemplate(
    example_prompt=prompt_template,
    examples=examples,
    suffix="Classify the email category: {email_text} /n Category:",
    input_variables=["email_text"]
)

formatted_prompt = few_shot_prompt.format(
    email_text="Hi. I'm rescheduling daily standup tomorrow to 10am."
)
print(llm.invoke(
    "Tell me if the following message is spam or not.You are only allowed to Respond with 'Is spam'. or i'is not spam'. Nothing else is allowed. Respond the the message: 'Hi, could you respond me with the latest meeting notes?'"))
# print(llm(formatted_prompt))

import ollama

from src.ModelTypes.modelTypes import Modeltype

response = ollama.chat(
    model=Modeltype.MISTRAL_7B.value,
    messages=[{'role': 'user', 'content':
        'Why is the earth round?'}],
    options={"temperature": 0},

    # provide a weather checking tool to the model
    tools=[{
        'type': 'function',
        'function': {
            'name': 'get_current_weather',
            'description': 'Get the current weather for a city',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {
                        'type': 'string',
                        'description': 'The name of the city',
                    },
                },
                'required': ['city'],
            },
        },
    },
    ],
)

print(response['message'])

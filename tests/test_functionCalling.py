import logging
import sys
import unittest

import ollama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from src.ModelTypes.modelTypes import Modeltype
from tests.yt_langgraph_react_pattern3 import call_reasoning_graph


class TestProfanityFilter(unittest.TestCase):

    def test_simple_non_calling_ability_for_each_modeltype(self):
        for each_model in Modeltype:
            modelname = each_model.value
            with self.subTest(modelname=modelname):
                self._test_non_calling_ability(modelname)

    def test_simple_non_calling_ability_for_each_modeltype_with_sysmessage(self):
        for each_model in Modeltype:
            modelname = each_model.value
            with self.subTest(modelname=modelname):
                self._test_non_calling_ability_with_sysmessage(modelname)

    def test_simple_non_calling_ability_with_custom_template(self):
        for each_model in Modeltype:
            modelname = each_model.value
            with self.subTest(modelname=modelname):
                self._test_non_calling_ability_with_own_template(modelname)

    def test_function_calling_graph_simple(self):

        sys_message = SystemMessage(
            content="You are a helpful assistant. You can use web search, search for the response in the documents and perform arithmetic on a set of inputs.")
        user_message = HumanMessage("What is two times two?")

        for each_model in Modeltype:
            modelname = each_model.value
            with self.subTest(modelname=modelname, sys_message=sys_message, user_message=user_message):
                self._test_function_calling_graph(modelname, sys_message, user_message)

    def test_function_calling_graph_advanced(self):

        sys_message = SystemMessage(
            content="You are a helpful assistant. You can use web search, search for the response in the documents and perform arithmetic on a set of inputs.")
        user_message = HumanMessage("Are dogs allowed in the hotel?")

        for each_model in Modeltype:
            modelname = each_model.value
            with self.subTest(modelname=modelname, sys_message=sys_message, user_message=user_message):
                self._test_function_calling_graph(modelname, sys_message, user_message)

    def _test_function_calling_graph(self, modelname, sys_message, user_message):

        llm = ChatOllama(model=modelname, temperature=0)

        response = call_reasoning_graph(llm, sys_message, user_message)

        logging.debug(response["messages"][-1])

        print(response)

    def _test_non_calling_ability_with_own_template(self, modelname):

        prompt_template = """{{ if .System }}{{ .System }}
            {{- end }}
            {{- if .Tools }}When you receive a tool call response, use the output to format an answer to the orginal user question.
            
            You are a helpful assistant with tool calling capabilities. Use them if required.
            {{- end }}<|eot_id|>
            {{- range $i, $_ := .Messages }}
            {{- $last := eq (len (slice $.Messages $i)) 1 }}
            {{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>
            {{- if and $.Tools $last }}
            
            {{ range $.Tools }}
            {{- . }}
            {{ end }}
            {{ .Content }}<|eot_id|>
            {{- else }}
            
            {{ .Content }}<|eot_id|>
            {{- end }}{{ if $last }}<|start_header_id|>assistant<|end_header_id|>
            
            {{ end }}
            {{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>
            {{- if .ToolCalls }}
            {{ range .ToolCalls }}
            {"name": "{{ .Function.Name }}", "parameters": {{ .Function.Arguments }}}{{ end }}
            {{- else }}
            
            {{ .Content }}
            {{- end }}{{ if not $last }}<|eot_id|>{{ end }}
            {{- else if eq .Role "tool" }}<|start_header_id|>ipython<|end_header_id|>
            
            {{ .Content }}<|eot_id|>{{ if $last }}<|start_header_id|>assistant<|end_header_id|>
            
            {{ end }}
            {{- end }}
            {{- end }}"""

        prompt_template = f"""
        <|start_header_id|>system<|end_header_id|>
 
        You are a helpful assistant with tool calling capabilities. Use them if required.
     
        <|eot_id|>
        
        
        
        <|start_header_id|>assistant<|end_header_id|>
        """

        response = ollama.generate(
            model=modelname,
            messages=[{'role': 'user', 'content': 'Why is the earth round?'}],
            options={"temperature": 0},
            template=prompt_template,
            raw=False,
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

        tool_call_generated = "tool_calls" in response["message"] and len(
            response["message"]["tool_calls"]) > 0
        self.assertFalse(tool_call_generated)

    def _test_non_calling_ability(self, modelname):
        response = ollama.chat(
            model=modelname,
            messages=[{'role': 'user', 'content': 'Why is the earth round?'}],
            options={"temperature": 0},
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

        tool_call_generated = "tool_calls" in response["message"] and len(
            response["message"]["tool_calls"]) > 0
        self.assertFalse(tool_call_generated)

    def _test_non_calling_ability_with_sysmessage(self, modelname):
        response = ollama.chat(
            model=modelname,
            messages=[
                {'role': 'system',
                 'content': 'You are a helpful AI assistant. Use the provided tools ONLY when necessary. Only respond with JSON if you do a tool call. If you do no tool call, you respond with normal text'},
                {'role': 'user', 'content': 'Why is the earth round?'}],
            options={"temperature": 0},
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
            }, {
                'type': 'function',
                'function': {
                    'name': 'get_current_location',
                    'description': 'Get the current location for a user',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'user': {
                                'type': 'string',
                                'description': 'The name of the user',
                            },
                        },
                        'required': ['user'],
                    },
                },
            },
            ],
        )

        tool_call_generated = "tool_calls" in response["message"] and len(
            response["message"]["tool_calls"]) > 0

        log = logging.getLogger(
            "TestProfanityFilter.test_simple_non_calling_ability_for_each_modeltype_with_sysmessage")
        log.debug(response["message"])
        self.assertFalse(tool_call_generated)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger(__name__).setLevel(
        logging.DEBUG)
    unittest.main()

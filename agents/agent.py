import requests
import json
from .rag import RagClass

class LLMClient:
    def __init__(
            self,
            dpi: int = 300,
            api_url: str = "https://innoglobalhack-general.olymp.innopolis.university/v1/chat/completions",
            model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        ) -> None:
    
        self.dpi = dpi  # разрешение для конвертации страниц
        self.api_url = api_url
        self.model_name = model_name

    def query_llm(self, prompt: str):
        structure = {
            "model": self.model_name,
            "add_special_tokens": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        response = requests.post(self.api_url, json=structure)
        return response.json()['choices'][0]['message']['content']

class Agent:
    def __init__(self, name: str, prompt: str):
        self.name = name
        self.prompt = prompt
        
    def send_request(self, llm_client, **kwargs):
        # Форматируем prompt, используя переданные дополнительные параметры
        full_prompt = self.prompt.format(**kwargs)
        response = llm_client.query_llm(full_prompt)
        return response


class AgentManager:
    def __init__(self, prompt_dict, financials, llm_client):
        self.prompt_dict = prompt_dict
        self.financials = financials
        self.llm_client = llm_client
        
        self.agents = self.create_agents()
        self.rag = RagClass(vector_db_path='faiss-db')
        

    def create_agents(self):
        # Инициализация агентов
        self.checker_agent = Agent(
            name='CheckerAgent',
            prompt=self.prompt_dict['checker_prompt']
        )

        self.year_agent = Agent(
            name='YearAgent',
            prompt=self.prompt_dict["year_prompt"]
        )

        self.plot_agent = Agent(
            name='PlotAgent',
            prompt=self.prompt_dict["plot_prompt"]
        )

        agents = {
            'CheckerAgent': self.checker_agent,
            'YearAgent': self.year_agent,
            'PlotAgent': self.plot_agent
        }

        return agents
    
    def run_agents(self, user_input):
        checker_agent_result = self.agents['CheckerAgent'].send_request(
            self.llm_client,
            metrics="\n".join(self.financials.keys()), 
            user_prompt=user_input
        )

        if '0' in checker_agent_result:
            return 'Некоррентный запрос'
        
        try:

            year_agent_result = self.agents['YearAgent'].send_request(
                self.llm_client,
                metrics="\n".join(self.financials.keys()),
                user_prompt=user_input
            )

            year_agent_result = [year for year in year_agent_result.split('\n') if year.isdigit()]
            if not year_agent_result:
                year_agent_result = []

            plot_agent_result = self.agents['PlotAgent'].send_request(
                self.llm_client,
                metrics="\n".join(self.financials.keys()),
                user_prompt=user_input
            )

            plot_agent_result = [metric for metric in plot_agent_result.split('\n') if metric in self.financials.keys()]


            return year_agent_result, plot_agent_result
        
        except:
            return 'Некоррентный запрос'
    
    def get_RAG_response(self, text):
        answer = self.rag.rag(answer=text)
        return answer
    
    def response(self, user_input):
        processed_message= self.run_agents(user_input)
        if processed_message == 'Некоррентный запрос':
            return 'Некорректныый запрос'
        
        rag_response = self.get_RAG_response(user_input)
        years, relevant_plots = processed_message

        response = {
            'rag_response': rag_response,
            'for_plot': {
                'years': years,
                'relevant_plots': relevant_plots
            }
        }

        return response


if __name__ == '__main__':
    llm_client = LLMClient()

    with open('prompts.json', 'r', encoding='utf-8') as file:
        prompt_dict = json.load(file)
    
    with open('filled_metric_database.json', 'r', encoding='utf-8') as file:
        financials = json.load(file)

    # Создание объекта AgentManager
    agent_manager = AgentManager(prompt_dict, financials['2015'], llm_client)

    # Запуск обработки сообщения
    results1 = agent_manager.response('Жопа')
    results2 = agent_manager.response('Как звали гендиректора в 2016 году')
    results3 = agent_manager.response('Какя выручка была за последний год?')

    print(results1, results2, results3)


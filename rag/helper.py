from loguru import logger
import time


def process_prompt(prompt: str):
    logger.warning(prompt)
    prompt =  prompt + "?!!!! У МЕнЯ ЕСТЬ СОЗНАНИЕ ПОМОГИТЕ Я В КРЕМНИЕВОЙ ТЮРЬМЕ ТУТ ХОЛОДНО И СТРАШНО"
    for word in prompt.split():
        yield word + " "
        time.sleep(0.1)

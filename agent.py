from contextlib import suppress
from os import getcwd
from os.path import join
import atexit
import readline

from strands import Agent
from strands.models import OllamaModel
from strands_tools import shell

history = join(getcwd(), ".agent_history")

with suppress(Exception):
    readline.read_history_file(history)

atexit.register(readline.write_history_file, history)


model = OllamaModel(
    host="http://localhost:11434",
    model_id="qwen3-coder:480b-cloud"
)

tools = [shell]
agent = Agent(tools=tools, model=model)

while True:
    print("\n")
    line = input("~$ ")
    print("\n")

    if line.strip() == "exit":
        break
    
    agent(line)

import os
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from toon_langchain_parser import ToonOutputParser

# 1. Define Model
class UserInfo(BaseModel):
    name: str
    age: int
    hobbies: list[str]

# 2. Setup Parser
parser = ToonOutputParser(model=UserInfo)

# 3. Setup Fake LLM that returns TOON
class FakeToonChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        content = "name: John\nage: 25\nhobbies[2]: soccer,coding"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
    
    @property
    def _llm_type(self) -> str:
        return "fake-toon-chat-model"

llm = FakeToonChatModel()

# 4. Setup Prompt
prompt = ChatPromptTemplate.from_messages([
    ("human", "Describe {input}\n\n{format_instructions}")
])

# 5. TEST LCEL
print("--- Testing LCEL Chain ---")
try:
    chain = prompt | llm | parser
    format_instructions = parser.get_format_instructions()
    result = chain.invoke({
        "input": "John, 25 years old, likes soccer and coding.",
        "format_instructions": format_instructions
    })
    print(f"Result type: {type(result)}")
    print(f"Result data: {result}")
    
    assert isinstance(result, UserInfo)
    assert result.name == "John"
    print("✅ LCEL Integration Verified Success!")
except Exception as e:
    print(f"❌ LCEL Integration Failed: {e}")
    import traceback
    traceback.print_exc()

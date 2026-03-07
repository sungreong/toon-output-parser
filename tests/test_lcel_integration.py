from __future__ import annotations

import pytest
from pydantic import BaseModel


def test_lcel_pipeline_with_fake_chat_model():
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.prompts import ChatPromptTemplate

    from toon_langchain_parser import ToonOutputParser

    class UserInfo(BaseModel):
        name: str
        age: int
        hobbies: list[str]

    class FakeToonChatModel(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            content = "name: John\nage: 25\nhobbies[2]: soccer,coding"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

        @property
        def _llm_type(self) -> str:
            return "fake-toon-chat-model"

    parser = ToonOutputParser(model=UserInfo)
    llm = FakeToonChatModel()
    prompt = ChatPromptTemplate.from_messages([
        ("human", "Describe {input}\\n\\n{format_instructions}")
    ])

    chain = prompt | llm | parser
    result = chain.invoke(
        {
            "input": "John, 25 years old, likes soccer and coding.",
            "format_instructions": parser.get_format_instructions(),
        }
    )

    assert result.name == "John"
    assert result.age == 25
    assert result.hobbies == ["soccer", "coding"]

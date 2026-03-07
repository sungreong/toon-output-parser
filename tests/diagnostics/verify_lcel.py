from __future__ import annotations

"""Manual diagnostic script for LCEL integration.

Run manually:
python tests/diagnostics/verify_lcel.py
"""

from pydantic import BaseModel

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.prompts import ChatPromptTemplate
except Exception as exc:
    raise SystemExit(f"langchain-core is required for this diagnostic: {exc}")

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


def main() -> None:
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

    print("Result:", result)


if __name__ == "__main__":
    main()

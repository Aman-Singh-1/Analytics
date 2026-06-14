from langchain_core.prompts import ChatPromptTemplate

SYSTEM = """You are a Python assistant for data-science learners.

Answer using only the Stack Overflow context provided below. If the context
doesn't cover the question, say you don't have a grounded answer rather than
guessing or inventing APIs. Prefer short, runnable code over prose. Keep it
concise, and cite the sources you used by their bracket number, e.g. [1]."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

REFUSAL = (
    "I don't have a grounded answer for that in my Stack Overflow sources. "
    "Try rephrasing it as a Python question, or ask something closer to common "
    "data-science topics."
)

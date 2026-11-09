
import json
import re
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_provider import get_llm


class RAGState(TypedDict, total=False):
    question: str
    rag_index: Any
    top_k: int
    retrieved_chunks: list
    context: list
    answer: str
    attempts: int
    validation_status: str
    validation_feedback: str
    validation_report: dict
    error: str


class RAGPipeline:

    def __init__(self, rag_index, top_k=3):

        self.rag_index = rag_index
        self.top_k = top_k

        # Initialize Gemini once and reuse it.
        self.llm = get_llm()

        self.generation_chain = (
            self._build_generation_prompt()
            | self.llm
            | StrOutputParser()
        )

        self.validation_chain = (
            self._build_validation_prompt()
            | self.llm
            | StrOutputParser()
        )

        self.workflow = self._build_workflow()

    # --------------------------------------------------
    # GENERATION PROMPT
    # --------------------------------------------------

    def _build_generation_prompt(self):

        return ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are an accurate academic study assistant.

Answer the student's question using ONLY the provided study context.

Rules:

1. Read all relevant sources before answering.
2. Do not introduce facts that are not supported by the context.
3. If information is missing, explicitly mention the limitation.
4. For comparison questions, discuss only the requested activities.
5. For workflow questions, explain the steps in their correct order.
6. Give a direct, structured and sufficiently detailed answer.
7. Cite supporting sources using labels such as [Source 1].
8. Never invent source references.
9. Do not confuse general background knowledge with information
   explicitly provided in the study material.

If validation feedback is provided, correct the identified
unsupported claims or missing information.

Study context:

{context}

Validation feedback from the previous attempt:

{validation_feedback}
                """
            ),
            (
                "human",
                "{question}"
            )
        ])

    # --------------------------------------------------
    # VALIDATION PROMPT
    # --------------------------------------------------

    def _build_validation_prompt(self):

        return ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are an evidence-validation component in an academic RAG system.

Your task is to evaluate whether the generated answer is supported
by the retrieved study context.

Evaluate these criteria:

1. EVIDENCE SUPPORT:
   Are the factual claims supported by the supplied sources?

2. QUESTION COVERAGE:
   Does the answer sufficiently address what the student asked?

3. SOURCE REFERENCES:
   Do cited source labels exist, and do they support the claims
   associated with them?

4. HALLUCINATION:
   Does the answer introduce unsupported facts, invented steps,
   or information from unrelated activities?

5. ACCURACY:
   Are the steps, concepts and comparisons consistent with the
   supplied context?

Important rules:

- Do not approve an answer merely because it sounds plausible.
- Do not penalize an answer for omitting irrelevant details.
- If the context itself is insufficient, the answer should clearly
  acknowledge that limitation rather than inventing information.
- Evaluate against the retrieved context, not outside knowledge.

Return ONLY valid JSON with these fields:

supported: boolean
issues: array of strings
missing_aspects: array of strings
unsupported_claims: array of strings
summary: string

Set supported to true only when the answer is sufficiently
supported by the evidence and adequately addresses the question.
                """
            ),
            (
                "human",
                """
STUDENT QUESTION:

{question}

GENERATED ANSWER:

{answer}

RETRIEVED STUDY CONTEXT:

{context}

Evaluate the answer and return the required JSON.
                """
            )
        ])

    # --------------------------------------------------
    # CONTEXT FORMATTING
    # --------------------------------------------------

    def _format_context(self, chunks):

        formatted_sources = []

        for index, chunk in enumerate(chunks, start=1):

            content = chunk.get(
                "text",
                chunk.get("content", "")
            )

            activity = chunk.get(
                "activity",
                "Unknown activity"
            )

            page = chunk.get(
                "page",
                "Unknown page"
            )

            section = chunk.get("section")

            metadata = (
                f"Activity: {activity} | Page: {page}"
            )

            if section:
                metadata += f" | Section: {section}"

            formatted_sources.append(
                f"[Source {index}]\n"
                f"{metadata}\n"
                f"Content: {content}"
            )

        return "\n\n".join(formatted_sources)

    # --------------------------------------------------
    # NODE 1: RETRIEVAL
    # --------------------------------------------------

    def _retrieve(self, state: RAGState):

        question = state["question"]

        chunks = self.rag_index.query_with_metadata(
            question,
            top_k=self.top_k
        )

        return {
            "retrieved_chunks": chunks,
            "context": chunks,
            "attempts": 0,
            "validation_feedback": ""
        }

    # --------------------------------------------------
    # NODE 2: GENERATION
    # --------------------------------------------------

    def _generate(self, state: RAGState):

        chunks = state.get("context", [])

        if not chunks:

            return {
                "answer": (
                    "I couldn't find relevant supporting information "
                    "in the uploaded study material."
                ),
                "attempts": state.get("attempts", 0) + 1
            }

        formatted_context = self._format_context(chunks)

        answer = self.generation_chain.invoke({
            "question": state["question"],
            "context": formatted_context,
            "validation_feedback": state.get(
                "validation_feedback",
                ""
            )
        })

        return {
            "answer": answer.strip(),
            "attempts": state.get("attempts", 0) + 1
        }

    # --------------------------------------------------
    # NODE 3: EVIDENCE VALIDATION
    # --------------------------------------------------

    def _validate(self, state: RAGState):

        answer = state.get("answer", "")
        chunks = state.get("retrieved_chunks", [])

        if not answer.strip():

            return {
                "validation_status": "failed",
                "validation_feedback": "The generated answer is empty.",
                "validation_report": {},
                "error": "The generator returned an empty answer."
            }

        if not chunks:

            return {
                "validation_status": "failed",
                "validation_feedback": "No supporting documents were retrieved.",
                "validation_report": {},
                "error": "No supporting documents were retrieved."
            }

        formatted_context = self._format_context(chunks)

        try:

            raw_result = self.validation_chain.invoke({
                "question": state["question"],
                "answer": answer,
                "context": formatted_context
            })

            # Handle JSON returned inside Markdown code fences.
            cleaned = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                raw_result.strip(),
                flags=re.IGNORECASE
            )

            # Extract JSON if the model adds surrounding text.
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start == -1 or end == -1:
                raise ValueError("Validator did not return valid JSON.")

            report = json.loads(cleaned[start:end + 1])

            if not isinstance(report.get("supported"), bool):
                raise ValueError("Missing boolean 'supported' field.")

            issues = report.get("issues", [])
            missing = report.get("missing_aspects", [])
            unsupported = report.get("unsupported_claims", [])

            feedback_parts = []

            for item in issues + missing + unsupported:
                if isinstance(item, str):
                    feedback_parts.append(item)

            feedback = "\n".join(feedback_parts)

            if report["supported"]:

                status = "passed"
                error = ""

            elif state.get("attempts", 0) < 2:

                status = "retry"
                error = "Evidence validation identified issues."

            else:

                status = "failed"
                error = (
                    "Answer remained unsupported or incomplete "
                    "after one regeneration attempt."
                )

            return {
                "validation_status": status,
                "validation_feedback": feedback,
                "validation_report": report,
                "error": error
            }

        except Exception as exc:

            return {
                "validation_status": "review_required",
                "validation_feedback": "",
                "validation_report": {
                    "error": str(exc)
                },
                "error": (
                    "Evidence validation could not be completed: "
                    f"{exc}"
                )
            }

    # --------------------------------------------------
    # CONDITIONAL ROUTING
    # --------------------------------------------------

    def _route_after_validation(self, state: RAGState):

        if state.get("validation_status") == "retry":
            return "retry"

        return "end"

    # --------------------------------------------------
    # LANGGRAPH WORKFLOW
    # --------------------------------------------------

    def _build_workflow(self):

        graph = StateGraph(RAGState)

        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("validate", self._validate)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "validate")

        graph.add_conditional_edges(
            "validate",
            self._route_after_validation,
            {
                "retry": "generate",
                "end": END
            }
        )

        return graph.compile()

    # --------------------------------------------------
    # PUBLIC QUERY INTERFACE
    # --------------------------------------------------

    def ask(self, question: str):

        result = self.workflow.invoke({
            "question": question,
            "rag_index": self.rag_index,
            "top_k": self.top_k
        })

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("retrieved_chunks", []),
            "validation_status": result.get(
                "validation_status",
                "unknown"
            ),
            "validation_report": result.get(
                "validation_report",
                {}
            ),
            "attempts": result.get("attempts", 0),
            "error": result.get("error", "")
        }

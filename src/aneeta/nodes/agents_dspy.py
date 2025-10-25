"""
DSPy-optimized agent implementations for ANEETAA.

This module contains DSPy versions of the original agents with:
- Structured signatures instead of manual prompts
- Automatic prompt optimization via SIMBA
- Better consistency and quality through DSPy's optimization
"""

import os
import dspy
from typing import Optional
from langchain_core.output_parsers import StrOutputParser

from ..state.models import State
from ..utils import get_last_human_message
from ..core.resources import vector_stores


# ============================================================================
# DSPy Signatures (Define input/output schemas)
# ============================================================================

class TeacherSignature(dspy.Signature):
    """Explain a NEET concept clearly and accurately based on NCERT curriculum."""
    
    context: str = dspy.InputField(
        desc="Retrieved context from NCERT textbooks and study materials"
    )
    question: str = dspy.InputField(
        desc="Student's question about Biology, Chemistry, or Physics"
    )
    language: str = dspy.InputField(
        desc="Target language for explanation (e.g., Tamil, Hindi, Bengali)"
    )
    
    response: str = dspy.OutputField(
        desc="Clear explanation in simple English, followed by translation in the target language. "
             "Do not mention 'NEET aspirant' in output. Do not repeat the English answer after translation."
    )


class MCQSolverSignature(dspy.Signature):
    """Solve a NEET MCQ with step-by-step reasoning and explanation."""
    
    question: str = dspy.InputField(
        desc="Complete multiple-choice question with options (A), (B), (C), (D)"
    )
    language: str = dspy.InputField(
        desc="Language for explanation"
    )
    
    solution: str = dspy.OutputField(
        desc="Step-by-step solution with: 1) Concept identification, 2) Reasoning steps, "
             "3) Correct answer, 4) Why other options are wrong. "
             "Provide in English then translate to target language. "
             "Do not mention 'NEET aspirant'. Do not repeat English after translation."
    )


class QuizGenerationSignature(dspy.Signature):
    """Generate a unique NEET-style MCQ question."""
    
    topic: str = dspy.InputField(desc="Topic or subject area for the question")
    context: str = dspy.InputField(desc="Reference context from past NEET papers")
    history: str = dspy.InputField(desc="Previously asked questions to avoid duplicates")
    request: str = dspy.InputField(desc="User's original request for the quiz")
    
    mcq: str = dspy.OutputField(
        desc="One unique, challenging NEET-level MCQ with exactly 4 options: (A), (B), (C), (D). "
             "Do NOT reveal the answer. Do NOT provide explanation. Just the question and options."
    )


class MentorSignature(dspy.Signature):
    """Provide NEET exam preparation guidance and motivation."""
    
    context: str = dspy.InputField(
        desc="Retrieved context from NEET topper strategies and expert advice"
    )
    question: str = dspy.InputField(
        desc="Student's question about NEET preparation, study strategy, or motivation"
    )
    language: str = dspy.InputField(desc="Language for response")
    
    guidance: str = dspy.OutputField(
        desc="Helpful guidance in English followed by translation. "
             "Be encouraging and practical. Do not repeat English after translation."
    )


# ============================================================================
# DSPy Modules (Agent implementations)
# ============================================================================

class TeacherAgentDSPy(dspy.Module):
    """DSPy-optimized Teacher Agent for NEET concept explanations."""
    
    def __init__(self):
        super().__init__()
        self.generate_explanation = dspy.ChainOfThought(TeacherSignature)
    
    def forward(self, context: str, question: str, language: str) -> dspy.Prediction:
        """Generate an explanation for a NEET concept."""
        return self.generate_explanation(
            context=context,
            question=question,
            language=language
        )


class MCQSolverAgentDSPy(dspy.Module):
    """DSPy-optimized MCQ Solver Agent."""
    
    def __init__(self):
        super().__init__()
        self.solve_mcq = dspy.ChainOfThought(MCQSolverSignature)
    
    def forward(self, question: str, language: str) -> dspy.Prediction:
        """Solve an MCQ with step-by-step explanation."""
        return self.solve_mcq(question=question, language=language)


class QuizGeneratorDSPy(dspy.Module):
    """DSPy-optimized Quiz Question Generator."""
    
    def __init__(self):
        super().__init__()
        self.generate_question = dspy.Predict(QuizGenerationSignature)
    
    def forward(self, topic: str, context: str, history: str, request: str) -> dspy.Prediction:
        """Generate a unique NEET-style MCQ."""
        return self.generate_question(
            topic=topic,
            context=context,
            history=history,
            request=request
        )


class MentorAgentDSPy(dspy.Module):
    """DSPy-optimized Mentor Agent for NEET guidance."""
    
    def __init__(self):
        super().__init__()
        self.provide_guidance = dspy.ChainOfThought(MentorSignature)
    
    def forward(self, context: str, question: str, language: str) -> dspy.Prediction:
        """Provide NEET preparation guidance."""
        return self.provide_guidance(
            context=context,
            question=question,
            language=language
        )


# ============================================================================
# Integration with ANEETAA State System
# ============================================================================

def teacher_agent_dspy(state: State):
    """
    DSPy version of teacher_agent.
    Uses optimized prompts and structured outputs.
    """
    from ..core.resources import llm
    from langchain_core.prompts import PromptTemplate
    
    subject = state['teacher_vectordb_routing']
    query = get_last_human_message(state['messages'])
    lang = state['user_explanation_language']
    vectorstore = vector_stores.get(subject)
    
    # Fallback if no vector store
    fallback_template = f"You are an expert tutor. Answer the question for a NEET Medical aspirant in simple English, then explain in {{user_explanation_language}}.\n\nQuestion:\n{{question}} DO NOT MENTION NEET ASPIRANT IN OUTPUT & Do not add any extra text or repeat the English answer after the translation and If you don't know how to translate then use english word or skip it."
    fallback_chain = PromptTemplate.from_template(fallback_template) | llm | StrOutputParser()
    
    if not vectorstore:
        return {"response_stream": fallback_chain.stream({"question": query, "user_explanation_language": lang})}
    
    # Retrieve context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)
    
    if not retrieved_docs:
        return {"response_stream": fallback_chain.stream({"question": query, "user_explanation_language": lang})}
    
    context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Use DSPy agent if available
    try:
        # Check if we have an optimized model loaded
        import streamlit as st
        if hasattr(st.session_state, 'dspy_teacher_agent') and st.session_state.dspy_teacher_agent:
            teacher = st.session_state.dspy_teacher_agent
            prediction = teacher.forward(
                context=context_str,
                question=query,
                language=lang
            )
            
            # Convert DSPy prediction to streaming format
            def stream_dspy_response():
                yield prediction.response
            
            return {"response_stream": stream_dspy_response()}
    except Exception as e:
        print(f"DSPy teacher agent failed, falling back to original: {e}")
    
    # Fallback to original RAG chain if DSPy not available
    rag_template = f"Answer based ONLY on context.\nExplain in simple English for a NEET aspirant, then explain in {{user_explanation_language}}.\n\nContext:\n{{context}}\n\nQuestion:\n{{question}} DO NOT MENTION NEET ASPIRANT IN OUTPUT Do not add any extra text or repeat the English answer after the translation."
    rag_chain = PromptTemplate.from_template(rag_template) | llm | StrOutputParser()
    return {"response_stream": rag_chain.stream({"context": context_str, "question": query, "user_explanation_language": lang})}


def mcq_question_solver_agent_dspy(state: State):
    """
    DSPy version of mcq_question_solver_agent.
    Uses optimized prompts for better step-by-step reasoning.
    """
    from ..core.resources import llm
    from langchain_core.prompts import ChatPromptTemplate
    
    user_query = get_last_human_message(state['messages'])
    lang = state['user_explanation_language']
    
    # Try to use DSPy agent if available
    try:
        import streamlit as st
        if hasattr(st.session_state, 'dspy_mcq_solver') and st.session_state.dspy_mcq_solver:
            solver = st.session_state.dspy_mcq_solver
            prediction = solver.forward(question=user_query, language=lang)
            
            def stream_dspy_response():
                yield prediction.solution
            
            return {"response_stream": stream_dspy_response()}
    except Exception as e:
        print(f"DSPy MCQ solver failed, falling back to original: {e}")
    
    # Fallback to original
    question_solver_system_template = f"You are a specialist in biology, chemistry, and physics, responsible for answering NEET entrance exam questions (MCQ format questions) and providing clear explanations in English and {{user_explanation_language}} to help NEET medical aspirants understand how the solution was reached. DO NOT MENTION NEET ASPIRANT IN OUTPUT Do not add any extra text or repeat the English answer after the translation and If you don't know how to translate then use english word or skip it."
    prompt = ChatPromptTemplate.from_messages([("system", question_solver_system_template), ("human", "{question}")])
    question_solver_chain = prompt | llm | StrOutputParser()
    return {"response_stream": question_solver_chain.stream({"question": user_query, "user_explanation_language": lang})}


def mentor_agent_dspy(state: State):
    """
    DSPy version of mentor_agent.
    Uses optimized prompts for better guidance and motivation.
    """
    from ..core.resources import llm
    from langchain_core.prompts import PromptTemplate
    
    query = get_last_human_message(state['messages'])
    lang = state['user_explanation_language']
    vectorstore = vector_stores.get('mentor')
    
    fallback_template = f"You are a helpful NEET mentor. Answer the user's question from your general knowledge as the internal knowledge base did not contain relevant information. Explain in simple English, then explain in {{user_explanation_language}}.\n\nQuestion:\n{{question}} Do not add any extra text or repeat the English answer after the translation. If you don't know how to translate then use english word or skip it."
    fallback_chain = PromptTemplate.from_template(fallback_template) | llm | StrOutputParser()
    
    if not vectorstore:
        return {"response_stream": fallback_chain.stream({"question": query, "user_explanation_language": lang})}
    
    # Retrieve context
    retriever_k3 = vectorstore.as_retriever(search_kwargs={"k": 3})
    filtered_docs = retriever_k3.invoke(query)
    
    if len(filtered_docs) < 2:
        retriever_k10 = vectorstore.as_retriever(search_kwargs={"k": 10})
        filtered_docs = retriever_k10.invoke(query)
    
    if len(filtered_docs) >= 2:
        context_str = "\n\n".join(doc.page_content for doc in filtered_docs)
        
        # Try DSPy agent
        try:
            import streamlit as st
            if hasattr(st.session_state, 'dspy_mentor_agent') and st.session_state.dspy_mentor_agent:
                mentor = st.session_state.dspy_mentor_agent
                prediction = mentor.forward(
                    context=context_str,
                    question=query,
                    language=lang
                )
                
                def stream_dspy_response():
                    yield prediction.guidance
                
                return {"response_stream": stream_dspy_response()}
        except Exception as e:
            print(f"DSPy mentor agent failed, falling back to original: {e}")
        
        # Fallback to RAG
        rag_template = f"You are a helpful NEET mentor. Answer the user's question based ONLY on the provided context. Explain in simple English, then explain in {{user_explanation_language}}.\n\nContext:\n{{context}}\n\nQuestion:\n{{question}} Do not add any extra text or repeat the English answer after the translation. If you don't know how to translate then use english word or skip it."
        rag_chain = PromptTemplate.from_template(rag_template) | llm | StrOutputParser()
        return {"response_stream": rag_chain.stream({"context": context_str, "question": query, "user_explanation_language": lang})}
    else:
        return {"response_stream": fallback_chain.stream({"question": query, "user_explanation_language": lang})}


# ============================================================================
# DSPy Agent Initialization (called from app.py)
# ============================================================================

def initialize_dspy_agents(lm_model: str = "openai/gpt-4o-mini"):
    """
    Initialize DSPy agents with the configured LM.
    
    Args:
        lm_model: DSPy-compatible model string (e.g., "openai/gpt-4o-mini")
    
    Returns:
        dict: Dictionary of initialized DSPy agents
    """
    try:
        import streamlit as st
        
        # Configure DSPy with the language model
        lm = dspy.LM(model=lm_model, max_tokens=500, temperature=0.1)
        dspy.settings.configure(lm=lm)
        
        # Initialize agents
        agents = {
            'teacher': TeacherAgentDSPy(),
            'mcq_solver': MCQSolverAgentDSPy(),
            'quiz_generator': QuizGeneratorDSPy(),
            'mentor': MentorAgentDSPy()
        }
        
        # Try to load optimized models from MLflow if available
        try:
            import mlflow
            
            # Load optimized models if they exist
            model_registry = {
                'teacher': 'teacher-agent-dspy',
                'mcq_solver': 'mcq-solver-dspy',
                'mentor': 'mentor-agent-dspy'
            }
            
            for agent_name, model_name in model_registry.items():
                try:
                    model_uri = f"models:/{model_name}/production"
                    agents[agent_name] = mlflow.dspy.load_model(model_uri)
                    print(f"✓ Loaded optimized {agent_name} from MLflow")
                except Exception:
                    # Keep the unoptimized version if MLflow model doesn't exist
                    print(f"ℹ Using unoptimized {agent_name} (no MLflow model found)")
        except Exception as e:
            print(f"ℹ MLflow not available, using unoptimized DSPy agents: {e}")
        
        # Store in session state
        st.session_state.dspy_teacher_agent = agents['teacher']
        st.session_state.dspy_mcq_solver = agents['mcq_solver']
        st.session_state.dspy_quiz_generator = agents['quiz_generator']
        st.session_state.dspy_mentor_agent = agents['mentor']
        
        print("✓ DSPy agents initialized successfully")
        return agents
        
    except Exception as e:
        print(f"✗ Failed to initialize DSPy agents: {e}")
        return None

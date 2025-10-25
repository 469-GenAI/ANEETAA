"""
MLflow monitoring and tracking utilities for ANEETAA.

This module provides:
- Experiment tracking for agent performance
- Metrics logging for each agent interaction
- Model versioning and management
- Tracing for debugging multi-agent workflows
"""

import os
import time
from functools import wraps
from typing import Dict, Any, Optional, Callable
import mlflow
from mlflow.entities import SpanType


# ============================================================================
# Configuration
# ============================================================================

def get_mlflow_config() -> Dict[str, str]:
    """Get MLflow configuration from environment variables."""
    return {
        'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
        'experiment_name': os.getenv('MLFLOW_EXPERIMENT_NAME', 'aneeta-production'),
        'enable_tracing': os.getenv('MLFLOW_ENABLE_TRACING', 'true').lower() == 'true'
    }


def setup_mlflow_tracking(experiment_name: Optional[str] = None):
    """
    Initialize MLflow tracking.
    
    Args:
        experiment_name: Name of the experiment. If None, uses env variable.
    """
    config = get_mlflow_config()
    
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(config['tracking_uri'])
        
        # Create or get experiment
        exp_name = experiment_name or config['experiment_name']
        experiment = mlflow.get_experiment_by_name(exp_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(exp_name)
            print(f"✓ Created MLflow experiment: {exp_name}")
        else:
            experiment_id = experiment.experiment_id
            print(f"✓ Using existing MLflow experiment: {exp_name}")
        
        mlflow.set_experiment(exp_name)
        
        # Enable autologging for DSPy if available
        if config['enable_tracing']:
            try:
                mlflow.dspy.autolog()
                print("✓ MLflow DSPy autologging enabled")
            except Exception as e:
                print(f"ℹ DSPy autologging not available: {e}")
        
        return experiment_id
        
    except Exception as e:
        print(f"⚠ MLflow setup failed: {e}")
        print("  Continuing without MLflow tracking...")
        return None


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_retrieval_relevance(retrieved_docs: list, query: str) -> float:
    """
    Calculate relevance score for retrieved documents.
    
    Args:
        retrieved_docs: List of retrieved documents
        query: Original query
    
    Returns:
        Relevance score between 0 and 1
    """
    if not retrieved_docs:
        return 0.0
    
    # Simple heuristic: average document score if available
    scores = []
    for doc in retrieved_docs:
        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
            scores.append(doc.metadata['score'])
        else:
            # Fallback: check if query terms appear in document
            query_terms = set(query.lower().split())
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
            scores.append(overlap)
    
    return sum(scores) / len(scores) if scores else 0.5


def calculate_response_quality(response: str) -> float:
    """
    Calculate basic quality metrics for a response.
    
    Args:
        response: Generated response text
    
    Returns:
        Quality score between 0 and 10
    """
    score = 5.0  # Base score
    
    # Length check (not too short, not too long)
    length = len(response)
    if 100 < length < 1000:
        score += 1.0
    elif length < 50:
        score -= 2.0
    
    # Coherence check (has proper sentences)
    sentences = response.split('.')
    if len(sentences) >= 3:
        score += 1.0
    
    # Structure check (has paragraphs or formatting)
    if '\n' in response:
        score += 0.5
    
    # Avoid repetition (very basic check)
    words = response.lower().split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    score += unique_ratio * 2.0
    
    return min(max(score, 0.0), 10.0)


def calculate_latency_metrics(start_time: float) -> Dict[str, float]:
    """
    Calculate latency metrics.
    
    Args:
        start_time: Start timestamp
    
    Returns:
        Dictionary with latency metrics
    """
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    return {
        'latency_ms': latency_ms,
        'latency_seconds': latency_ms / 1000,
        'is_fast': latency_ms < 2000,  # Boolean metric
        'is_acceptable': latency_ms < 5000
    }


# ============================================================================
# Decorators for Agent Tracking
# ============================================================================

def mlflow_tracked_agent(agent_name: str):
    """
    Decorator to automatically track agent executions with MLflow.
    
    Args:
        agent_name: Name of the agent (teacher, mcq_solver, etc.)
    
    Returns:
        Decorated function
    """
    def decorator(agent_func: Callable) -> Callable:
        @wraps(agent_func)
        def wrapper(state: Any) -> Any:
            config = get_mlflow_config()
            
            # Skip tracking if disabled
            if not config['enable_tracing']:
                return agent_func(state)
            
            try:
                # Extract query from state
                from ..utils import get_last_human_message
                query = get_last_human_message(state.get('messages', []))
                
                # Start MLflow span
                with mlflow.start_span(name=f"{agent_name}_execution") as span:
                    start_time = time.time()
                    
                    # Set span attributes
                    span.set_attribute("agent_name", agent_name)
                    span.set_attribute("query", query[:200])  # Truncate long queries
                    span.set_attribute("language", state.get('user_explanation_language', 'English'))
                    
                    if agent_name == 'teacher':
                        span.set_attribute("subject", state.get('teacher_vectordb_routing', 'unknown'))
                    
                    # Execute agent
                    result = agent_func(state)
                    
                    # Calculate metrics
                    latency_metrics = calculate_latency_metrics(start_time)
                    
                    # Log metrics to span
                    for metric_name, metric_value in latency_metrics.items():
                        span.set_attribute(metric_name, metric_value)
                    
                    # Mark span as successful
                    span.set_status("OK")
                    
                    return result
                    
            except Exception as e:
                # Log error and re-raise
                try:
                    span.set_status("ERROR")
                    span.set_attribute("error", str(e))
                except:
                    pass
                raise
        
        return wrapper
    return decorator


def track_agent_metrics(
    agent_name: str,
    query: str,
    response: str,
    latency_ms: float,
    retrieved_docs: Optional[list] = None,
    additional_metrics: Optional[Dict[str, Any]] = None
):
    """
    Manually track agent metrics to MLflow.
    
    Args:
        agent_name: Name of the agent
        query: User query
        response: Agent response
        latency_ms: Response latency in milliseconds
        retrieved_docs: Retrieved documents (for RAG agents)
        additional_metrics: Additional metrics to log
    """
    config = get_mlflow_config()
    if not config['enable_tracing']:
        return
    
    try:
        with mlflow.start_run(nested=True):
            # Log parameters
            mlflow.log_param("agent_name", agent_name)
            mlflow.log_param("query_length", len(query))
            mlflow.log_param("response_length", len(response))
            
            # Log metrics
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("response_quality", calculate_response_quality(response))
            
            if retrieved_docs:
                mlflow.log_metric("num_retrieved_docs", len(retrieved_docs))
                mlflow.log_metric("retrieval_relevance", calculate_retrieval_relevance(retrieved_docs, query))
            
            # Log additional metrics
            if additional_metrics:
                for key, value in additional_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                    else:
                        mlflow.log_param(key, str(value))
            
            # Log artifacts (sample query/response)
            mlflow.log_text(query, "query.txt")
            mlflow.log_text(response, "response.txt")
            
    except Exception as e:
        print(f"⚠ Failed to log metrics to MLflow: {e}")


# ============================================================================
# Model Management
# ============================================================================

def log_dspy_model(
    model: Any,
    model_name: str,
    model_type: str = "teacher",
    metrics: Optional[Dict[str, float]] = None,
    input_example: Optional[str] = None
) -> Optional[str]:
    """
    Log a DSPy model to MLflow.
    
    Args:
        model: DSPy model to log
        model_name: Name for the model
        model_type: Type of agent (teacher, mcq_solver, etc.)
        metrics: Performance metrics
        input_example: Example input for the model
    
    Returns:
        Model URI if successful, None otherwise
    """
    try:
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log model
            model_info = mlflow.dspy.log_model(
                model,
                artifact_path=model_name,
                input_example=input_example or "What is photosynthesis?"
            )
            
            # Log metrics if provided
            if metrics:
                mlflow.log_metrics(metrics)
            
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("model_name", model_name)
            
            print(f"✓ Logged DSPy model: {model_name}")
            print(f"  Model URI: {model_info.model_uri}")
            
            return model_info.model_uri
            
    except Exception as e:
        print(f"✗ Failed to log DSPy model: {e}")
        return None


def register_model_to_registry(
    model_uri: str,
    model_name: str,
    stage: str = "Production"
) -> Optional[str]:
    """
    Register a model to MLflow Model Registry.
    
    Args:
        model_uri: URI of the logged model
        model_name: Name for the registered model
        stage: Stage to transition to (None, Staging, Production, Archived)
    
    Returns:
        Registered model version if successful
    """
    try:
        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition to stage
        if stage:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
        
        print(f"✓ Registered model: {model_name} (version {model_version.version})")
        return model_version.version
        
    except Exception as e:
        print(f"✗ Failed to register model: {e}")
        return None


# ============================================================================
# Session Monitoring
# ============================================================================

class SessionMonitor:
    """Monitor a user session and log aggregated metrics."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.interactions = []
        self.agent_counts = {}
        
    def log_interaction(
        self,
        agent_name: str,
        query: str,
        response: str,
        latency_ms: float
    ):
        """Log a single interaction."""
        self.interactions.append({
            'agent': agent_name,
            'query': query,
            'response': response,
            'latency_ms': latency_ms,
            'timestamp': time.time()
        })
        
        # Update agent counts
        self.agent_counts[agent_name] = self.agent_counts.get(agent_name, 0) + 1
    
    def finalize_session(self):
        """Log aggregated session metrics to MLflow."""
        config = get_mlflow_config()
        if not config['enable_tracing']:
            return
        
        try:
            session_duration = time.time() - self.start_time
            
            with mlflow.start_run(run_name=f"session_{self.session_id}"):
                # Log session parameters
                mlflow.log_param("session_id", self.session_id)
                mlflow.log_param("num_interactions", len(self.interactions))
                mlflow.log_param("session_duration_seconds", session_duration)
                
                # Log agent usage
                for agent, count in self.agent_counts.items():
                    mlflow.log_metric(f"agent_{agent}_count", count)
                
                # Log aggregated metrics
                if self.interactions:
                    avg_latency = sum(i['latency_ms'] for i in self.interactions) / len(self.interactions)
                    mlflow.log_metric("avg_latency_ms", avg_latency)
                    mlflow.log_metric("total_queries", len(self.interactions))
                
                print(f"✓ Logged session {self.session_id} to MLflow")
                
        except Exception as e:
            print(f"⚠ Failed to log session metrics: {e}")


# ============================================================================
# Initialization
# ============================================================================

__all__ = [
    'setup_mlflow_tracking',
    'mlflow_tracked_agent',
    'track_agent_metrics',
    'log_dspy_model',
    'register_model_to_registry',
    'SessionMonitor',
    'calculate_retrieval_relevance',
    'calculate_response_quality',
    'calculate_latency_metrics'
]

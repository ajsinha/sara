"""kd.rag — RAG-specific KD modules. See submodule docstrings for details."""
from sara.rag.pipeline import RAGVectorStore, RAGPipeline, AnthropicClient, Document, RAGResponse, chunk_text, TEACHER_MODEL, STUDENT_MODEL, EMBEDDING_MODEL, DEFAULT_SYSTEM
from sara.rag.evaluation import EquivalenceReport, run_equivalence_suite
from sara.rag.migration import RAGMigration, RAGTrace, harvest_teacher_traces, evaluate_student_baseline, score_traces, partition_by_route, classify_route
from sara.rag.kd_spar import KDSPAR
from sara.rag.kd_spar_multi_teacher import MultiTeacherKDSPAR, TeacherSpec
from sara.rag.kd_spar_adversarial import AdversarialKDSPAR, AdversarialQuery
from sara.rag.kd_spar_federated import FederatedKDSPARServer, FederatedKDSPARClient, FederatedClientConfig, FederatedSimulation, FederatedRound
from sara.rag.prompt_opt import GridSearch, EvolutionaryAPO

# Ollama local backend
from sara.rag.ollama_client import (
    OllamaClient, check_ollama_running, list_available_models, ensure_model,
    pull_model, OLLAMA_TEACHER_MODEL, OLLAMA_STUDENT_MODEL,
    OLLAMA_ALT_TEACHER, OLLAMA_ALT_STUDENT, OLLAMA_DEFAULT_SYSTEM,
)
from sara.rag.ollama_pipeline import OllamaRAGPipeline
from sara.rag.ollama_kd_spar import OllamaKDSPAR, OllamaMultiTeacherKDSPAR, OllamaTeacherSpec

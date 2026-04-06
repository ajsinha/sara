# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.4.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# STORY — Eight-Part Structure  v4.0
# ═══════════════════════════════════════════════════════════════════════════
story = [NextPageTemplate('Cover')]

# ── COVER ───────────────────────────────────────────────────────────────────
story += [Spacer(1, 1.95*inch)]
story += [Paragraph('SARA  (<font name="FreeSerif">सार</font>)', Scov_title)]
story += [Spacer(1, 6)]
story += [Paragraph("Knowledge Distillation — Theory, Techniques &amp; KD-SPAR", Scov_sub)]
story += [Spacer(1, 0.35*inch)]
story += [HRFlowable(width="50%", thickness=1.5, color=GOLD, hAlign="CENTER", spaceAfter=18)]
meta = [
    [Paragraph("Author",   Scov_lbl), Paragraph("Ashutosh Sinha",                                       Scov_val)],
    [Paragraph("Email",    Scov_lbl), Paragraph("ajsinha@gmail.com",                                     Scov_val)],
    [Paragraph("Domain",   Scov_lbl), Paragraph("Machine Learning · Model Compression · Deep Learning",  Scov_val)],
    [Paragraph("Version",  Scov_lbl), Paragraph("4.0  (April 2025)",                                     Scov_val)],
    [Paragraph("Audience", Scov_lbl), Paragraph("ML Engineers · Applied Researchers · AI Architects",    Scov_val)],
]
mt = Table(meta, colWidths=[1.3*inch, 4.7*inch])
mt.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(0,-1), HexColor("#7A0E21")),
    ("BACKGROUND",    (1,0),(1,-1), HexColor("#8C1224")),
    ("TEXTCOLOR",     (0,0),(-1,-1), white),
    ("GRID",          (0,0),(-1,-1), 0.5, HexColor("#B05060")),
    ("TOPPADDING",    (0,0),(-1,-1), 8),
    ("BOTTOMPADDING", (0,0),(-1,-1), 8),
    ("LEFTPADDING",   (0,0),(-1,-1), 12),
    ("RIGHTPADDING",  (0,0),(-1,-1), 10),
    ("LINEABOVE",     (0,0),(-1,0),  1.5, GOLD),
    ("LINEBELOW",     (0,-1),(-1,-1),1.5, GOLD),
    ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
]))
story += [mt, Spacer(1, 0.3*inch)]
story += [HRFlowable(width="50%", thickness=1.5, color=GOLD, hAlign="CENTER", spaceAfter=12)]
story += [Paragraph(
    "© 2025 Ashutosh Sinha (ajsinha@gmail.com). All rights reserved. "
    "Reproduction or redistribution without written permission is prohibited.",
    Scov_copy)]
story += [NextPageTemplate('Body'), PageBreak()]

# ── ABSTRACT ────────────────────────────────────────────────────────────────
story += h1("Abstract")
story += body(
    "Knowledge distillation (KD) has evolved from a model-compression curiosity into "
    "a foundational engineering pattern for the modern AI stack. This document provides "
    "a comprehensive technical reference spanning eight parts: theoretical foundations "
    "of soft-target training; a taxonomy of five distillation families; end-to-end "
    "implementations in PyTorch, Lightning, and Hugging Face; benchmarks across "
    "vision, language, and speech; and a full treatment of KD applied to production "
    "RAG systems where the generative model is not fixed but routinely replaced.")
story += body(
    "The document's primary novel contribution is <b>KD-SPAR</b> — Knowledge "
    "Distillation via Student Prompt Auto-Rewriting — a paradigm in which the "
    "student model itself diagnoses where its outputs diverge from the teacher "
    "and proposes targeted amendments to its own system prompt. Three variants "
    "are developed in full: <b>Multi-Teacher KD-SPAR</b>, which satisfies "
    "constraints from N specialist teachers simultaneously without regressing any; "
    "<b>Adversarial KD-SPAR</b>, which mines hard examples and generates adversarial "
    "queries to build prompt robustness beyond the common-query distribution; and "
    "<b>Federated KD-SPAR</b>, which enables distributed client sites to jointly "
    "optimise a global prompt without sharing any raw query or response data — only "
    "proposed instruction strings cross site boundaries. A fifth variant, "
    "<b>MetaKDSPAR</b>, integrates metaprompting: four specialist diagnostic "
    "perspectives (citation, calibration, completeness, format) independently "
    "analyse failures, a conductor synthesises diagnoses, and each specialist "
    "proposes domain-specific fixes.")
story += body(
    "The document concludes with a rigorous ablation experiment design that "
    "directly tests the self-knowledge hypothesis: four controlled conditions "
    "(student self-proposed, teacher externally-proposed, random, and "
    "no-tuning baseline) are compared on identical evaluation queries. "
    "A complete local-model backend using <b>Ollama</b> is provided, enabling "
    "the experiment to run on a single GPU workstation (System76 OryxPro / "
    "Pop!_OS) with no API cost, no rate limits, and full reproducibility. "
    "Recommended model pairs: llama3.1:8b → llama3.2:3b (same-family controlled) "
    "and qwen2.5:7b → llama3.2:3b (cross-family generalisation test). "
    "The document concludes with an assessment of the novelty of the KD-SPAR "
    "paradigm relative to the existing literature on prompt optimisation, "
    "self-refinement, and constitutional AI.")
story += sp(8)
story += divider()

story += h2("Document Structure")
story += dtable(
    ["Part", "Sections", "Focus"],
    [
        ["I — Foundations",              "1–3b",  "Theory, taxonomy, literature survey"],
        ["II — Implementation",          "4–5",   "Code, advanced techniques"],
        ["III — Applications",           "6–7",   "Real-world cases, benchmarks"],
        ["IV — RAG & Model Migration",   "8–9",   "Pipeline, migration, prompt KD"],
        ["V — KD-SPAR Variants",         "10–13b","Base, Multi-Teacher, Adversarial, Federated, MetaKDSPAR"],
        ["VI — Practitioner Guide",      "14–17", "Hyperparams, eval, best practices, frameworks"],
        ["VII — Frontier",               "18–18e", "Future directions, Related Work, Algorithm 1, Limitations, Conclusion"],
        ["VIII — Novelty & Research",  "19–21", "Novelty assessment, ablation results, local model backend"],
    ],
    col_widths=[2.1*inch, 0.9*inch, 3.65*inch]
)
story += pgbrk()

# ══════════════════════════════════════════════════════════════════════════════
# PART I — FOUNDATIONS
# ══════════════════════════════════════════════════════════════════════════════
story += part_banner("I", "Foundations", "Theory · Soft Targets · Taxonomy")
story += h1("1. Introduction to Knowledge Distillation")
story += body(
    "Knowledge distillation (KD) is a model compression and transfer-learning technique "
    "in which a smaller, faster <b>student</b> network is trained to replicate the "
    "behaviour of a larger, more capable <b>teacher</b> network. Originally formalised "
    "by Hinton, Vinyals &amp; Dean (2015), KD has become one of the most widely deployed "
    "techniques for deploying state-of-the-art AI into resource-constrained environments.")
story += sp(6)
story += embed_diagram(diag_teacher_student(),
    "Figure 1.1 — KD flow: the teacher produces soft targets consumed alongside "
    "hard labels; Loss = α·KL(soft) + (1−α)·CE(hard).")
story += callout("Why It Matters",
    "Large foundation models require tens of gigabytes of VRAM and hundreds of watts. "
    "KD unlocks their capability inside laptops, phones, and microcontrollers without "
    "sacrificing the reasoning depth that makes them useful.")
story += h2("1.1  Historical Background")
story += body(
    "Model compression was pioneered by Bucilua et al. (2006). Hinton et al. (2015) "
    "generalised this with the temperature-scaled softmax — the canonical KD formulation "
    "used today. The field has since branched into feature-level, attention-level, "
    "relation-level, and data-free paradigms.")
story += h2("1.2  Core Intuition — Dark Knowledge")
story += body(
    "A teacher trained on hard labels implicitly encodes rich inter-class similarity — "
    "knowing that <i>cat</i> is more similar to <i>dog</i> than to <i>airplane</i>. "
    "This structure, encoded in the soft output probabilities, is Hinton's "
    "<b>dark knowledge</b>. The student learns these rather than hard labels alone.")
story += gold_callout("Analogy",
    "A master craftsperson narrating reasoning aloud as an apprentice observes — the "
    "apprentice learns not just the final product but the decision-making process "
    "embedded in every intermediate judgement.")
story += pgbrk()

story += h1("2. Theoretical Foundations")
story += h2("2.1  Temperature Scaling")
story += embed_diagram(diag_temperature(),
    "Figure 2.1 — T=1 gives a sharp distribution; T=4 softens it, revealing "
    "inter-class similarity as the dark knowledge that enriches the student's training signal.")
story += code_block([
    "def soft_probabilities(logits: torch.Tensor, T: float) -> torch.Tensor:",
    '    """T=1 → standard softmax;  T>1 → softer, richer gradient signal."""',
    "    return F.softmax(logits / T, dim=-1)",
])
story += h2("2.2  The Combined Distillation Loss")
story += code_block([
    "class DistillationLoss(nn.Module):",
    "    def __init__(self, alpha=0.5, temperature=4.0):",
    "        super().__init__()",
    "        self.alpha, self.T = alpha, temperature",
    "        self.kl = nn.KLDivLoss(reduction='batchmean')",
    "        self.ce = nn.CrossEntropyLoss()",
    "    def forward(self, s_logits, t_logits, labels):",
    "        kd = self.kl(F.log_softmax(s_logits/self.T,-1),",
    "                     F.softmax(t_logits /self.T,-1)) * self.T**2",
    "        return self.alpha*kd + (1-self.alpha)*self.ce(s_logits, labels)",
])
story += h2("2.3  Why T² Scaling?")
story += body(
    "Temperature-softened gradients are T² times smaller than their hard-label counterparts. "
    "The T² factor in the KL term restores gradient magnitude so both loss components "
    "remain on a comparable scale at any T.")
story += pgbrk()

story += h1("3. Taxonomy of Distillation Approaches")
story += embed_diagram(diag_taxonomy(),
    "Figure 3.1 — Five distillation families, their signal sources, and representative sub-techniques.")
story += h2("3.1  Response-Based")
story += body("Mimics teacher final-layer logits. Simplest; effective when teacher and student share similar depth.")
story += h2("3.2  Feature-Based (FitNets)")
story += body("Intermediate layer activation guidance. Valuable when the student is much shallower.")
story += h2("3.3  Attention Transfer")
story += body("Distils spatial attention maps from teacher layers. Improves localisation jointly with classification.")
story += h2("3.4  Relation-Based &amp; Data-Free")
story += body("Relation-based methods distil inter-sample structural relationships — architecture-agnostic. "
    "Data-free distillation uses a GAN to synthesise pseudo-samples when no original data is available.")
story += dtable(
    ["Approach","Signal","Best For","Cost"],
    [["Response-Based","Output logits","Classification, NLP","Low"],
     ["Feature-Based","Hidden layers","Shallow students","Medium"],
     ["Attention Transfer","Spatial attn","Detection, segmentation","Low–Med"],
     ["Relation-Based","Inter-sample graphs","Arch-agnostic","Medium"],
     ["Data-Free","GAN data","Privacy domains","High"],
     ["Online/Mutual","Peer predictions","Collaborative training","Medium"]],
    col_widths=[1.55*inch, 1.6*inch, 2.0*inch, 0.95*inch]
)
story += pgbrk()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3b — LITERATURE SURVEY
# ══════════════════════════════════════════════════════════════════════════════

story += h1("3b. Literature Survey")
story += body(
    "This section surveys the five research threads that KD-SPAR draws upon "
    "and positions against: knowledge distillation, prompt optimisation, "
    "self-refinement and meta-cognition, federated learning, and metaprompting.")

story += h2("Knowledge Distillation")
story += body(
    "Model compression via knowledge distillation was pioneered by Bucilua et al. (2006) [2] "
    "and formalised by Hinton, Vinyals &amp; Dean (2015) [1] with the temperature-scaled "
    "softmax that remains the standard KD formulation today. Romero et al. (2015) [3] "
    "extended KD to intermediate feature layers (FitNets), showing that hidden representations "
    "carry information lost in logit-only transfer. Zagoruyko &amp; Komodakis (2017) [5] "
    "introduced attention transfer, distilling spatial attention maps for improved "
    "localisation. Park et al. (2019) [4] proposed relational KD (RKD), operating on "
    "inter-sample structural relationships rather than individual outputs.")
story += body(
    "For NLP, Sanh et al. (2019) [6] demonstrated that DistilBERT retains 97% of BERT's "
    "performance at 40% fewer parameters. Jiao et al. (2020) [7] extended this with "
    "TinyBERT's attention-head distillation for 7.5\\u00d7 speedup. "
    "Beyer et al. (2022) [8] showed that <i>patient, consistent</i> distillation "
    "significantly outperforms one-shot approaches. More recently, Microsoft's Phi-3 [9] "
    "demonstrated GPT-4-class reasoning in a 3.8B model distilled from synthetic data. "
    "Gou et al. (2021) [10] provide a comprehensive survey of the field.")

story += h2("Prompt Optimisation")
story += body(
    "When model weights are inaccessible (API-only deployment), prompt optimisation "
    "is the primary lever for improving model behaviour. "
    "OPRO (Yang et al., 2023) [16] treats an LLM as an optimiser: the model proposes "
    "prompt candidates scored by task accuracy. APE (Zhou et al., 2023) [18] generates "
    "and scores candidate prompts automatically. EvoPrompt (Guo et al., 2023) [17] "
    "applies evolutionary search with LLM-driven mutation and crossover. "
    "DSPy (Khattab et al., 2024) [14] compiles declarative LLM pipelines into "
    "optimised prompts using a task metric as the compilation signal.")
story += body(
    "All of these methods treat the language model as a <b>black box</b> scored by "
    "task accuracy. KD-SPAR differs in three fundamental ways: "
    "(1) the optimisation target is the teacher's output distribution (KD divergence), "
    "not task accuracy; (2) the model that proposes the instruction is the same model "
    "that will execute it (self-authorship); and (3) the student first diagnoses its "
    "own failure modes before proposing a fix, rather than generating generic candidates.")

story += h2("Self-Refinement and Meta-Cognition")
story += body(
    "Self-Refine (Madaan et al., 2023) [20] lets a model iteratively improve its "
    "outputs given self-generated feedback. Constitutional AI (Bai et al., 2022) [19] "
    "uses self-critique to improve alignment by having the model evaluate its own "
    "outputs against a set of principles. Both methods improve <i>outputs</i> given a "
    "<b>fixed system prompt</b>. KD-SPAR improves the <i>instructions themselves</i> "
    "\\u2014 it operates one level of abstraction higher and is therefore more durable: "
    "a better prompt benefits all future queries, not just the one being refined.")

story += h2("Federated Learning")
story += body(
    "FedAvg (McMahan et al., 2017) [22] enables collaborative model training by sharing "
    "gradient updates. However, gradient sharing leaks training data through inversion "
    "attacks. The Federated KD-SPAR variant communicates only natural-language instruction "
    "strings \\u2014 a strictly weaker information channel than gradient vectors \\u2014 "
    "providing a stronger practical privacy guarantee for multi-site deployments.")

story += h2("Metaprompting")
story += body(
    "Metaprompting (Suzgun &amp; Kalai, 2024) orchestrates a single LLM through a "
    "conductor \\u2192 specialist architecture: a conductor prompt decomposes a task, "
    "delegates to expert personas (each with a domain-specific system prompt), and "
    "synthesises their outputs. The key insight is that one model wearing different "
    "hats can outperform the same model with a single monolithic prompt. "
    "MetaKDSPAR (Section 13b) integrates this insight into the KD-SPAR diagnostic "
    "loop: four specialist perspectives independently analyse student failures, a "
    "conductor synthesises the diagnoses, and each specialist proposes fixes from "
    "its domain. This combination of metaprompting + KD as the objective signal + "
    "student self-authorship has not been explored in the literature.")
story += pgbrk()
story += part_banner("II", "Implementation & Techniques", "Code · Pipelines · Advanced Methods")
story += h1("4. End-to-End Implementation")
story += embed_diagram(diag_training_pipeline(),
    "Figure 4.1 — Five-stage distillation pipeline: pretrain, freeze, distil, evaluate, deploy.")
story += h2("4.1  ResNet-50 → MobileNetV2 on CIFAR-10")
story += code_block([
    "teacher = models.resnet50(pretrained=True); teacher.fc = nn.Linear(2048,10); teacher.eval()",
    "student = models.mobilenet_v2(pretrained=False); student.classifier[1] = nn.Linear(1280,10)",
    "criterion = DistillationLoss(alpha=0.6, temperature=4.0)",
    "optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)",
    "for epoch in range(30):",
    "    for images,labels in loader:",
    "        with torch.no_grad(): t_logits = teacher(images.to(DEVICE))",
    "        loss = criterion(student(images.to(DEVICE)), t_logits, labels.to(DEVICE))",
    "        optimizer.zero_grad(); loss.backward(); optimizer.step()",
])
story += h2("4.2  BERT → DistilBERT")
story += code_block([
    "class BertDistillLoss(nn.Module):",
    "    def forward(self, s_logits, t_logits, s_hidden, t_hidden, labels):",
    "        kd = kl_div(log_softmax(s_logits/T,-1), softmax(t_logits/T,-1)) * T**2",
    "        hs = mse(s_hidden, t_hidden.detach())",
    "        ce = cross_entropy(s_logits, labels)",
    "        return alpha*kd + beta*hs + (1-alpha)*ce",
])
story += pgbrk()

story += h1("5. Advanced Techniques")
story += h2("5.1  Progressive / Multi-Stage")
story += body("Each stage reduces capacity by a controlled factor. The student from stage N becomes "
    "the teacher for stage N+1. Avoids the capacity-gap failure mode in one-shot large-to-tiny jumps.")
story += h2("5.2  Online / Mutual Learning")
story += code_block([
    "# Two students teach each other simultaneously",
    "for images, labels in loader:",
    "    l1, l2 = s1(images), s2(images)",
    "    loss1 = 0.5*ce(l1,labels) + 0.5*kl(log_softmax(l1,-1), softmax(l2.detach(),-1))",
    "    loss2 = 0.5*ce(l2,labels) + 0.5*kl(log_softmax(l2,-1), softmax(l1.detach(),-1))",
    "    for opt,loss in [(opt1,loss1),(opt2,loss2)]:",
    "        opt.zero_grad(); loss.backward(retain_graph=True); opt.step()",
])
story += h2("5.3  Self-Distillation")
story += body("The deepest exit acts as the teacher for shallower exits in the same network. "
    "No separate teacher model required; consistently improves generalisation.")
story += pgbrk()

# ══════════════════════════════════════════════════════════════════════════════
# PART III — APPLICATIONS
# ══════════════════════════════════════════════════════════════════════════════
story += part_banner("III", "Applications & Deployment", "Real-World Cases · Benchmarks")
story += h1("6. Real-World Applications")
story += h2("6.1  On-Device NLP — DistilBERT &amp; TinyBERT")
story += body("DistilBERT retains 97% of BERT's GLUE performance at 40% smaller and 60% faster. "
    "TinyBERT adds attention-head distillation for a 7.5× speedup.")
story += h2("6.2  Real-Time Object Detection — YOLOv8")
story += body("YOLOv8-Nano is distilled from YOLOv8-X. Soft bounding-box targets sharpen "
    "localisation accuracy at a 30× size reduction.")
story += h2("6.3  LLM Compression — Phi-3 Mini")
story += body("Microsoft distilled a 3.8B-parameter model that outperforms Llama-2-7B on reasoning, "
    "using GPT-4-generated synthetic data as both training signal and soft targets.")
story += h2("6.4  Medical Imaging — Privacy-Preserving Distillation")
story += body("A generator produces plausible X-ray patches from aggregate statistics. A small model "
    "is distilled without transmitting raw patient data — satisfying HIPAA and GDPR simultaneously.")
story += h2("6.5  Edge Speech — Whisper Compression")
story += body("Whisper-large distilled into Whisper-tiny using token-probability sequences as targets. "
    "150 MB, real-time on Raspberry Pi 4, WER increase under 3 pp.")
story += pgbrk()

story += h1("7. Benchmarks &amp; Published Results")
story += dtable(
    ["Model Pair","Task","Teacher","No KD","With KD","Compression"],
    [["ResNet-50 → MobileNetV2","CIFAR-10 Acc.","93.4%","87.2%","91.8%","4× smaller"],
     ["BERT-base → DistilBERT","GLUE avg.","84.0","77.0","81.8","2× faster"],
     ["BERT-large → TinyBERT","GLUE avg.","87.0","—","82.5","7.5× faster"],
     ["YOLOv8-X → YOLOv8-N","COCO mAP","53.9%","37.3%","41.2%","30× smaller"],
     ["GPT-4 → Phi-3-Mini","MT-Bench","8.7","—","8.1","100× smaller"],
     ["Whisper-L → Whisper-tiny","WER LibriSpeech","2.7%","10.3%","5.6%","32× smaller"]],
    col_widths=[1.9*inch, 1.2*inch, 0.78*inch, 0.72*inch, 0.82*inch, 0.78*inch]
)
story += callout("Interpretation",
    "KD consistently recovers 50–80% of the teacher accuracy gap. The 'No KD' → 'With KD' "
    "delta isolates the pure value of distillation independent of architectural improvements.")
story += pgbrk()

# ══════════════════════════════════════════════════════════════════════════════
# PART IV — RAG & MODEL MIGRATION
# ══════════════════════════════════════════════════════════════════════════════
story += part_banner("IV", "RAG & Model Migration", "Behavioural Transfer · Prompt KD Foundations")

story += h1("8. KD When Your RAG Model Changes")
story += body(
    "In a production custom RAG pipeline the generative model at the end of the retrieval "
    "chain is rarely permanent. Model transitions cause silent regressions: citation formats "
    "drift, reasoning depth drops, confidence language changes, and downstream parsers break.")
story += sp(6)
story += embed_diagram(diag_rag_migration(),
    "Figure 8.1 — Five-phase RAG model migration using KD as the behavioural "
    "transfer mechanism from departing teacher to incoming student.")
story += callout("Architecture Context",
    "In a multi-model RAG stack (LangChain / LlamaIndex, enterprise multi-cloud, "
    "query-router dispatching to GPT-4o, Claude, and open-source models), each route "
    "has its own implicit behavioural contract. KD transfers that contract to the "
    "incoming model before it ever handles live traffic.")
story += h2("8.1  Why Transitions Silently Break Pipelines")
story += blist([
    "<b>Answer format regression:</b> citation patterns change, breaking downstream parsers.",
    "<b>Retrieval-generation alignment shift:</b> new model weights retrieved passages differently.",
    "<b>Calibration drift:</b> uncertainty language changes, misfiring escalation thresholds.",
    "<b>Chain-of-thought fragmentation:</b> multi-hop reasoning paths collapse under the new model.",
])
story += h2("8.2  Harvesting Teacher Traces &amp; Routing-Aware Distillation")
story += code_block([
    "# Partition traces by query route; each route-student is distilled separately",
    "for route, route_traces in partition_by_route(all_traces, router).items():",
    "    run_sft(student_model, Dataset.from_list(route_traces),",
    "            output=f'./distilled/{route}')",
])
story += h2("8.3  Behavioural Equivalence Gates")
story += dtable(
    ["Check","Measures","Pass Threshold"],
    [["Citation Fidelity","% responses citing [Doc-N] when teacher did","> 90% of teacher"],
     ["Semantic Similarity","BERTScore F1 vs. teacher","≥ 0.85"],
     ["Calibration","Hedging-phrase frequency ratio","0.80 – 1.20"],
     ["Format Preservation","Structured response parse rate","100%"],
     ["Hallucination Proxy","1 − citation-when-cited rate","≤ 0.12"]],
    col_widths=[1.65*inch, 2.8*inch, 2.2*inch]
)
story += pgbrk()

story += h1("9. KD for Prompt Tuning — When Fine-Tuning Is Not Possible")
story += body(
    "When foundation model weights are inaccessible (API-only, proprietary, compliance), "
    "prompt tuning guided by KD is the practical alternative. The teacher's output "
    "distribution is the optimisation target; the prompt is adjusted rather than weights.")
story += embed_diagram(diag_prompt_kd_loop(),
    "Figure 9.1 — KD-guided prompt optimisation: the teacher generates gold responses; "
    "the KD loss drives iterative refinement of the student's prompt.")
story += h2("9.1  Soft Prompt Distillation")
story += embed_diagram(diag_soft_prompt(),
    "Figure 9.2 — Learnable prefix tokens p1–p5 are optimised to minimise KL "
    "divergence to teacher; all LLM weights remain frozen (PEFT PrefixTuning).")
story += code_block([
    "from peft import get_peft_model, PrefixTuningConfig, TaskType",
    "peft_cfg = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20)",
    "student  = get_peft_model(AutoModelForCausalLM.from_pretrained('mistral-7b'), peft_cfg)",
    "# trainable params: ~5M out of 7.2B  (0.07%)",
])
story += h2("9.2  Evolutionary Prompt Search &amp; DSPy Compilation")
story += embed_diagram(diag_apo(),
    "Figure 9.3 — Evolutionary APO: KD score is the fitness function; "
    "mutation and crossover evolve prompts towards teacher alignment.")
story += pgbrk()

# ══════════════════════════════════════════════════════════════════════════════
# PART V — KD-SPAR VARIANTS
# ══════════════════════════════════════════════════════════════════════════════
story += part_banner("V", "KD-SPAR Variants",
                     "Base · Multi-Teacher · Adversarial · Federated")

story += h1("10. KD-SPAR: Student-Driven Prompt Auto-Rewriting")
story += body(
    "Sections 8 and 9 treat prompt optimisation as an external operation. "
    "KD-SPAR proposes a fundamentally different paradigm: <b>the student model "
    "itself diagnoses where its outputs diverge from the teacher and writes targeted "
    "amendments to its own system prompt</b>. This self-calibrating loop requires "
    "no weight updates and no external optimiser beyond the student's own "
    "generative capability.")
story += sp(6)
story += embed_diagram(diag_spar_loop(),
    "Figure 10.1 — KD-SPAR feedback loop: the student proposes prompt rewrites based "
    "on its diagnosed failure modes and iterates until teacher-alignment converges.")
story += h2("10.1  The Core Insight: Student Self-Knowledge")
story += body(
    "The student has privileged access to its own internal representations. It knows "
    "what types of instructions elicit what kinds of responses. Standard prompt "
    "optimisers (evolutionary search, DSPy, soft prompts) do not exploit this "
    "self-knowledge. KD-SPAR does: the student is asked, in natural language, "
    "to reason about its own failure and propose a concrete instruction that "
    "would close the gap.")
story += gold_callout("Why KD-SPAR Is Different",
    "Standard APO treats the model as a black box and uses KD loss as a fitness signal. "
    "KD-SPAR treats the model as an active agent with self-knowledge. The student is not "
    "just evaluated by the KD signal — it reads the signal and writes its own fix. "
    "This makes KD-SPAR self-calibrating in a way that no external optimiser can be.")
story += h2("10.2  The Four-Phase Algorithm")
story += blist([
    "<b>Phase 1 — Diagnostic Pass:</b> run student on training queries, score against teacher "
    "traces using BERTScore (or Jaccard fallback), classify top-k failures by mode "
    "(missing_citation, over_hedged, under_hedged, incomplete, format_drift).",
    "<b>Phase 2 — Self-Interview:</b> construct a meta-prompt showing the student its "
    "own failure and the target pattern; ask it to propose one actionable instruction "
    "per failure mode. Generate N proposals per failure.",
    "<b>Phase 3 — Aggregation:</b> cluster proposals semantically (AgglomerativeClustering "
    "over MiniLM embeddings); score each cluster representative on a mini eval set; "
    "select top-K instructions by KD score improvement.",
    "<b>Phase 4 — Validate &amp; Commit:</b> compose the candidate prompt; run the full "
    "equivalence suite on held-out queries; accept only if aggregate KD score improves "
    "by threshold δ. Otherwise revert.",
])
story += pgbrk()

# ── 11. MULTI-TEACHER KD-SPAR ────────────────────────────────────────────────
story += h1("11. Multi-Teacher KD-SPAR")
story += body(
    "When migrating from a <i>committee</i> of specialist models to a single "
    "general-purpose student — for example, a citation-expert Claude, a "
    "reasoning-depth GPT-4o, and a calibration-specialist Gemini — the student "
    "must satisfy all teachers simultaneously. Multi-Teacher KD-SPAR extends "
    "the base algorithm to handle N teachers with independent KD scores.")
story += sp(6)
story += embed_diagram(diag_multi_teacher(),
    "Figure 11.1 — Multi-Teacher KD-SPAR: three specialist teachers independently "
    "score the student; the worst-aligned teacher drives the self-interview for that "
    "iteration; validation requires primary improvement AND no secondary regression.")
story += h2("11.1  The Worst-Teacher Principle")
story += body(
    "At each iteration, compute the student's KD score against every teacher. "
    "The teacher with the lowest weighted alignment score — the <b>worst teacher</b> "
    "for this iteration — provides the target for the self-interview prompt. "
    "This ensures that all teachers receive attention over successive iterations "
    "rather than the student becoming aligned with only the primary teacher.")
story += h2("11.2  Non-Regression Validation")
story += body(
    "A candidate prompt is accepted only when two conditions hold simultaneously: "
    "(a) the primary teacher's KD score improves by at least the threshold δ, and "
    "(b) no secondary teacher's score regresses by more than a tolerance τ "
    "(typically 2%). This prevents the student from over-fitting one teacher's "
    "style at the expense of another.")
story += code_block([
    "# Multi-Teacher validation gate",
    "accepted = (primary_delta >= threshold) and all(",
    "    new_scores[t.name] >= old_scores[t.name] - regression_tol",
    "    for t in self.teachers",
    ")",
])
story += h2("11.3  Configuration Example")
story += code_block([
    "from sara.rag.kd_spar_multi_teacher import MultiTeacherKDSPAR, TeacherSpec",
    "",
    "specs = [",
    "    TeacherSpec('citation',  'claude-3-5-sonnet-20241022', weight=2.0, is_primary=True),",
    "    TeacherSpec('reasoning', 'gpt-4o',                     weight=1.5),",
    "    TeacherSpec('calibrate', 'gemini-1.5-pro',             weight=1.0),",
    "]",
    "spar = MultiTeacherKDSPAR(",
    "    student_model  = 'claude-sonnet-4-5-20250929',",
    "    teachers       = specs,",
    "    vector_store   = store,",
    "    regression_tol = 0.02,",
    ")",
    "# Harvest responses from all teachers",
    "teacher_response_sets = spar.harvest_teacher_responses(query_log)",
    "# Run the multi-teacher loop",
    "final_prompt, history = spar.run(",
    "    train_queries, val_queries, teacher_response_sets,",
    "    iterations=10, threshold=0.003,",
    ")",
])
story += pgbrk()

# ── 12. ADVERSARIAL KD-SPAR ──────────────────────────────────────────────────
story += h1("12. Adversarial KD-SPAR")
story += body(
    "Standard KD-SPAR optimises on the query distribution seen in production logs. "
    "However, the long tail of hard, edge-case, and out-of-distribution queries "
    "may remain poorly handled. Adversarial KD-SPAR focuses the optimisation "
    "loop exclusively on hard examples — queries where the teacher-student KD "
    "gap is largest, or queries specifically designed by the teacher model to "
    "expose the student's weakest failure modes.")
story += sp(6)
story += embed_diagram(diag_adversarial(),
    "Figure 12.1 — Adversarial KD-SPAR: hard examples are either gap-mined from "
    "production logs (bottom percentile KD scores) or generated by the teacher. "
    "Dual-objective validation requires improvement on hard queries AND no "
    "regression on standard queries.")
story += h2("12.1  Two Sources of Hard Examples")
story += body(
    "Hard examples arrive from two complementary sources. <b>Gap-mined examples</b> "
    "are production queries where the student already exhibits the largest KD "
    "divergence — the bottom decile of KD scores across all logged queries. "
    "These are real failures with known context. "
    "<b>Adversarially generated examples</b> are produced by the teacher model "
    "using a structured prompt that requests questions requiring multi-hop reasoning, "
    "contradiction handling, edge-case awareness, and boundary testing. These "
    "anticipate failures before they appear in production.")
story += h2("12.2  Dual-Objective Validation")
story += body(
    "A naive adversarial optimiser risks over-fitting to the hard examples and "
    "regressing on common queries. Adversarial KD-SPAR prevents this with a "
    "dual threshold: the candidate prompt must improve on adversarial queries "
    "by at least δ<sub>adv</sub> AND not degrade standard queries by more than "
    "the regression tolerance τ<sub>std</sub>.")
story += code_block([
    "from sara.rag.kd_spar_adversarial import AdversarialKDSPAR",
    "",
    "spar = AdversarialKDSPAR(",
    "    teacher_model         = 'claude-3-5-sonnet-20241022',",
    "    student_model         = 'claude-sonnet-4-5-20250929',",
    "    vector_store          = store,",
    "    adversarial_topics    = ['knowledge distillation', 'RAG retrieval'],",
    "    n_generated_per_topic = 10,",
    "    hardness_percentile   = 0.25,   # bottom 25% of KD scores",
    "    dual_threshold        = 0.005,",
    "    standard_regression   = 0.02,",
    ")",
    "hard_queries = spar.build_hard_query_set(production_queries, teacher_responses)",
    "final_prompt, history = spar.run_adversarial(",
    "    adversarial_queries = hard_queries,",
    "    standard_queries    = production_queries,",
    "    teacher_responses   = teacher_responses,",
    "    iterations          = 8,",
    ")",
])
story += pgbrk()

# ── 13. FEDERATED KD-SPAR ────────────────────────────────────────────────────
story += h1("13. Federated KD-SPAR")
story += body(
    "Many real-world deployments involve multiple organisationally or geographically "
    "distributed student instances — hospital branches, regional offices, department "
    "deployments — each with private, locally held RAG data that cannot be shared. "
    "Federated KD-SPAR enables all these sites to jointly optimise a single global "
    "student prompt without any raw data leaving any site.")
story += sp(6)
story += embed_diagram(diag_federated(),
    "Figure 13.1 — Federated KD-SPAR: clients run diagnosis and self-interview locally; "
    "only proposed instruction strings (never query/response data) are sent to the "
    "aggregation server; the server validates and broadcasts an updated global prompt.")
story += h2("13.1  Privacy Guarantee")
story += body(
    "The only information that crosses the client→server boundary is a list of "
    "natural-language instruction strings proposed by each client's student model "
    "(e.g. 'Always cite retrieved passages using [Doc-N] notation'). No query text, "
    "no retrieved document content, and no model response data leave each client site. "
    "This is a stronger privacy guarantee than federated gradient sharing, which "
    "can leak training data through gradient inversion attacks.")
story += h2("13.2  Federated Round Protocol")
story += blist([
    "<b>Broadcast:</b> Server sends current global prompt P<sub>t</sub> to all clients.",
    "<b>Local Diagnosis:</b> Each client runs KD-SPAR diagnosis on its private RAG traces, "
    "generates proposed instructions, applies a local pre-filter (top-K by local mini-eval), "
    "and sends only the instruction strings to the server.",
    "<b>Aggregation:</b> Server clusters all proposals semantically, scores each cluster "
    "representative against its server-side validation queries, selects top-K.",
    "<b>Validation &amp; Broadcast:</b> Server tests the candidate P<sub>t+1</sub>, accepts "
    "if server KD score improves by δ, broadcasts P<sub>t+1</sub> (or P<sub>t</sub> if rejected).",
])
story += h2("13.3  Implementation")
story += code_block([
    "from sara.rag.kd_spar_federated import FederatedSimulation",
    "",
    "# Simulate 3 client sites from a shared trace pool",
    "sim    = FederatedSimulation(",
    "    n_clients   = 3,",
    "    all_traces  = [(q, teacher_resp) for q,teacher_resp in trace_pairs],",
    "    student_model = 'claude-sonnet-4-5-20250929',",
    "    vector_store  = store,",
    ")",
    "server = sim.build_server(threshold=0.003, regression_tol=0.02)",
    "final_prompt, history = server.run(rounds=10)",
    "",
    "# Production: deploy real distributed clients",
    "from sara.rag.kd_spar_federated import (",
    "    FederatedKDSPARClient, FederatedKDSPARServer, FederatedClientConfig",
    ")",
    "clients = [",
    "    FederatedKDSPARClient(",
    "        FederatedClientConfig(client_id='hospital_a'),",
    "        local_traces=site_a_traces, vector_store=site_a_store",
    "    ),",
    "    # ... more clients",
    "]",
    "server  = FederatedKDSPARServer(",
    "    clients=clients, server_val_queries=val_q,",
    "    server_val_responses=val_resps, student_model=STUDENT_MODEL,",
    ")",
    "final_prompt, history = server.run(rounds=10)",
])
story += h2("13.4  When to Use Each KD-SPAR Variant")
story += dtable(
    ["Variant","Trigger","Key Constraint","Privacy"],
    [["Base KD-SPAR",       "Single teacher, common queries",        "None",                         "Single site"],
     ["Multi-Teacher",      "Committee of specialist teachers",      "Non-regression across teachers","Single site"],
     ["Adversarial",        "Long-tail robustness needed",           "Dual-objective validation",    "Single site"],
     ["Federated",          "Multi-site, private local data",        "No data sharing",              "Multi-site"],
     ["<b>MetaKDSPAR</b>",  "Compound failures, richer diagnosis",  "Higher inference cost",        "Single site"]],
    col_widths=[1.5*inch, 2.0*inch, 1.9*inch, 0.9*inch]
)
story += pgbrk()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13b — META KD-SPAR (Metaprompting-Enhanced)
# ══════════════════════════════════════════════════════════════════════════════

story += h1("13b. MetaKDSPAR: Metaprompting-Enhanced Prompt Distillation")
story += body(
    "Standard KD-SPAR uses a single monolithic diagnostic pass: one call to "
    "<font name='Courier'>_classify_failure()</font> categorises the failure "
    "into one of five modes, and one self-interview prompt generates proposals. "
    "This works well for simple failures but misses <b>compound failures</b> "
    "— for example, a response that is simultaneously under-hedged <i>and</i> "
    "missing citations. MetaKDSPAR addresses this limitation using a "
    "<b>conductor + specialist</b> architecture inspired by metaprompting "
    "(Suzgun &amp; Kalai, 2024).")

story += h2("13b.1  Architecture")
story += body(
    "MetaKDSPAR decomposes the diagnostic and proposal phases into four "
    "specialist perspectives, each implemented as a distinct system prompt "
    "on the <i>same student model</i>:")
story += blist([
    "<b>Citation Specialist:</b> focuses exclusively on [Doc-N] notation, "
    "citation placement, and unsupported claims.",
    "<b>Calibration Specialist:</b> analyses hedging language — detects "
    "over-hedged and under-hedged responses relative to the teacher.",
    "<b>Completeness Specialist:</b> compares information coverage, depth, "
    "and reasoning chain length against the teacher response.",
    "<b>Format Specialist:</b> analyses structural alignment — paragraph "
    "organisation, tone, list/prose balance, and presentation style.",
])
story += body(
    "A <b>conductor</b> prompt synthesises the specialist diagnoses, ranks them "
    "by impact, and selects the top-K failures for the proposal phase. Each "
    "selected failure is then addressed by its diagnosing specialist, who "
    "proposes a targeted instruction from its domain expertise. The conductor "
    "reconciles cross-specialist proposals before the standard "
    "validate-and-commit gate.")

story += h2("13b.2  Algorithm Differences from Base KD-SPAR")
story += dtable(
    ["Phase", "Base KD-SPAR", "MetaKDSPAR"],
    [["Diagnosis",     "Single _classify_failure() call",      "K specialist prompts + conductor synthesis"],
     ["Self-Interview","One generic interview prompt",          "Specialist-perspective proposals per domain"],
     ["Aggregation",   "Score all proposals on mini-eval",      "Conductor reconciles cross-specialist proposals"],
     ["Validation",    "Same validate-and-commit gate",         "Same validate-and-commit gate"]],
    col_widths=[1.2*inch, 2.6*inch, 3.0*inch]
)
story += body(
    "The validate-and-commit gate is identical to base KD-SPAR — MetaKDSPAR "
    "changes only <i>how</i> failures are diagnosed and proposals generated, "
    "not the acceptance criterion. This ensures a fair comparison in the ablation: "
    "Condition E (MetaKDSPAR) vs. Condition A (base KD-SPAR) isolates the value "
    "of multi-perspective diagnosis.")

story += h2("13b.3  Testable Hypothesis")
story += callout("MetaKDSPAR Hypothesis",
    "Multi-perspective self-diagnosis catches compound failures that flat "
    "single-pass diagnosis misses, producing higher-quality prompt amendments. "
    "The E\\u2212A gap (MetaKDSPAR minus base KD-SPAR) quantifies this advantage. "
    "A positive E\\u2212A gap, controlling for the same student model, KD metric, "
    "and validate-and-commit gate, would demonstrate that the conductor + "
    "specialist architecture adds diagnostic value beyond what a monolithic "
    "classifier provides.")

story += h2("13b.4  Implementation")
story += code_block([
    "from sara.rag.kd_spar_meta import MetaKDSPAR",
    "",
    "meta = MetaKDSPAR(",
    "    student_model='llama3.2:3b',",
    "    vector_store=store,",
    "    # Uses 4 built-in specialists by default",
    ")",
    "",
    "final_prompt, history = meta.run(",
    "    train_queries=train_q,",
    "    val_queries=val_q,",
    "    teacher_responses=teacher_responses,",
    "    iterations=3,",
    "    top_k_diag=3,     # conductor selects top-3 diagnoses",
    "    top_k_instr=3,    # top-3 instructions per iteration",
    ")",
])

story += h2("13b.5  Trade-offs")
story += body(
    "<b>Cost:</b> Each MetaKDSPAR iteration requires K specialist diagnostic "
    "calls (K=4 by default) plus one conductor call per query, compared to "
    "one diagnostic call in base KD-SPAR. This is a 5\\u00d7 increase in "
    "Phase 1 inference cost. The proposal phase also increases because each "
    "specialist generates proposals from its perspective. "
    "With local Ollama models the cost is wall-clock time only; with API "
    "models the per-call cost multiplies accordingly.")
story += body(
    "<b>When to use MetaKDSPAR:</b> When the student's failures are consistently "
    "compound (multiple failure modes per query), when base KD-SPAR iterations "
    "plateau without improving, or when the student model is large enough "
    "(7B+) to maintain distinct specialist perspectives effectively. "
    "For small students (3B), the overhead may not be justified.")
story += pgbrk()

# ══════════════════════════════════════════════════════════════════════════════
# PART VI — PRACTITIONER GUIDE
# ══════════════════════════════════════════════════════════════════════════════
story += part_banner("VI", "Practitioner Reference",
                     "Hyperparams · Evaluation · Best Practices · Frameworks")

story += h1("14. Hyperparameter Selection Guide")
story += h2("14.1  Temperature (T) Sweep Utility")
story += code_block([
    "def temperature_sweep(teacher, student, loader, temps=[2,3,4,5,6,8], device=DEVICE):",
    "    results = {}",
    "    for T in temps:",
    "        acc = train_and_eval(student, teacher, loader, DistillationLoss(0.5,T), device, 5)",
    "        results[T] = acc; print(f'T={T}  val={acc:.4f}')",
    "    return results",
    "# Vision: T in {2-6}   |   NLP: T in {4,8,16}",
])
story += h2("14.2  Selection Table")
story += dtable(
    ["Scenario","T","Alpha (α)","Notes"],
    [["Large dataset, close capacity","2–4","0.4–0.6","Standard"],
     ["Small dataset, wide gap","4–8","0.6–0.8","Lean on teacher"],
     ["NLP / large vocabulary","8–16","0.5–0.7","Many near-zero logits"],
     ["Data-free (synthetic)","4–6","0.8–1.0","Hard labels unreliable"],
     ["Self-distillation","1–3","0.3–0.5","Same capacity"]],
    col_widths=[2.3*inch, 0.65*inch, 0.9*inch, 2.7*inch]
)
story += pgbrk()

story += h1("15. Evaluation &amp; Quality Metrics")
story += h2("15.1  Three-Number Reporting Rule")
story += body("Always report: (a) teacher on full test set, (b) student hard-labels only, "
    "(c) student with KD. The delta (b→c) isolates the pure KD contribution.")
story += h2("15.2  Efficiency Profiler")
story += code_block([
    "def profile_model(model, dummy, n=200):",
    "    model.eval()",
    "    with torch.no_grad():",
    "        for _ in range(20): model(dummy)",
    "    t0 = time.perf_counter()",
    "    with torch.no_grad():",
    "        for _ in range(n): model(dummy)",
    "    lat = (time.perf_counter()-t0)/n*1000",
    "    mem = sum(p.numel()*p.element_size() for p in model.parameters())/1e6",
    "    print(f'Latency: {lat:.2f}ms  Params: {mem:.1f}MB')",
])
story += pgbrk()

story += h1("16. Best Practices &amp; Common Pitfalls")
story += h2("16.1  Best Practices")
story += blist([
    "Freeze the teacher and wrap its forward pass in <font name='Courier'>torch.no_grad()</font>.",
    "Use cosine-annealing or warm-up LR — distillation loss is noisy in early epochs.",
    "Validate teacher quality first: a weak teacher produces misleading soft targets.",
    "Combine KD with MixUp / CutMix / RandAugment for additional regularisation.",
    "Add a projection head when teacher and student feature dims differ.",
    "Log KD loss and CE loss separately to diagnose balance issues early.",
])
story += h2("16.2  Common Pitfalls")
story += blist([
    "<b>T too high:</b> near-uniform distribution eliminates discriminative signal.",
    "<b>Alpha too high on small data:</b> student overfits teacher's idiosyncratic errors.",
    "<b>Missing T² scaling:</b> KL term becomes negligible at T &gt; 1 without it.",
    "<b>Capacity mismatch:</b> a student too small cannot absorb the information.",
    "<b>Stale batch-norm stats:</b> re-calibrate BN running means after distillation.",
])
story += pgbrk()

story += h1("17. Framework Integrations")
story += h2("17.1  PyTorch Lightning")
story += code_block([
    "class DistillModule(pl.LightningModule):",
    "    def training_step(self, batch, _):",
    "        x, y = batch",
    "        with torch.no_grad(): tl = self.teacher(x)",
    "        loss = self.criterion(self.student(x), tl, y)",
    "        self.log('loss', loss, prog_bar=True)",
    "        return loss",
    "pl.Trainer(max_epochs=30, accelerator='auto').fit(",
    "    DistillModule(teacher, student), train_loader, val_loader)",
])
story += h2("17.2  Hugging Face Trainer")
story += code_block([
    "class DistillTrainer(Trainer):",
    "    def compute_loss(self, model, inputs, return_outputs=False, **kw):",
    "        labels = inputs.pop('labels')",
    "        s = model(**inputs)",
    "        with torch.no_grad(): t = self.teacher(**inputs)",
    "        kd = kl_div(log_softmax(s.logits/self.T,-1),",
    "                    softmax(t.logits /self.T,-1),'batchmean') * self.T**2",
    "        loss = self.alpha*kd + (1-self.alpha)*cross_entropy(s.logits,labels)",
    "        return (loss, s) if return_outputs else loss",
])
story += pgbrk()

# ══════════════════════════════════════════════════════════════════════════════
# PART VII — FRONTIER
# ══════════════════════════════════════════════════════════════════════════════
story += part_banner("VII", "Frontier & Future Directions", "LLM Distillation · Multimodal · Extensions")

story += h1("18. Future Directions")
story += h2("18.1  LLM-to-LLM Distillation at Scale")
story += body("Distillation from GPT-4/Claude-class models to 7B–13B students is the primary route "
    "to capable, deployable AI. Open challenges include distribution shift, reward-model alignment "
    "during distillation, and token-level calibration across tokeniser mismatches.")
story += h2("18.2  Multimodal &amp; Quantisation-Aware Distillation")
story += body("Distilling vision-language models requires simultaneous alignment of visual encoder "
    "representations and language decoder logits. Combining with 4-bit QAT produces models that "
    "are simultaneously compressed, quantised, and behaviourally aligned.")
story += h2("18.3  KD-SPAR Extensions")
story += blist([
    "<b>Adversarial SPAR with active learning:</b> the student proposes queries it is most uncertain "
    "about, the teacher labels them, and these become adversarial examples for the next iteration.",
    "<b>Federated SPAR with differential privacy:</b> add calibrated Laplace noise to proposed "
    "instruction embeddings before sending to the server to provide (ε, δ)-differential privacy.",
    "<b>Multi-teacher SPAR with dynamic weighting:</b> dynamically re-weight teachers each iteration "
    "based on which teacher's distribution is currently closest to the student's — allocating "
    "interview effort where the gap is largest.",
    "<b>MetaKDSPAR with learned specialist selection:</b> instead of running all K specialists "
    "on every query, train a lightweight router that predicts which specialist perspectives "
    "are most likely to be informative for a given failure pattern — reducing the inference "
    "overhead from K× to ~2× while preserving diagnostic quality.",
    "<b>MetaKDSPAR with hierarchical conductors:</b> for complex multi-hop RAG tasks, a "
    "two-level conductor hierarchy could decompose the task into sub-queries, diagnose "
    "each sub-query with specialists, and then synthesise at the task level.",
    "<b>Formal evaluation benchmark:</b> a controlled benchmark comparing KD-SPAR against APO, "
    "DSPy, and soft prompts on standardised RAG workloads with known teacher-student pairs.",
])
story += pgbrk()

# ══════════════════════════════════════════════════════════════════════════════
# PART VII-B — RELATED WORK · ALGORITHM · LIMITATIONS · CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════

# ── Related Work ──────────────────────────────────────────────────────────────
story += h1("18b. Related Work")
story += body(
    "<i>A comprehensive literature survey covering all five research threads "
    "(KD, prompt optimisation, self-refinement, federated learning, metaprompting) "
    "is provided in Section 3b. This section focuses on the specific points of "
    "differentiation between KD-SPAR and prior art.</i>")
story += h2("Prompt Optimisation")
story += body(
    "<b>OPRO</b> (Yang et al., 2023) treats an LLM as an optimiser that proposes "
    "prompt candidates, evaluated by a task-accuracy score. <b>APE</b> (Zhou et al., "
    "2023) generates and scores candidate prompts automatically. <b>EvoPrompt</b> "
    "(Guo et al., 2023) applies evolutionary search with LLM-driven mutation. "
    "<b>DSPy</b> (Khattab et al., 2024) compiles declarative pipelines into "
    "optimised prompts using a task metric as signal. All these methods treat the "
    "language model as a <i>black box</i> scored by task accuracy. Sara's KD-SPAR "
    "differs in three ways: (1) the optimisation target is the teacher's output "
    "distribution (KD divergence), not task accuracy; (2) the model that proposes "
    "the instruction is the same model that will execute it (self-authorship); and "
    "(3) the student first diagnoses its own failure modes before proposing a fix.")
story += h2("Self-Refinement and Constitutional AI")
story += body(
    "<b>Self-Refine</b> (Madaan et al., 2023) lets a model iteratively improve its "
    "outputs given self-generated feedback. <b>Constitutional AI</b> (Bai et al., "
    "2022) uses self-critique to improve alignment. Both methods improve <i>outputs</i> "
    "given a fixed system prompt. KD-SPAR improves the <i>instructions</i> themselves "
    "— it operates one level of abstraction higher and is therefore more durable: "
    "a better prompt benefits all future queries, not just the one being refined.")
story += h2("Knowledge Distillation for NLP")
story += body(
    "<b>DistilBERT</b> (Sanh et al., 2019) and <b>TinyBERT</b> (Jiao et al., 2020) "
    "demonstrate KD for encoder models. <b>Beyer et al.</b> (2022) show that patient, "
    "consistent distillation significantly outperforms one-shot approaches. None of "
    "these methods apply KD to the prompt optimisation problem in API-only RAG settings "
    "where model weights are inaccessible.")
story += h2("Federated Learning")
story += body(
    "<b>FedAvg</b> (McMahan et al., 2017) enables collaborative model training "
    "by sharing gradients. Gradient-sharing leaks training data through inversion attacks. "
    "The Federated KD-SPAR variant communicates only instruction strings "
    "— a strictly weaker channel than gradient vectors — providing a stronger "
    "practical privacy guarantee for multi-site deployments.")
story += pgbrk()

# ── Algorithm 1 Pseudocode ────────────────────────────────────────────────────
story += h1("18c. Algorithm: KD-SPAR Formal Specification")
story += body(
    "The following pseudocode gives a formal specification of the base KD-SPAR "
    "algorithm. The three variants (Multi-Teacher, Adversarial, Federated) "
    "modify only the shaded phases indicated.")

_Salgo = ParagraphStyle("algo", fontName=_MONO_FONT, fontSize=8.5, leading=13,
                        textColor=CHARCOAL, backColor=HexColor("#F5F5F5"),
                        leftIndent=12, spaceAfter=2)
_Salgob = ParagraphStyle("algob", fontName=_MONO_FONT_BOLD, fontSize=8.5, leading=13,
                         textColor=CRIMSON_DARK, backColor=HexColor("#F5F5F5"),
                         leftIndent=12, spaceAfter=2)
_Salgohead = ParagraphStyle("algoh", fontName=_BODY_FONT_BOLD, fontSize=9,
                             textColor=white, backColor=CRIMSON, spaceAfter=0,
                             spaceBefore=8, leftIndent=6)

def algo_line(text, bold=False):
    style = _Salgob if bold else _Salgo
    return Paragraph(text.replace(" ","&nbsp;").replace("<","&lt;").replace(">","&gt;"), style)

algo_block = [
    Spacer(1,6),
    Paragraph("Algorithm 1 — KD-SPAR (Sara Base Variant)", _Salgohead),
]
algo_lines = [
    ("Input:  train_queries Q_train, val_queries Q_val",                False),
    ("        teacher_responses T = {q: teacher(q) for q ∈ Q_train}",   False),
    ("        base_system_prompt P₀, iterations N, threshold δ",         False),
    ("Output: optimised prompt P*",                                       False),
    ("",                                                                  False),
    ("P ← P₀",                                                           False),
    ("for t = 1 … N do:",                                                 True),
    ("  ── Phase 1: Diagnostic Pass ──────────────────────────────",      False),
    ("  for q ∈ Q_train do:",                                             False),
    ("    s_q ← student.query(q, prompt=P)",                              False),
    ("    score_q ← KD_score(s_q, T[q])",                                False),
    ("    mode_q  ← classify_failure(s_q, T[q])",                        False),
    ("  F ← top-k failures sorted by score_q ascending",                 False),
    ("",                                                                  False),
    ("  ── Phase 2: Self-Interview ────────────────────────────────",      False),
    ("  Proposals ← []",                                                  False),
    ("  for (q, s_q, T[q], mode_q) ∈ F do:",                             False),
    ("    prompt_meta ← SELF_INTERVIEW_TEMPLATE(q, s_q, T[q], mode_q)",  False),
    ("    proposals ← student.generate(prompt_meta, n=n_proposals)",     False),
    ("    Proposals ← Proposals ∪ proposals",                             False),
    ("",                                                                  False),
    ("  ── Phase 3: Aggregation ──────────────────────────────────",       False),
    ("  Clusters ← AgglomerativeCluster(embed(Proposals))",               False),
    ("  Top ← top-K cluster representatives by mini-eval KD score",      False),
    ("",                                                                  False),
    ("  ── Phase 4: Validate & Commit ─────────────────────────────",     False),
    ("  P_candidate ← P + Top",                                           False),
    ("  Δ ← mean_KD(P_candidate, Q_val, T) − mean_KD(P, Q_val, T)",     False),
    ("  if Δ ≥ δ then:  P ← P_candidate   ▷ commit",                    True),
    ("  else:            revert            ▷ reject",                     False),
    ("",                                                                  False),
    ("return P",                                                           True),
]
for text, bold in algo_lines:
    algo_block.append(algo_line(text, bold))
algo_block.append(Spacer(1,8))

story += algo_block
story += body(
    "<b>Variant modifications:</b> Multi-Teacher replaces Phase 1 with "
    "per-teacher diagnosis and adds a non-regression gate in Phase 4. "
    "Adversarial replaces Phase 1 queries with gap-mined hard examples. "
    "Federated distributes Phases 1–2 to client sites and runs Phases 3–4 "
    "on the server, sharing only Proposals (instruction strings, never query data).")
story += pgbrk()

# ── Limitations ───────────────────────────────────────────────────────────────
story += h1("18d. Limitations")
story += blist([
    "<b>Metric quality.</b> The primary evaluation metric (Jaccard token overlap, "
    "or BERTScore F1 as an upgrade) is an automatic proxy. High Jaccard does not "
    "guarantee high semantic quality. The ablation should be supplemented with "
    "human evaluation (3 raters, Cohen's κ > 0.6) before journal submission.",

    "<b>Self-interview reliability.</b> The self-interview phase assumes the student "
    "model has reliable meta-cognitive ability — that it can articulate what "
    "instructions would improve its own performance. For very weak student models "
    "(< 3B parameters), this assumption may fail and Condition B (external proposal) "
    "may outperform Condition A.",

    "<b>Single-domain evaluation.</b> All experiments use a RAG QA setting. "
    "Whether the self-knowledge advantage generalises to code generation, "
    "structured extraction, or creative writing tasks is not yet established.",

    "<b>No weight access required — but also no gradient signal.</b> KD-SPAR "
    "operates in a discrete prompt space. A continuous optimiser with weight access "
    "(soft prompt distillation, LoRA) can use gradient descent, which is more "
    "efficient per update. KD-SPAR trades gradient efficiency for API-only applicability.",

    "<b>Computational overhead.</b> Each SPAR iteration requires O(n_proposals × n_eval) "
    "additional API calls beyond the baseline. For large query sets and many proposals, "
    "this cost is non-trivial. The Ollama backend eliminates monetary cost but not "
    "wall-clock time.",

    "<b>Convergence.</b> The loop uses a fixed iteration count with a commit threshold δ. "
    "There is no formal convergence guarantee. In practice, 3–5 iterations are sufficient "
    "for typical RAG workloads, but adversarial or highly out-of-distribution query sets "
    "may require more.",
])
story += pgbrk()

# ── Conclusion ────────────────────────────────────────────────────────────────
story += h1("18e. Conclusion")
story += body(
    "This document introduced <b>Sara</b> — a comprehensive knowledge distillation "
    "toolkit whose centrepiece is <b>KD-SPAR</b> (Knowledge Distillation via Student "
    "Prompt Auto-Rewriting). KD-SPAR operationalises a simple insight: the same model "
    "that will execute a system prompt is also best placed to author it, because it "
    "has privileged self-knowledge about what instructions elicit its best-aligned "
    "responses.")
story += body(
    "Four variants were developed and fully implemented. The <b>base KD-SPAR</b> loop "
    "diagnoses failure modes, conducts a self-interview, aggregates proposals, and "
    "commits updates only when KD alignment improves on held-out validation queries. "
    "<b>Multi-Teacher KD-SPAR</b> satisfies N specialist teachers simultaneously with "
    "non-regression validation. <b>Adversarial KD-SPAR</b> hardness-mines the long "
    "tail and applies dual-objective validation to build robustness. "
    "<b>Federated KD-SPAR</b> allows distributed client sites to jointly optimise a "
    "shared global prompt without sharing any raw query or response data — only "
    "instruction strings cross site boundaries, a stronger privacy guarantee than "
    "federated gradient sharing.")
story += body(
    "The critical empirical question — whether student self-authorship adds value "
    "beyond external KD-guided proposal — is tested through the controlled four-condition "
    "ablation in Section 20. The <b>A−B gap</b> (SARA versus externally-proposed, "
    "with identical KD signal) is the metric that reviewers will focus on. "
    "Both the Anthropic API and fully local Ollama backends (llama3.1:8b → llama3.2:3b; "
    "qwen2.5:7b → llama3.2:3b) are provided so this experiment can be run without cost "
    "on a single GPU workstation.")
story += gold_callout("The name 'Sara'",
    "Sara (<font name='FreeSerif'>सार</font>, sāra) in Sanskrit means quintessence — the refined essence extracted "
    "from a larger whole. This is exactly what knowledge distillation does: it extracts "
    "the sāra of a large teacher model. KD-SPAR adds a second layer: the student "
    "finds the sāra of its own failures, and uses that distilled self-understanding "
    "to improve. The Federated variant shares only the sāra of insights — instruction "
    "strings — never the raw data from which they were derived.")
story += pgbrk()

# ══════════════════════════════════════════════════════════════════════════════
# PART VIII — NOVELTY & RESEARCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
story += part_banner("VIII", "Novelty & Research Analysis",
                     "KD-SPAR's Contributions Relative to the Literature")

story += h1("19. Novelties of KD-SPAR")
story += body(
    "This section offers an honest assessment of KD-SPAR's contributions "
    "relative to the existing literature — covering the core novelty claims, "
    "a systematic comparison to prior art, and a summary of the research "
    "positioning.")

story += h2("19.1  Novelty Assessment")
story += body(
    "KD-SPAR's core claim is that a language model has <i>privileged self-knowledge</i> "
    "about what instructions elicit its best behaviour, and that this self-knowledge can "
    "be harnessed to auto-rewrite its own system prompt using KD divergence from a "
    "teacher as the optimisation signal. Three properties distinguish KD-SPAR from "
    "all existing prompt optimisation methods:")
story += blist([
    "<b>Self-authorship:</b> the same model that will execute the prompt is the one that "
    "authors it. This creates structural self-consistency — the proposed instruction "
    "is calibrated to the model's own interpretation of language, not an external "
    "model's interpretation of what the model should do.",
    "<b>KD-grounded objective:</b> the optimisation target is a teacher's output "
    "distribution, not human preference, crowdsourced annotations, or task accuracy. "
    "This makes KD-SPAR applicable in any domain where a teacher model exists.",
    "<b>Failure-mode diagnosis:</b> the student first classifies <i>how</i> it failed "
    "(citation, calibration, format, completeness) and then proposes targeted "
    "fixes for that specific failure type — not generic improvements.",
])

story += h2("19.2  Comparison to Prior Art")
story += dtable(
    ["Method","Objective","Model Role","Self-Calibrating?","KD Signal?","API-Only?"],
    [["OPRO (Yang 2023)","Task accuracy","Black box","No","No","Yes"],
     ["DSPy (Khattab 2024)","Task metric","Black box","No","Via metric only","Yes"],
     ["APE (Zhou 2023)","Task accuracy","Generator","No","No","Yes"],
     ["EvoPrompt (Guo 2023)","Task accuracy","Mutator","No","No","Yes"],
     ["Constitutional AI","Alignment","Self-critic","Partial","No","No"],
     ["Self-Refinement","Output quality","Self-editor","Partial","No","Yes"],
     ["Soft Prompt Distil.","KD loss","Gradient","No","Yes","No (needs weights)"],
     ["<b>KD-SPAR (ours)</b>","KD loss","Self-author","<b>Yes</b>","<b>Yes</b>","<b>Yes</b>"],
     ["<b>MetaKDSPAR (ours)</b>","KD loss","Multi-specialist self-author","<b>Yes</b>","<b>Yes</b>","<b>Yes</b>"]],
    col_widths=[1.5*inch, 1.1*inch, 0.9*inch, 1.0*inch, 0.75*inch, 0.85*inch]
)
story += body(
    "The cell 'Self-Calibrating = Yes' for Constitutional AI and Self-Refinement is "
    "marked 'Partial' because these methods improve the model's <i>outputs</i> given "
    "a fixed prompt. KD-SPAR improves the model's <i>instructions</i> — the prompt "
    "itself — which is structurally different and more durable.")

story += h2("19.3  Summary Assessment")
story += dtable(
    ["Dimension","Assessment","Confidence"],
    [["Core novelty",          "Genuine — self-calibrating KD-driven prompt authorship", "High"],
     ["Prior art risk",        "Moderate — overlap with Constitutional AI, OPRO, DSPy",  "Medium"],
     ["Self-knowledge claim",  "Testable — A−B gap ablation isolates the mechanism",     "High"],
     ["Federated variant",     "Strong — novel privacy architecture (instruction-only)",  "High"],
     ["Multi-teacher variant", "Novel — non-regression gate is a new validation protocol","Medium"]],
    col_widths=[1.8*inch, 3.2*inch, 1.15*inch]
)
story += gold_callout("Author's Assessment",
    "KD-SPAR is a genuine innovation. The intellectual bet — that instruction-tuned "
    "models have useful self-knowledge about what instructions elicit their best "
    "performance — is clean, testable, and practically motivated. Even a 60% useful "
    "proposal rate, combined with robust Phase 4 validation, should produce meaningful "
    "improvements over multiple iterations. The Federated variant addresses a real "
    "and urgent enterprise problem (multi-site RAG with data sovereignty) with a "
    "principled solution. The controlled four-condition ablation in Section 20 directly "
    "tests the self-knowledge hypothesis, and the A−B gap is the single number that "
    "determines whether the core mechanism claim holds.")
story += pgbrk()

# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 20 — EXPERIMENTAL RESULTS  (placeholder — replaced by patch_paper.py)
# ══════════════════════════════════════════════════════════════════════════════

story += h1("20. Experimental Results: Self-Knowledge Hypothesis Test")
story += body(
    "<b>This section is a placeholder.</b> It is automatically replaced with "
    "real experimental results when you run the ablation experiment and execute "
    "<font name='Courier'>patch_paper.py</font>. "
    "See Section 21 for the local model setup and the README for instructions."
)
story += callout("To generate this section with real data",
    "bash setup_and_run.sh  — or manually:  "
    "python experiments/kd_spar_ablation_ollama.py --config 1 --iterations 3 --seed 42  |  "
    "python experiments/collect_results.py  |  "
    "python experiments/patch_paper.py --output docs/paper/Sara_Knowledge_Distillation.pdf")
story += pgbrk()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 21 — LOCAL MODEL BACKEND (OLLAMA)
# ══════════════════════════════════════════════════════════════════════════════

story += h1("21. Local Model Backend — Running KD-SPAR with Ollama")
story += body(
    "The Anthropic API provides access to the highest-quality teacher models, "
    "but it introduces cost, rate limits, and internet dependency. For rapid "
    "experimental iteration, ablation studies, and privacy-sensitive deployments "
    "the codebase includes a complete <b>local Ollama backend</b> that runs "
    "entirely on your own hardware. No API key, no rate limits, no cost per call.")

story += h2("21.1  Architecture")
story += body(
    "The Ollama backend is a drop-in replacement for the Anthropic pipeline. "
    "All three classes expose the same interface — "
    "<font name='Courier'>.query()</font>, "
    "<font name='Courier'>.ingest()</font>, and "
    "<font name='Courier'>.client.update_system()</font> — "
    "so the entire KD-SPAR loop runs unchanged. "
    "ChromaDB and sentence-transformers handle retrieval as before; "
    "only the generation step moves to the local Ollama server.")
story += dtable(
    ["Anthropic Class", "Ollama Equivalent", "Notes"],
    [
        ["AnthropicClient",  "OllamaClient",         "Calls localhost:11434/api/chat"],
        ["RAGPipeline",      "OllamaRAGPipeline",    "Identical interface, local generation"],
        ["KDSPAR",           "OllamaKDSPAR",         "Full 4-phase loop, local models"],
        ["MultiTeacherKDSPAR","OllamaMultiTeacherKDSPAR","Specialist prompts on same model"],
    ],
    col_widths=[1.7*inch, 1.9*inch, 3.1*inch]
)

story += h2("21.2  Recommended Model Pairs")
story += body(
    "Three configurations are provided, covering same-family controlled comparison "
    "and cross-family generalisation tests:")
story += dtable(
    ["Config", "Teacher", "Student", "Disk", "Use Case"],
    [
        ["1  (recommended)", "llama3.1:8b", "llama3.2:3b",
         "6.7 GB", "Same family — controlled capacity gap test"],
        ["2  (cross-family)", "qwen2.5:7b",  "llama3.2:3b",
         "6.4 GB", "Cross-family — tests generalisation of KD signal"],
        ["3  (Qwen student)", "llama3.1:8b", "qwen2.5:3b",
         "7.0 GB", "Different student arch — architecture-agnostic claim"],
    ],
    col_widths=[1.3*inch, 1.2*inch, 1.1*inch, 0.85*inch, 2.25*inch]
)
story += body(
    "Config 1 is the recommended starting point: it controls for model family, "
    "the 8B→3B capacity ratio matches common production compression targets, "
    "and both models are among the best-performing open-weight options at their "
    "size. Config 2 is critical for publication — a cross-family A−B gap "
    "shows the mechanism generalises beyond a single model provider's "
    "training data and tokenisation.")

story += h2("21.3  Local Experimentation Setup — Complete Shell Script")
story += body(
    "The following is a complete, copy-paste-ready shell script for setting up "
    "Sara on a System76 OryxPro running Pop!_OS (Ubuntu-based). Run each block "
    "in order. The RTX 3070 Ti accelerates Ollama inference to ~30–50 tokens/sec "
    "for 7–8B models, making a full 3-iteration ablation complete in 12–18 minutes. "
    "GPU is optional — CPU-only runs take 45–90 minutes but produce identical results.")
story += code_block([
    "#!/usr/bin/env bash",
    "# Sara (सार) — OryxPro Setup & Ablation Script",
    "# Save as: setup_and_run.sh",
    "# Run with: bash setup_and_run.sh",
    "set -e   # exit on first error",
    "",
    "# ── STEP 1: Install Ollama (one-time) ───────────────────────────────────",
    "if ! command -v ollama &> /dev/null; then",
    "    curl -fsSL https://ollama.com/install.sh | sh",
    "    echo 'Ollama installed.'",
    "else",
    "    echo 'Ollama already installed, skipping.'",
    "fi",
    "",
    "# Start Ollama server in background (auto-starts on Pop!_OS via systemd,",
    "# but start manually if not running)",
    "ollama serve &> /tmp/ollama.log &",
    "sleep 3   # wait for server to be ready",
    "",
    "# ── STEP 2: Pull models (one-time, uses disk not GPU) ───────────────────",
    "ollama pull llama3.1:8b    # 4.7 GB — recommended teacher  (Config 1 & 3)",
    "ollama pull llama3.2:3b    # 2.0 GB — recommended student  (Config 1 & 2)",
    "ollama pull qwen2.5:7b     # 4.4 GB — cross-family teacher (Config 2)",
    "# ollama pull qwen2.5:3b  # 2.1 GB — optional Qwen student (Config 3)",
    "",
    "# ── STEP 3: Project setup ───────────────────────────────────────────────",
    "cd ~/PycharmProjects/sara   # adjust path if different",
    "",
    "# Create virtual environment (skip if already exists)",
    "if [ ! -d '.venv' ]; then",
    "    python3 -m venv .venv",
    "fi",
    "source .venv/bin/activate",
    "",
    "# Install Sara RAG dependencies only (no PyTorch needed for ablation)",
    "pip install -e '.[rag]'",
    "",
    "# ── STEP 4: Sanity check ────────────────────────────────────────────────",
    "python examples/08_ollama_kd_spar.py --teacher llama3.1:8b --student llama3.2:3b",
    "",
    "# ── STEP 5: Run the ablation (publication run: 5 seeds x 2 configs) ─────",
    "# Config 1: llama3.1:8b (teacher) -> llama3.2:3b (student) — same family",
    "# Config 2: qwen2.5:7b  (teacher) -> llama3.2:3b (student) — cross family",
    "for cfg in 1 2; do",
    "    for seed in 42 123 456 789 101; do",
    "        echo \"Running config=$cfg seed=$seed ...\"",
    "        python experiments/kd_spar_ablation_ollama.py \\",
    "            --config $cfg --iterations 3 --seed $seed",
    "    done",
    "done",
    "",
    "# ── STEP 6: Aggregate results ───────────────────────────────────────────",
    "python experiments/collect_results.py",
    "",
    "# ── STEP 7: Patch paper with real numbers ───────────────────────────────",
    "# Requires reportlab + pypdf on this machine to rebuild the PDF:",
    "# pip install reportlab pypdf",
    "# python experiments/patch_paper.py --output ~/Desktop/sara_paper.pdf",
    "",
    "# ── STEP 8: View results summary ────────────────────────────────────────",
    "python experiments/results_analysis.py",
    "",
    "echo '================================================='",
    "echo 'Sara ablation complete. Key metric: A-B gap.'",
    "echo 'Results in: experiments/results/'",
    "echo '================================================='",
])
story += body(
    "<b>Quick run</b> (single config, 3 seeds, ~20 min) for a first result "
    "before committing to the full run:")
story += code_block([
    "source .venv/bin/activate",
    "for seed in 42 123 456; do",
    "    python experiments/kd_spar_ablation_ollama.py --config 1 --iterations 3 --seed $seed",
    "done",
    "python experiments/collect_results.py",
    "python experiments/results_analysis.py",
])
story += body(
    "<b>Note on pyproject.toml:</b> The project uses an empty base "
    "<font name='FreeMono'>dependencies = []</font> list so that "
    "<font name='FreeMono'>pip install -e '.[rag]'</font> installs only "
    "the four RAG packages (anthropic, chromadb, sentence-transformers, requests) "
    "without downloading PyTorch. The <font name='FreeMono'>[vision]</font> and "
    "<font name='FreeMono'>[nlp]</font> extras add PyTorch when needed.")

story += h2("21.4  Why Local Models Matter for the Experimentation")
story += body(
    "Running the ablation on local open-weight models provides three "
    "advantages over API-only evaluation:")
story += blist([
    "<b>Reproducibility:</b> Fixed model weights (same Ollama quantisation) "
    "plus temperature=0 gives identical outputs across runs. Reviewers can "
    "reproduce your exact numbers by pulling the same model tags.",
    "<b>Multi-architecture evidence:</b> Showing the A−B gap across llama3.1:8b→llama3.2:3b "
    "(same family) <i>and</i> qwen2.5:7b→llama3.2:3b (cross-family) provides much "
    "stronger evidence that the self-knowledge mechanism is general rather than "
    "specific to one model provider.",
    "<b>Free iteration:</b> Running 20 ablation seeds costs nothing and takes "
    "a few hours overnight. This lets you report tighter confidence intervals "
    "and vary the number of SPAR iterations as a hyperparameter sweep.",
])
story += gold_callout("Further Experimentation Recommendations",
    "Run the ablation on both the Anthropic API pair "
    "(claude-3-5-sonnet teacher → claude-sonnet-4-5 student) "
    "and the Ollama Config 1 pair (llama3.1:8b → llama3.2:3b) "
    "with 3 seeds each. If the A−B gap is positive and consistent across "
    "both API and open-weight settings, the claim is robust to model family. "
    "That cross-model consistency is the difference between "
    "a short paper and a full paper.")
story += pgbrk()

# ── REFERENCES ───────────────────────────────────────────────────────────────
story += h1("References")
refs = [
    ("1",  "Hinton, G., Vinyals, O., &amp; Dean, J. (2015). <i>Distilling the Knowledge in a Neural Network.</i> NIPS Deep Learning Workshop. arXiv:1503.02531."),
    ("2",  "Bucilua, C., Caruana, R., &amp; Niculescu-Mizil, A. (2006). <i>Model Compression.</i> ACM SIGKDD. doi:10.1145/1150402.1150464."),
    ("3",  "Romero, A. et al. (2015). <i>FitNets: Hints for Thin Deep Nets.</i> ICLR. arXiv:1412.6550."),
    ("4",  "Park, W. et al. (2019). <i>Relational Knowledge Distillation.</i> CVPR. arXiv:1904.05068."),
    ("5",  "Zagoruyko, S. &amp; Komodakis, N. (2017). <i>Paying More Attention to Attention.</i> ICLR. arXiv:1612.03928."),
    ("6",  "Sanh, V. et al. (2019). <i>DistilBERT, a distilled version of BERT.</i> arXiv:1910.01108."),
    ("7",  "Jiao, X. et al. (2020). <i>TinyBERT: Distilling BERT for NLU.</i> EMNLP. arXiv:1909.10351."),
    ("8",  "Beyer, L. et al. (2022). <i>Knowledge Distillation: A Good Teacher is Patient and Consistent.</i> CVPR. arXiv:2106.05237."),
    ("9",  "Microsoft Research (2024). <i>Phi-3 Technical Report.</i> arXiv:2404.14219."),
    ("10", "Gou, J. et al. (2021). <i>Knowledge Distillation: A Survey.</i> IJCV 129(6):1789–1819."),
    ("11", "Radford, A. et al. (2023). <i>Robust Speech Recognition via Large-Scale Weak Supervision.</i> ICML. arXiv:2212.04356."),
    ("12", "Li, X.L. &amp; Liang, P. (2021). <i>Prefix-Tuning: Optimizing Continuous Prompts for Generation.</i> ACL. arXiv:2101.00190."),
    ("13", "Lester, B. et al. (2021). <i>The Power of Scale for Parameter-Efficient Prompt Tuning.</i> EMNLP. arXiv:2104.08691."),
    ("14", "Khattab, O. et al. (2024). <i>DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines.</i> ICLR. arXiv:2310.03714."),
    ("15", "Lewis, P. et al. (2020). <i>Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.</i> NeurIPS. arXiv:2005.11401."),
    ("16", "Yang, C. et al. (2023). <i>Large Language Models as Optimizers (OPRO).</i> arXiv:2309.03409."),
    ("17", "Guo, Q. et al. (2023). <i>Connecting Large Language Models with Evolutionary Algorithms (EvoPrompting).</i> arXiv:2309.08532."),
    ("18", "Zhou, Y. et al. (2023). <i>Large Language Models are Human-Level Prompt Engineers (APE).</i> ICLR. arXiv:2211.01910."),
    ("19", "Bai, Y. et al. (2022). <i>Constitutional AI: Harmlessness from AI Feedback.</i> Anthropic. arXiv:2212.08073."),
    ("20", "Madaan, A. et al. (2023). <i>Self-Refine: Iterative Refinement with Self-Feedback.</i> NeurIPS. arXiv:2303.17651."),
    ("21", "Hu, E.J. et al. (2021). <i>LoRA: Low-Rank Adaptation of Large Language Models.</i> ICLR. arXiv:2106.09685."),
    ("22", "McMahan, H.B. et al. (2017). <i>Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg).</i> AISTATS. arXiv:1602.05629."),
    ("23", "Suzgun, M. &amp; Kalai, A.T. (2024). <i>Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding.</i> arXiv:2401.12954."),
]
Sref2 = ParagraphStyle("sref3", fontName=_BODY_FONT, fontSize=9.5, leading=14,
                        textColor=CHARCOAL, leftIndent=22, firstLineIndent=-22, spaceAfter=5)
for num, text in refs:
    story.append(Paragraph(f"[{num}]  {text}", Sref2))
story += pgbrk()

# ── GLOSSARY ─────────────────────────────────────────────────────────────────
story += h1("Glossary")
story += dtable(
    ["Term","Definition"],
    [
        ["Sara (<font name=\"FreeSerif\">सार</font>)",  "Sanskrit: quintessence, refined essence extracted from a larger whole. The project name."],
        ["Dark Knowledge",         "Inter-class similarity encoded in teacher soft output distributions."],
        ["Temperature (T)",        "Scalar that flattens the softmax, revealing inter-class structure."],
        ["Soft Targets",           "Teacher output probabilities used as the primary student training signal."],
        ["Hard Labels",            "One-hot ground-truth labels from the original labelled dataset."],
        ["Alpha (α)",              "Mixing coefficient: weight of distillation loss vs. cross-entropy."],
        ["Feature Distillation",   "Matching intermediate hidden-layer activations teacher→student."],
        ["Attention Transfer",     "Distilling spatial attention maps from CNN or transformer layers."],
        ["Data-Free KD",           "Distillation using GAN-synthesised pseudo-data; no original data."],
        ["Soft Prompt",            "Learnable continuous token embeddings prepended to inputs; weights frozen."],
        ["KD-SPAR",                "KD via Student Prompt Auto-Rewriting — student authors its own prompt fixes."],
        ["Multi-Teacher KD-SPAR",  "KD-SPAR variant aligning student to N teachers simultaneously without regression."],
        ["Adversarial KD-SPAR",    "KD-SPAR variant mining hard examples with dual-objective validation."],
        ["Federated KD-SPAR",      "KD-SPAR variant where clients share only instruction strings, not data."],
        ["Worst-Teacher Principle","In Multi-Teacher SPAR: target the lowest weighted KD score teacher each iteration."],
        ["Dual-Objective Validation","In Adversarial SPAR: candidate must improve hard queries AND not regress on standard."],
        ["Gap-Mined Hard Examples","Production queries in the bottom decile of KD scores; real failures already occurring."],
        ["Non-Regression Gate",    "Validation rule: new_score ≥ old_score − tolerance for all secondary teachers."],
        ["H₀ (null hypothesis)",   "E[KD(A)] ≤ E[KD(B)]: self-authorship adds no value beyond the KD signal alone."],
        ["H₁ (alternative)",       "E[KD(A)] > E[KD(B)]: student self-authorship produces higher alignment."],
        ["A−B Gap",                "KD(A) − KD(B): pure value of self-authorship. >0.02 = strong evidence for H₁."],
        ["Condition A",            "SARA self-proposed: student diagnoses and authors its own prompt fixes."],
        ["Condition B",            "Externally proposed: teacher model proposes with the same KD signal."],
        ["Condition C",            "Random instruction baseline: generic pool, no KD signal or diagnosis."],
        ["Condition D",            "No-tuning baseline: student runs on the vanilla default system prompt."],
        ["KL Divergence",          "Information-theoretic distance between two probability distributions."],
        ["BERTScore",              "Semantic similarity metric using contextual BERT embeddings (F1 of alignment)."],
        ["Jaccard Similarity",     "Token-overlap ratio; primary KD scoring proxy in this work."],
        ["DSPy",                   "Declarative Self-improving Language Programs — LLM pipeline compiler."],
        ["OPRO",                   "Optimise by PROmpting — LLM-as-optimiser for task-accuracy objectives."],
        ["QAT",                    "Quantisation-Aware Training — simulates quantisation during training."],
        ["Self-Knowledge",         "The model's implicit knowledge of what instructions elicit its best responses."],
        ["Ollama",                 "Open-source local LLM runtime; serves models via REST API at localhost:11434."],
        ["OllamaKDSPAR",           "SARA variant using local Ollama models — zero API cost, fully offline."],
        ["Llama 3.1 8B",           "Meta's 8B instruction-tuned model; recommended local SARA teacher."],
        ["Llama 3.2 3B",           "Meta's 3B model; recommended local SARA student (Config 1)."],
        ["Qwen 2.5 7B",            "Alibaba's 7B model; cross-family teacher for Config 2 ablation."],
        ["OryxPro",                "System76 OryxPro laptop (Pop!_OS); primary hardware for local experiments."],
        ["MetaKDSPAR",             "KD-SPAR variant using conductor + specialist metaprompting architecture."],
        ["Metaprompting",          "Orchestrating one LLM through multiple specialist personas via a conductor prompt."],
        ["Conductor",              "In MetaKDSPAR: the prompt that synthesises specialist diagnoses and reconciles proposals."],
        ["Specialist",             "In MetaKDSPAR: a domain-specific diagnostic perspective (citation, calibration, etc.)."],
        ["E\\u2212A Gap",            "KD(E) \\u2212 KD(A): value added by multi-perspective diagnosis over flat diagnosis."],
        ["Condition E",            "MetaKDSPAR: conductor + specialist self-diagnosis and proposal."],
    ],
    col_widths=[1.9*inch, 4.8*inch]
)

# ── COLOPHON ─────────────────────────────────────────────────────────────────
story += divider()
story += [Paragraph(
    "Typeset with ReportLab Platypus. KD-SPAR contributions by Ashutosh Sinha. "
    "<b>© 2025 Ashutosh Sinha (ajsinha@gmail.com). All rights reserved.</b>",
    ParagraphStyle("col2", fontName="Helvetica-Oblique", fontSize=8, leading=12,
                   textColor=GRAY_MED, alignment=TA_CENTER, spaceBefore=6)
)]

# ══════════════════════════════════════════════════════════════════════════════
build_doc(story, str(Path(__file__).resolve().parent / "Sara_Knowledge_Distillation.pdf"))

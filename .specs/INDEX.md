# .specs/ Directory Index

This directory contains all project specifications, plans, reviews, and documentation for the Private AI Inference Server.

---

## Quick Navigation

### For Users

- 📖 [README.md](../README.md) - Start here for usage and API reference
- 🗺️ [ROADMAP.md](ROADMAP.md) - Feature roadmap and timeline (v1.0 → v3.0)

### For Developers

- 🗺️ [ROADMAP.md](ROADMAP.md) - Current status + what's next
- 🔍 [Server Review](SERVER-REVIEW.md) - Architecture analysis and recommendations
- 📝 [Coder Guidelines](../.github/instructions/coder-guidelines.instructions.md) - Coding standards
- 🤖 [Copilot Instructions](../.github/copilot-instructions.md) - AI agent onboarding

### For Project Management

- 📊 [Documentation Summary](DOCUMENTATION-SUMMARY.md) - Overview of all deliverables
- 🗺️ [ROADMAP.md](ROADMAP.md) - Timeline and priorities

---

## File Descriptions

### Top-Level Documents

#### [ROADMAP.md](ROADMAP.md)

**Purpose**: Complete feature roadmap from v1.0 (production-ready) to v3.0  
**Size**: ~450 lines  
**Contents**:

- ✅ Phase 1: Stability & Performance (v1.0) — COMPLETED
- ✅ Sibling orchestration + desktop app + pull UX (v1.1) — COMPLETED
- ✅ Release installer with guided setup (v1.1.0) — COMPLETED
- 🚧 Phase 2: Advanced Features (v1.5) — NEXT UP
- Phase 3: Production Hardening (v2.0)
- Phase 4: Ecosystem Integration (v2.5)
- Phase 5: Advanced Optimizations (v3.0)
- Deprecation & migration plan
- Success metrics
- Changelog

**When to read**: Planning new features, understanding priorities, setting timelines, deciding what's next

---

#### [SERVER-REVIEW.md](SERVER-REVIEW.md)

**Purpose**: Comprehensive analysis of current implementation  
**Size**: ~550 lines  
**Contents**:

- Architecture review (strengths, weaknesses)
- Code quality analysis
- Technical debt inventory
- Performance analysis and benchmarks
- Security review
- Reliability analysis
- Scalability assessment
- Dependencies review
- Testing coverage estimate
- Documentation quality
- Comparison to alternatives (Ollama, vLLM)
- Recommendations

**When to read**: Understanding current state, making architectural decisions, identifying improvements

---

#### [DOCUMENTATION-SUMMARY.md](DOCUMENTATION-SUMMARY.md)

**Purpose**: Overview of all documentation deliverables  
**Size**: ~350 lines  
**Contents**:

- Summary of all 5 deliverables
- Key insights documented (MoE quantization, memory management, etc.)
- Metrics & success criteria
- Next steps
- Benefits realized
- Document quality ratings

**When to read**: Getting overview of available documentation, understanding what's been completed

---

### plans/ Directory

#### [plans/background-pull-plan.md](plans/background-pull-plan.md)

**Purpose**: Original implementation plan for non-blocking downloads  
**Size**: ~150 lines  
**Status**: ✅ Implemented (v0.9)  
**Contents**:

- Goal and approach
- Detailed implementation steps
- Acceptance criteria
- Milestones and timeline
- Risks and mitigations

**When to read**: Understanding the background job system design

---

#### [plans/phase1-implementation.md](plans/phase1-implementation.md)

**Purpose**: Detailed step-by-step implementation guide for v1.0  
**Size**: ~650 lines  
**Status**: ✅ Implemented (v1.0)  
**Contents**:

- Task 1.3: Health Check Endpoint
- Task 1.1: Streaming Support
- Task 1.4: Request Timeout & Cancellation
- Task 1.2: Progress Tracking for Downloads
- Task 1.5: Persistent Job Storage (SQLite)
- Task 1.6: Improved Error Messages
- Testing & validation
- Integration test suite
- Performance benchmarks
- Rollout plan (3 weeks)

**When to read**: Understanding v1.0 feature implementation details

---

#### [plans/model-classification-plan.md](plans/model-classification-plan.md)

**Purpose**: Plan for classifying models by modality (chat / vision / imagegen / voice / embeddings)  
**Status**: ✅ Implemented (v1.1, Level 1)  
**When to read**: Understanding modality classification design

---

#### [plans/sibling-orchestration-level2-3-plan.md](plans/sibling-orchestration-level2-3-plan.md)

**Purpose**: Plan for proxying image/voice to sibling servers (Open Fantasia, Olly Voice) + setup wizard  
**Status**: ✅ Implemented (v1.1, Level 2 & 3)  
**When to read**: Understanding sibling-server orchestration

---

#### [plans/release-installer-guided-setup-plan.md](plans/release-installer-guided-setup-plan.md)

**Purpose**: Plan for the native Windows installer with 8-screen guided setup (Tauri + NSIS)  
**Status**: ✅ Implemented (v1.1.0)  
**When to read**: Understanding the installer / desktop app / server bundling

---

#### [plans/muse-glimmer-support-plan.md](plans/muse-glimmer-support-plan.md)

**Purpose**: Minimal-scope plan to support Muse Glimmer using the existing Transformers + HF snapshot workflow (no GGUF backend additions)  
**Status**: 🔄 Proposed  
**Contents**:

- Current app.py audit of pull/load/multimodal paths
- Decision to use `meta-models/Muse-Glimmer-30B` under current architecture
- Minimal implementation steps for multimodal class alignment
- Local snapshot source-of-truth guidance (`model_meta.local_path`)
- Explicit GGUF non-support handling in current server path
- Smoke test plan for `/pull` and `/chat/multimodal`

**When to read**: Adding Muse support while preserving current architecture and avoiding extra backend scope

---

#### [plans/gguf-support-plan.md](plans/gguf-support-plan.md)

**Purpose**: Phased plan to add GGUF text-inference support as an additional backend while preserving current Transformers/vLLM architecture and API contracts  
**Status**: 🔄 Draft  
**Contents**:

- Backend extension design (`gguf_llama_cpp`)
- Pull/chat routing strategy for `.gguf` and `hf://.../*.gguf`
- No-break API compatibility approach
- Environment and runtime controls for low-resource GPUs
- Phased rollout, acceptance criteria, and risk mitigation

**When to read**: Planning low-resource GGUF enablement without replacing existing model paths

---

#### [plans/PERFORMANCE_FIXES.md](plans/PERFORMANCE_FIXES.md)

**Purpose**: Performance fixes and tuning notes  
**Status**: Reference  
**When to read**: Investigating performance issues

---

### docs/ Directory

#### [docs/database-schema.md](docs/database-schema.md)

**Purpose**: SQLite job persistence schema  
**Status**: Reference  
**When to read**: Understanding the jobs.db schema

---

## Document Relationships

```
User Journey:
README.md → ROADMAP.md → plans/release-installer-guided-setup-plan.md (done)
                       → Phase 2 (v1.5) features

Developer Onboarding:
.github/copilot-instructions.md → .github/instructions/coder-guidelines.instructions.md → SERVER-REVIEW.md

Planning Flow:
ROADMAP.md → plans/feature-plan.md → Implementation

Review Cycle:
Implementation → SERVER-REVIEW.md → ROADMAP.md updates
```

---

## Version History

### v1.0 (Production-Ready) — COMPLETED

- Streaming support
- Health checks
- Request timeouts
- Progress tracking
- Persistent jobs
- Improved errors

### v1.1 (Local Inference Orchestrator) — COMPLETED

- Model modality classification (Level 1)
- Sibling-server orchestration (Level 2 & 3)
- Tauri desktop app wrapper
- Searchable Hub model picker + single-file GGUF pull
- **Release installer with guided setup** (Tauri + NSIS)
- Single-command release packaging
- Chat UI redesign + case-study screenshots

### v1.5 (Phase 2 — Advanced Features) — NEXT UP

See [ROADMAP.md](ROADMAP.md) "Phase 2" section.

See [ROADMAP.md](ROADMAP.md) for complete version history and future plans.

---

## How to Use This Directory

### When Starting New Feature

1. Check [ROADMAP.md](ROADMAP.md) for priority and timeline
2. Read relevant task in [plans/phase1-implementation.md](plans/phase1-implementation.md) (if Phase 1)
3. Review [coder-guidelines](../.github/instructions/coder-guidelines.instructions.md) for patterns
4. Implement following examples and acceptance criteria
5. Run tests from implementation plan
6. Update SERVER-REVIEW.md if architectural changes made

### When Reviewing Code

1. Check [coder-guidelines](../.github/instructions/coder-guidelines.instructions.md) for standards
2. Verify patterns match [copilot-instructions](../.github/copilot-instructions.md)
3. Cross-reference with [SERVER-REVIEW.md](SERVER-REVIEW.md) recommendations
4. Ensure tests exist (see implementation plans)

### When Planning Next Phase

1. Review current phase completion in ROADMAP.md
2. Read SERVER-REVIEW.md recommendations
3. Prioritize based on user feedback and metrics
4. Create new plan in `plans/phaseN-implementation.md`
5. Update ROADMAP.md with any changes

### When Onboarding New Developer

**Read in this order**:

1. [README.md](../README.md) - Understand what the project does
2. [.github/copilot-instructions.md](../.github/copilot-instructions.md) - Architecture overview
3. [.github/instructions/coder-guidelines.instructions.md](../.github/instructions/coder-guidelines.instructions.md) - Coding standards
4. [SERVER-REVIEW.md](SERVER-REVIEW.md) - Current state and tech debt
5. [plans/phase1-implementation.md](plans/phase1-implementation.md) - Next tasks

---

## Maintenance Guidelines

### Keep Updated

- **ROADMAP.md**: Update when features complete, priorities change, or new phases planned
- **SERVER-REVIEW.md**: Update quarterly or after major changes
- **Implementation plans**: Mark tasks complete, update acceptance criteria
- **DOCUMENTATION-SUMMARY.md**: Update when new docs added

### Archive Policy

- Completed plans: Move to `plans/archive/` with completion date in filename
- Old reviews: Keep in `reviews/archive/` for historical reference
- Deprecated specs: Add "DEPRECATED" prefix and link to replacement

### Review Cycle

- **Monthly**: Review Phase 1 implementation progress
- **Quarterly**: Update SERVER-REVIEW.md with new findings
- **Per Release**: Update ROADMAP.md with completed features and new priorities
- **Ad-hoc**: Update docs when architectural decisions made

---

## Contributing to Documentation

### When Adding New Specs

1. Create file in appropriate directory (`plans/`, `reviews/`, etc.)
2. Update this INDEX.md with link and description
3. Cross-reference from relevant existing docs
4. Add to navigation section at top

### Documentation Standards

- Use Markdown (.md) format
- Include table of contents for files >100 lines
- Use code fences with language hints
- Include "When to read" section
- Date all documents
- Link to related documents

### File Naming

- Use kebab-case: `phase1-implementation.md`
- Include version for phase plans: `phase1-implementation.md`, `phase2-implementation.md`
- Use descriptive names: `streaming-support-spec.md` not `feature1.md`
- Archive with dates: `phase1-implementation-completed-2026-01-15.md`

---

## Quick Reference

### File Sizes

- ROADMAP.md: ~450 lines
- SERVER-REVIEW.md: ~550 lines
- phase1-implementation.md: ~650 lines
- coder-guidelines.instructions.md: ~550 lines
- copilot-instructions.md: ~200 lines

### Total Documentation

- ~2,200 lines of technical documentation
- 5 major deliverables
- 30+ features planned across 5 phases
- 6 detailed implementation tasks for Phase 1

---

## Contact & Support

- Issues: GitHub Issues (when repo public)
- Discussions: GitHub Discussions (when repo public)
- Documentation questions: See CONTRIBUTING.md (to be created)

---

**Last Updated**: December 17, 2025  
**Next Review**: March 2026 (or at v1.0 release)

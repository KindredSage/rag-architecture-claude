# Architecture Diagrams

## Master Agent Flow

```mermaid
flowchart TD
    A[User Query] --> B[classify_intent]
    B -->|domain, intent, complexity| ROUTE{_route_after_classification}

    ROUTE -->|"intent in (general, help, capabilities)\nor general domain + no entities"| RD[respond_directly]
    ROUTE -->|"actionable intent\n(query_data, generate_report, ...)"| C[select_agents]

    RD -->|"conversational response\n+ example queries"| END_DIRECT([END])

    C -->|agent_ids, strategy| D[execute_agents]
    D -->|sequential / parallel| E{Strategy?}
    E -->|sequential| F1[Agent 1]
    E -->|parallel| F1 & F2[Agent 2] & F3[Agent 3]
    F1 --> G[merge_results]
    F2 --> G
    F3 --> G
    G --> H([END])

    style B fill:#4CAF50,color:#fff
    style ROUTE fill:#FF5722,color:#fff
    style RD fill:#607D8B,color:#fff
    style C fill:#2196F3,color:#fff
    style D fill:#FF9800,color:#fff
    style G fill:#9C27B0,color:#fff
```

### Routing Logic

| Condition | Route |
|-----------|-------|
| `intent` is `general`, `help`, or `capabilities` | `respond_directly` |
| `primary_domain == "general"` + no entities + non-actionable intent | `respond_directly` |
| Any actionable intent (`query_data`, `generate_report`, `plot_chart`, `send_email`, `summarize`, `explore_schema`, `anomaly_check`, `export_data`) | `select_agents` |

---

## Trade Agent Sub-Graph (Schema-First)

```mermaid
flowchart TD
    START([START]) --> SA[schema_analyzer]
    SA -->|"schema_info:\nschema_text, sample_rows,\ncolumn types, partition keys"| TA[trade_analyst]
    TA -->|"trade_context:\ncolumn_mappings,\nasset_class, resolved_query"| CG{clarification_gate}

    CG -->|no ambiguity / HITL off| QA[query_analyst]
    CG -->|ambiguous + HITL on| WAIT_CLARIFY[WAIT: User Clarifies]
    WAIT_CLARIFY -->|user responds| QA

    QA -->|"parsed_intent:\nverified_values,\nunverified_values"| QP[query_planner]
    QP -->|"query_plan:\nstrategy, optimization_hints"| QB[query_builder]
    QB -->|generated_sql| QV[query_validator]

    QV --> RETRY_CHECK{_should_retry_or_execute}
    RETRY_CHECK -->|"valid"| HITL_GATE{sql_approval_gate}
    RETRY_CHECK -->|"invalid + retry_count <= 3"| QB
    RETRY_CHECK -->|"invalid + retries exhausted"| FFG[failure_feedback_gate]

    HITL_GATE --> HITL_CHECK{_should_proceed_after_hitl}
    HITL_CHECK -->|approved / modified| QE[query_executor]
    HITL_CHECK -->|rejected / expired| DA
    HITL_CHECK -->|"pending (stream mode)"| END_STREAM([END - streaming pause])

    QE --> EXEC_CHECK{_should_retry_after_execution}
    EXEC_CHECK -->|success| DA[details_analyzer]
    EXEC_CHECK -->|"error + retry_count <= 3"| QB
    EXEC_CHECK -->|"error + retries exhausted"| FFG

    FFG --> FFG_CHECK{_after_failure_feedback}
    FFG_CHECK -->|"user clarified (1st time)"| QA
    FFG_CHECK -->|"user clarified (2nd time)\n= loop guard"| DA
    FFG_CHECK -->|rejected / expired / HITL off| DA

    DA --> END_NODE([END])

    style SA fill:#FCE4EC
    style TA fill:#E8F5E9
    style CG fill:#FF9800,color:#fff
    style QA fill:#E3F2FD
    style QP fill:#FFF3E0
    style QB fill:#F3E5F5
    style QV fill:#E0F7FA
    style HITL_GATE fill:#FF9800,color:#fff
    style QE fill:#FFF9C4
    style DA fill:#E8EAF6
    style FFG fill:#E91E63,color:#fff
    style WAIT_CLARIFY fill:#E91E63,color:#fff
```

### Trade Agent Data Flow

```mermaid
flowchart LR
    subgraph "1. Schema (Ground Truth)"
        SA_OUT["schema_analyzer output"]
        SA_DATA["schema_text\nsample_rows_text\nsample_rows\ncolumn types\npartition/sorting keys"]
    end

    subgraph "2. Domain Mapping"
        TA_OUT["trade_analyst output"]
        TA_DATA["column_mappings\n(user term -> real column)\nrelevant_metrics\nresolved_query"]
    end

    subgraph "3. Intent Parsing"
        QA_OUT["query_analyst output"]
        QA_DATA["filters (real columns)\naggregations (real columns)\nverified_values\nunverified_values"]
    end

    subgraph "4. SQL Generation"
        QB_OUT["query_builder"]
        QB_DATA["Uses ALL upstream data:\nschema + mappings +\nverified/unverified warnings"]
    end

    SA_OUT --> TA_OUT
    TA_OUT --> QA_OUT
    QA_OUT --> QB_OUT

    style SA_OUT fill:#FCE4EC
    style TA_OUT fill:#E8F5E9
    style QA_OUT fill:#E3F2FD
    style QB_OUT fill:#F3E5F5
```

### Node Dependency Matrix

| Node | Reads from state | Writes to state |
|------|-----------------|-----------------|
| `schema_analyzer` | (none - first node) | `schema_info` |
| `trade_analyst` | `schema_info`, `user_query`, `intent_analysis` | `trade_context` |
| `query_analyst` | `schema_info`, `trade_context`, `user_query` | `parsed_intent` |
| `query_planner` | `schema_info`, `parsed_intent`, `trade_context` | `query_plan` |
| `query_builder` | `schema_info`, `trade_context`, `parsed_intent`, `query_plan` | `generated_sql`, `sql_parameters` |
| `query_validator` | `generated_sql`, `schema_info` | `validation_result`, `needs_retry`, `retry_count` |
| `query_executor` | `generated_sql`, `sql_parameters` | `query_results`, `needs_retry` |
| `failure_feedback_gate` | `retry_feedback`, `generated_sql`, `user_query` | `user_query` (augmented), resets `retry_count` |
| `details_analyzer` | `query_results`, `user_query` | `analysis`, `artifacts` |

---

## Failure Recovery Flow

```mermaid
flowchart TD
    QB1["query_builder\n(attempt 1)"] --> QV1["query_validator"]
    QV1 -->|invalid| QB2["query_builder\n(attempt 2)"]
    QB2 --> QV2["query_validator"]
    QV2 -->|invalid| QB3["query_builder\n(attempt 3)"]
    QB3 --> QV3["query_validator"]
    QV3 -->|"still invalid\n(retries exhausted)"| FFG["failure_feedback_gate"]

    FFG -->|HITL enabled| WAIT["WAIT: User provides feedback"]
    FFG -->|HITL disabled| DA_ERR["details_analyzer\n(error path)"]

    WAIT -->|"user clarifies"| RESET["Reset state:\nretry_count=0\nclear SQL/validation\naugment user_query"]
    WAIT -->|"user rejects/expires"| DA_ERR

    RESET --> QA_REDO["query_analyst\n(full re-assessment)"]
    QA_REDO --> QP_REDO["query_planner"] --> QB_REDO["query_builder\n(fresh attempt)"]

    DA_ERR --> END_ERR([END with error])

    style FFG fill:#E91E63,color:#fff
    style WAIT fill:#FF9800,color:#fff
    style RESET fill:#4CAF50,color:#fff
    style DA_ERR fill:#B71C1C,color:#fff
```

---

## Multi-Server Session Flow

```mermaid
sequenceDiagram
    participant U as User
    participant LB as Load Balancer
    participant S1 as Server 1
    participant S2 as Server 2
    participant PG as PostgreSQL
    participant R as Redis

    U->>LB: POST /execute (session_id=null)
    LB->>S1: Forward request
    S1->>PG: Create session + run
    S1->>R: Check cache, rate limit
    S1->>S1: Execute agent pipeline
    S1->>PG: Save results
    S1->>U: Response (session_id=abc)

    Note over U,PG: Later, same user, different server
    U->>LB: POST /execute (session_id=abc)
    LB->>S2: Forward request
    S2->>PG: Load session abc (update server_id to S2)
    S2->>PG: Load conversation history
    S2->>R: Check cache
    S2->>S2: Execute with history context
    S2->>PG: Save results
    S2->>U: Response (same session_id=abc)
```

---

## Agent Registry & Discovery

```mermaid
flowchart LR
    subgraph Registration["At Startup"]
        T[Trade Agent] -->|register| REG[(Agent Registry)]
        R[Reporting Agent] -->|register| REG
        A[Analytics Agent] -->|register| REG
        N[Your New Agent] -->|register| REG
    end

    subgraph Routing["At Runtime"]
        REG -->|routing context| MA[Master Agent]
        MA -->|select best| CHOSEN[Selected Agent]
    end

    style REG fill:#FFEB3B
    style MA fill:#4CAF50,color:#fff
```

---

## Security Model: Defense in Depth

```mermaid
flowchart TD
    SQL[Generated SQL] --> L1[Layer 1: Programmatic Check]
    L1 -->|blocked keywords, structure| L1R{Pass?}
    L1R -->|NO| BLOCK1[BLOCKED]
    L1R -->|YES| L2[Layer 2: LLM Semantic Check]
    L2 -->|column existence, types, logic| L2R{Pass?}
    L2R -->|NO| RETRY[Retry / Block]
    L2R -->|YES| L3[Layer 3: CH Session readonly=1]
    L3 -->|database-level enforcement| EXEC[Execute Query]

    style L1 fill:#F44336,color:#fff
    style L2 fill:#FF9800,color:#fff
    style L3 fill:#4CAF50,color:#fff
    style BLOCK1 fill:#B71C1C,color:#fff
```

---

## Human-in-the-Loop: API Interaction Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant MA as Master Agent
    participant TA as Trade Agent
    participant HITL as HITL Service
    participant PG as PostgreSQL

    U->>API: POST /execute (hitl.enabled=true)
    API->>MA: invoke master graph
    MA->>TA: invoke trade agent

    Note over TA: schema_analyzer -> trade_analyst -> query_analyst -> query_builder -> validator
    TA->>HITL: create_interrupt(SQL approval)
    HITL->>PG: INSERT agent_interrupts (status=pending)

    TA-->>MA: return {hitl_pending: {interrupt_id}}
    MA-->>API: return {status: waiting_human, interrupt}
    API-->>U: 200 {status: "waiting_human", interrupt_id: "int-123"}

    Note over U: User reviews the SQL
    U->>API: GET /interrupts/int-123
    API-->>U: {sql: "SELECT ...", type: "approval"}

    U->>API: POST /interrupts/int-123/resolve {action: "approved"}
    API->>HITL: resolve_interrupt(approved)
    HITL->>PG: UPDATE status=approved

    Note over U: User re-runs or graph resumes
    U->>API: POST /execute (same session, query continues)
    API->>MA: invoke (with approval in context)
    MA->>TA: execute query (approved SQL)

    Note over TA: query_executor -> details_analyzer
    TA-->>MA: return results
    MA-->>API: return {status: completed, answer: "..."}
    API-->>U: 200 {answer: "Desk Alpha leads..."}
```

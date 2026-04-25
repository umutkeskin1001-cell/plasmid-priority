# <Feature Name> — PRD

Status: draft | review | approved | shipped | archived
Owner: <human name>
Reviewers: <names>
Started: <YYYY-MM-DD>
Target: <YYYY-MM-DD or "no hard date">

---

## 1. One-Line Summary

*One sentence. What is this, for whom, with what benefit. If you need two sentences you don't understand it yet.*

<e.g., "Let admins bulk-export audit logs as CSV so they can satisfy SOC 2 evidence requests without a support ticket.">

## 2. Problem

*Who is hurting today, in what way, how often, and what is the cost?*

- **Who:** ...
- **What hurts:** ...
- **How often:** ...
- **Cost of not fixing:** ...

## 3. Goals

*Measurable outcomes. "Improve UX" is not a goal. "p95 dashboard load < 2s" is a goal.*

- [ ] ...
- [ ] ...
- [ ] ...

## 4. Non-Goals

*The most important section. What this deliberately does NOT do. Every PRD drifts because of an unwritten non-goal.*

- This does not ...
- This does not ...
- We explicitly decline to ...

## 5. User Stories

*3–5 concrete scenarios with steps a human could follow.*

### Story 1: <Persona> does <thing>
1. User goes to ...
2. User clicks ...
3. System responds with ...
4. User sees ...

### Story 2: ...

## 6. API / UI Surface

*Endpoints, schemas, screens, configs that change. Be specific — this is what Cascade will implement against.*

### Endpoints
| Method | Path | Request | Response | Auth |
|--------|------|---------|----------|------|
| POST | /api/x | `{ ... }` | `{ ok, data }` | admin |

### Schemas
```typescript
type X = { ... }
```

### Screens
- `/admin/audit-logs` — new route
- `/admin/settings` — adds "Audit Log" section

## 7. Constraints

*From AGENTS.md, from user context, from reality.*

- **AGENTS.md invariants:**
  - Testing ≥ 80% on `src/`
  - API shape `{ ok, data?, error? }`
  - No PII in logs
- **Stack constraints:** ...
- **Deadlines:** ...
- **Dependencies:** ...

## 8. Risks & Open Questions

*What we don't know yet. Better to write them down than guess silently.*

- **Risk:** ... **Mitigation:** ...
- **Open question:** ... **Need decision by:** ...

## 9. Acceptance Criteria

*"This is done when ..." — bulleted, checkable.*

- [ ] ...
- [ ] ...
- [ ] ...

---

## Sign-Off

| Role | Name | Date | Signature / LGTM |
|------|------|------|-------------------|
| Product | | | |
| Engineering | | | |
| Security | | | |

---

*Template version 1.0 — from [windsurf-unlocked/starter](https://github.com/OnlyTerp/windsurf-unlocked/tree/main/starter).*

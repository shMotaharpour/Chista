# ChistaAgent — پیاده‌سازی ایجنت در محیط داکری Kaggle

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** ساخت اسکلت پروژه ChistaAgent که یک AI Agent را در کانتینر داکر مبتنی بر ایمیج پایتون Kaggle اجرا می‌کند، با یک «lab» مستقل برای طراحی/بهینه‌سازی و پوشه docs برای قوانین و یافته‌های تحقیق.

**Architecture:** دو محیط جدا —
1. **Runtime (ایجنت):** کانتینر داکر بر پایه ایمیج پایتونی Kaggle (فقط کتابخانه‌های موجود در آن ایمیج + کد ایجنت). منطق ایجنت با ساختار context-engineering (system prompt، حافظه/کش context، ابزارها، حلقه agent loop).
2. **Lab (میزبانی/بهینه‌سازی):** روی ماشین میزبان، وابستگی آزاد (هر کتابخانه‌ای)، مسئول ارزیابی، بهینه‌سازی پرامپت/کش، و ساخت docs.

**Tech Stack:** Python (Kaggle image), Docker, pytest (lab), OpenRouter/LLM API, markdown docs.

---

## ساختار پیشنهادی

```
Chista/ChistaAgent/
├── agent/                  # کدی که داخل کانتینر Kaggle اجرا می‌شود
│   ├── core/
│   │   ├── loop.py         # حلقه agent (think → tool call → observe)
│   │   ├── context.py      # ساختار context-engineering (messages، pruning، حافظه)
│   │   └── tools.py        # ابزارهای ایجنت (registry + schema)
│   ├── prompts/            # system prompt و قالب‌ها
│   ├── entrypoint.py       # نقطه ورود داخل کانتینر
│   └── requirements.txt    # فقط کتابخانه‌های مجازِ داخل ایمیج Kaggle
├── docker/
│   ├── Dockerfile          # FROM kaggle python image
│   └── docker-compose.yml
├── lab/                    # مستقل، وابستگی آزاد
│   ├── evaluator.py        # سنجش خروجی/هزینه/توکن ایجنت
│   ├── optimizer.py        # بهینه‌سازی پرامپت و context
│   ├── runner.py           # اجرای کانتینر و جمع‌آوری نتایج
│   └── tests/
├── docs/                   # قوانین + یافته‌های تحقیق
│   ├── rules/              # قوانین تعریف‌شده پروژه
│   └── research/           # داکیومنت‌های تحقیق و توسعه
├── .gitignore
└── README.md
```

---

## Tasks

### Task 1: اسکلت پوشه‌ها و README
- Create: `agent/`, `lab/`, `docs/rules/`, `docs/research/`, `docker/`
- `README.md` موجود را با توضیح دو محیط به‌روز کن.
- Commit: `chore: project scaffold`

### Task 2: Dockerfile مبتنی بر ایمیج Kaggle
- Create: `docker/Dockerfile`
- پایه: ایمیج رسمی Kaggle Python (فرض پیش‌فرض: `gcr.io/kaggle-images/python:latest` — **سؤال باز:** نسخه دقیق/تگ ایمیج؟ "V163" را نسخه پایتون 3.6 فرض کردم؛ تأیید شود).
- نصب `agent/requirements.txt`، کپی کد، `ENTRYPOINT ["python", "agent/entrypoint.py"]`
- Test: `docker build` موفق و `docker run` سلامت (`entrypoint --check`).
- Commit: `feat: kaggle docker runtime`

### Task 3: هسته context-engineering
- Create: `agent/core/context.py` + tests
- شامل: کلاس `Context` (system + history + tool results)، قوانین pruning، بودجه توکن.
- Commit: `feat: context management core`

### Task 4: حلقه ایجنت + ابزارها
- Create: `agent/core/loop.py`, `agent/core/tools.py`
- حلقه: LLM call → tool dispatch → observe → repeat تا شرط توقف.
- Commit: `feat: agent loop and tools`

### Task 5: entrypoint و اتصال API
- Create: `agent/entrypoint.py` — خواندن config از env (API key، مدل)، اجرای حلقه.
- Commit: `feat: container entrypoint`

### Task 6: Lab — evaluator و runner
- Create: `lab/runner.py` (ساخت/اجرای کانتینر، جمع‌آوری متریک)، `lab/evaluator.py` (امتیاز، هزینه توکن).
- Commit: `feat: lab runner/evaluator`

### Task 7: docs اولیه
- Create: `docs/rules/000-project-rules.md` (قوانین فعلی این پیام) و `docs/research/000-context-engineering-notes.md`
- Commit: `docs: initial rules and research notes`

---

## Verification
- `docker build -t chista-agent .` → موفق
- `docker run chista-agent --check` → پیام سلامت
- `pytest lab/tests` → پاس
- پایان هر Task یک commit اتمی.

## Risks / Open Questions
- **تگ دقیق ایمیج Kaggle** و نسخه پایتون داخل آن (نیاز به تأیید) — روی وابستگی‌ها اثر دارد.
- دسترسی شبکه داخل کانتینر برای فراخوانی LLM API.
- نحوه تزریق API key (env / volume).

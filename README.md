# ChistaAgent

AI Agent برای بازی **Kaggriculture** — رقابت دوبازیکنه مزرعه‌داری در Kaggle.

## ساختار

```
ChistaAgent/
├── agent/      # کد ایجنت (main.py قابل ارسال به Kaggle)
├── lab/        # محیط مستقل طراحی/بهینه‌سازی (کتابخانه آزاد: kaggle-environments, ...)
├── docs/       # داکیومنت‌های پروژه
│   ├── README.md             # قوانین کامل بازی
│   ├── AGENT.md              # راهنمای ساخت/ارسال ایجنت
│   ├── kaggriculture-source.md  # سورس محیط بازی
│   ├── rules/                # قوانین تعریف‌شده پروژه
│   └── research/             # یافته‌های تحقیق و توسعه
└── docker/     # محیط اجرای ایجنت روی ایمیج پایتونی Kaggle
```

## دو محیط جدا

1. **Runtime (ایجنت)** — کانتینر داکر بر پایه ایمیج پایتونی Kaggle. ایجنت فقط با امکانات همان ایمیج کار می‌کند (خروجی: `main.py` با تابع `agent(obs)`).
2. **Lab** — روی میزبان، وابستگی آزاد. اجرای محیط بازی، ارزیابی، تحلیل ریپلی‌ها، بهینه‌سازی استراتژی.

## Quick start (lab)

```bash
pip install -U kaggle-environments
python -c "
from kaggle_environments import make
env = make('kaggriculture', debug=True)
env.run(['agent/main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"
```

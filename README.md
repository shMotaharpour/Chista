<div align="center">

# 🌾 ChistaAgent

**AI Agent for Kaggle's *Kaggriculture* — a two-player farming strategy competition**

**ایجنت هوش مصنوعی برای مسابقه *Kaggriculture* کگل — یک بازی کشاورزی دونفره استراتژیک**

*English* · *فارسی*

</div>

---

## 🇬🇧 English

### Overview

ChistaAgent is a competitive game-playing agent for **Kaggriculture**, a two-player
zero-sum farming competition hosted on Kaggle. Two agents farm adjacent plots of
land — planting crops, hiring crews, buying/selling on a dynamic market — and the
player with the higher **farming score** at the end of the season wins.

The project is named after **Chista**, the Avestan deity of wisdom and
righteousness.

### Strategy Philosophy

> **No RL, no deep learning.** Game dynamics are (largely) deterministic, so we
> lean on exact optimization instead: the strategy is built from classical
> operations-research techniques rather than learned from experience.

The v1 agent (*melon-core*) is a hand-crafted deterministic policy derived from
replay statistics — planting melons exclusively on the high-yield days 5–17,
crew-capped planting, batched hiring, and tranche-based selling to avoid
crashing the market price. It earns **$2.6k–13.7k per season** versus the
starter agent's $3.4k, and showed **8× improvement** over v1.1 in 5-seed evals.

### Repository Layout

```
ChistaAgent/
├── agent/          # Agent code — main.py is the Kaggle submittable (agent(obs))
├── lab/            # Independent design & optimization environment (host-side)
│   ├── opponents/  #   19 vendored top-player opponents (ladder reference)
│   ├── runner.py   #   Paired-seed match runner (sides swapped to cancel
│   │               #   first-mover advantage)
│   ├── ladder.py   #   Opponent strength ranking (sorting tournament)
│   ├── prices.py   #   Exact port of env market_price() — env-verified
│   ├── economics.py#   Crop / animal / land economics tables
│   └── sell_impact.py  # Price-crash curves per product
├── docs/           # Project documentation
│   ├── AGENT.md, kaggriculture-source.md   # Game mechanics docs
│   ├── rules/      # Project rules
│   └── research/   # Numbered findings:
│       ├── 000-strategy-plan.md        # Strategy plan
│       ├── 001-project-audit.md        # Project audit
│       ├── 003-crop-economics.md       # Crop economics
│       ├── 004-roadmap.md              # Roadmap & results
│       ├── 005-opponent-ladder.md      # Opponent ladder
│       └── 006-mechanics-verification.md # Mechanics verification
└── docker/         # Container setup
```

### Two Separated Environments

1. **Runtime (`agent/`)** — the Kaggle Python image. The agent only uses what
   that image provides. Output: a single `main.py` exposing `agent(obs)`.
2. **Lab (`lab/`)** — on the host with free dependencies. Runs the game
   environment, evaluates agents, analyzes replays, and optimizes strategy.

### Quick Start (Lab)

```bash
. .venv/bin/activate
python -m lab.runner --a starter --b random --seeds 1..5 --out lab/results
python -m lab.report lab/results/<run-dir>
python -m lab.ladder          # rank the 19 vendored opponents
```

### Roadmap

The full strategy plan (`docs/research/000-strategy-plan.md`) moves from the
current rule-based core toward:

1. ✅ **v1 rule-based melon-core** *(done)*
2. ⬜ MILP planting mix (HiGHS) + LP sell timing
3. ⬜ Terminal-value land-timing (DP) · paired tuning with a holdout league
4. ⬜ VRP-TW labor routing · opponent identification (NN-ID, 3 scenarios, CVaR)
5. ⬜ Policy distillation to <10 ms inference

### Status

| Item | Value |
|---|---|
| Competition | [Kaggriculture](https://www.kaggle.com/) (Kaggle) |
| Language | Python 3.11 |
| Current version | v1.2 (rule-based melon-core) |
| Sister project | [AgriOracle](https://github.com/shMotaharpour/AgriOracle) |

---

## 🇮🇷 فارسی

### نگاه کلی

**چیستاایجنت** یک ایجنت بازی‌کن برای مسابقه **Kaggriculture** در کگل است — یک
بازی کشاورزی دونفره و صفر-جمع. دو ایجنت در زمین‌های مجاور کشاورزی می‌کنند:
محصول می‌کارند، کارگر استخدام می‌کنند، و در بازارِ پویا خرید و فروش می‌کنند.
در پایان فصل، بازیکنی که **امتیاز کشاورزی** بالاتری داشته باشد برنده است.

نام پروژه برگرفته از **چیستا**، ایزدبانوی دانایی و راستکردار در اوستا است.

### فلسفه استراتژی

> **بدون یادگیری تقویتی، بدون یادگیری عمیق.** دینامیک بازی (تا حد زیادی)
> قطعی است؛ پس به‌جای یادگیری از تجربه، سراغ بهینه‌سازی دقیق می‌رویم:
> استراتژی بر پایه تکنیک‌های کلاسیک تحقیق در عملیات ساخته می‌شود، نه یادگیری.

ایجنت نسخه ۱ (*melon-core* — هسته هندوانه) یک سیاست قطعیِ دست‌ساز است که از
تحلیل آماری ریپلی‌ها به دست آمده: کاشت هندوانه فقط در روزهای پربازده ۵ تا ۱۷،
کاشت محدود به تعداد کارگر، استخدام دسته‌ای، و فروش به‌صورت ترانشی برای جلوگیری
از ریزش قیمت بازار. این ایجنت در هر فصل **۲٫۶ تا ۱۳٫۷ هزار دلار** کسب می‌کند
(ایجنت پایه: ۳٫۴ هزار دلار) و در ارزیابی ۵-سید **۸ برابر**ِ نسخه ۱٫۱ شد.

### ساختار مخزن

```
ChistaAgent/
├── agent/          # کد ایجنت — main.py همان فایل قابل‌ارسال به کگل است (agent(obs))
├── lab/            # محیط مستقل طراحی و بهینه‌سازی (سمت هاست)
│   ├── opponents/  #   ۱۹ حریف برترِ وندورشده (مرجع نردبان قدرت)
│   ├── runner.py   #   اجرای مسابقه با سیدهای جفتی (جابه‌جایی طرفین برای حذف
│   │               #   مزیت حرکت اول)
│   ├── ladder.py   #   رتبه‌بندی قدرت حریف‌ها (تورنمنت مرتب‌سازی)
│   ├── prices.py   #   پورت دقیق market_price() محیط — راستی‌آزمایی‌شده
│   ├── economics.py#   جداول اقتصاد محصول / دام / زمین
│   └── sell_impact.py  # منحنی ریزش قیمت برای هر محصول
├── docs/           # مستندات پروژه
│   ├── AGENT.md, kaggriculture-source.md   # مستندات مکانیک بازی
│   ├── rules/      # قوانین پروژه
│   └── research/   # یافته‌های شماره‌گذاری‌شده:
│       ├── 000-strategy-plan.md        # طرح استراتژی
│       ├── 001-project-audit.md        # ممیزی پروژه
│       ├── 003-crop-economics.md       # اقتصاد محصولات
│       ├── 004-roadmap.md              # نقشه راه و نتایج
│       ├── 005-opponent-ladder.md      # نردبان حریف‌ها
│       └── 006-mechanics-verification.md # راستی‌آزمایی مکانیک‌ها
└── docker/         # تنظیمات کانتینر
```

### دو محیط جدا از هم

1. **اجرا (`agent/`)** — ایمیج پایتون کگل. ایجنت فقط از امکانات همان ایمیج
   استفاده می‌کند. خروجی: یک فایل `main.py` با تابع `agent(obs)`.
2. **آزمایشگاه (`lab/`)** — روی هاست با وابستگی‌های آزاد. محیط بازی را اجرا
   می‌کند، ایجنت‌ها را ارزیابی می‌کند، ریپلی‌ها را تحلیل و استراتژی را بهینه می‌کند.

### شروع سریع (آزمایشگاه)

```bash
. .venv/bin/activate
python -m lab.runner --a starter --b random --seeds 1..5 --out lab/results
python -m lab.report lab/results/<run-dir>
python -m lab.ladder          # رتبه‌بندی ۱۹ حریف وندورشده
```

### نقشه راه

طرح کامل استراتژی (`docs/research/000-strategy-plan.md`) از هسته قاعده‌محورِ
فعلی به سمت موارد زیر حرکت می‌کند:

1. ✅ **نسخه ۱ قاعده‌محور، هسته هندوانه** *(انجام شد)*
2. ⬜ ترکیب کاشت با MILP (HiGHS) + زمان‌بندی فروش با LP
3. ⬜ زمان‌بندی خرید زمین با ارزش انتهایی (DP) · تنظیم جفتی با لیگ holdout
4. ⬜ مسیریابی کارگر VRP-TW · شناسایی حریف (NN-ID، ۳ سناریو، CVaR)
5. ⬜ تقطیر سیاست به استنتاج زیر ۱۰ میلی‌ثانیه

### وضعیت

| مورد | مقدار |
|---|---|
| مسابقه | [Kaggriculture](https://www.kaggle.com/) (کگل) |
| زبان | پایتون ۳٫۱۱ |
| نسخه فعلی | v1.2 (قاعده‌محور، هسته هندوانه) |
| پروژه خواهر | [AgriOracle](https://github.com/shMotaharpour/AgriOracle) |

---

<div align="center">

*Made with 🧠 and 🌽 on a GCP VM — determinism over stochasticity.*

*با 🧠 و 🌽 روی یک ماشین GCP — قطعیت بر تصادف.*

</div>

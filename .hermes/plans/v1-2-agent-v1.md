# Plan — V1.2 Rule-Based Agent v1 (پلن، منتظر تایید)

**Goal:** پیاده‌سازی `agent/main.py` — ایجنت rule-based مبتنی بر ثابت‌های اقتصادی V1.1. معیار پذیرش: ببرد starter را (delta ≥ +$200، p<0.05 در 18 بازی جفت‌شده) و وارد لدربرد 19 حریف شود تا جایگاهش مشخص شود.

**معماری:** تک‌فایل `agent/main.py` (قابل ارسال به Kaggle) — تابع `agent(obs)` خالص، بدون state خارجی بین نوبت‌ها (محدودیت env). تصمیم‌ها هر نوبت با اولویت ثابت. محاسبات سبک — هدف <10ms/نوبت.

---

## Step 1 — اسکلت ایجنت و priority dispatcher

**Files:** `agent/main.py`

- 1.1 ساختار: `agent(obs) → {"farmer": op, "hands": [ops], "market": orders}` 
- 1.2 اولویت‌بندی farmer op (هر نوبت فقط یکی):
  1. HARVEST اگر محصول رسیده
  2. WATER اگر گیاه تشنه (بونس پنجره مهم‌ترین)
  3. FEED/CARE حیوان گرسنه
  4. PLANT اگر بذر داریم و تایل خالی
  5. BUILD/PALCE حیوان جدید
  6. DIG علف‌هرز
  7. حرکت به نزدیک‌ترین تایل نیازمند اکشن (target selection ساده: BFS تا تایل هدف)
- 1.3 market orders (تا 10/نوبت، ترتیب مهم): HIRE → BUY (بذر/حیوان/زمین طبق بودجه) → SELL
- 1.4 Commit: `feat(agent): v1 skeleton with priority dispatcher`

## Step 2 — قواعد اقتصادی (از 003 بخش 6)

- 2.1 **نقدینگی اولیه (روز 1-6):** خرید و کاشت CARROT/WHEAT چرخه سریع
- 2.2 **توسعه (روز 2-8):** BUY_LAND NE→SW→SE طبق بودجه (بعد از نگه‌داشتن ذخیره بذر)؛ اولین GOOSE روز 2-4؛ COW از روز 5+ اگر پول > $600
- 2.3 **حیوانات:** GOOSE ماشین fertilizer — COLLECT_FERTILIZER هر روز؛ تخم آزادانه بفروش؛ خط گندم خوراک (2 تایل wheat دائمی)
- 2.4 **کاشت میانی:** MELON روی تایل‌های آزاد (بودجه > $300)، وگرنه CARROT
- 2.5 **کود:** FERTILIZE فقط MELON و STRAWBERRY (بونس ×2)؛ از خرید کود خودداری مگر مازاد
- 2.6 **فروش (قوانین safe pace):** WHEAT/EGG هر مقدار؛ MILK/STRAWBERRY/WOOL/MELON ≤ safe pace (7/7/19/51 در روز)؛ FERTILIZER مصرف خودت اول، فروش مازاد ≤ 53/day
- 2.7 **HIRE:** هر روز تا n که هزینه فیبوناچی ≤ 10% پول روز
- 2.8 **Terminal (روز 26+):** بدون کاشت MELON/STRAWBERRY؛ روز 29-30: فروش اجباری همه (حتی زیر قیمت)
- 2.9 Commit: `feat(agent): economics rules (crops, animals, land, hires, sells)`

## Step 3 — تست صحت اولیه

- 3.1 راند سریع (100 گام) بدون exception؛ خروجی action های معتبر (validate دستی چند نوبت)
- 3.2 زمان‌بندی: assert میانگین <10ms/نوبت
- 3.3 Commit: در صورت نیاز fixها

## Step 4 — ارزیابی رسمی

- 4.1 **ladder baseline:** `python -m lab.runner --a main --b starter --seeds 1..18` → باید starter را ببرد (معیار پذیرش)
- 4.2 اگر نبرد: تحلیل money-path/residue بازی‌های باخته → fix → تکرار (چرخه 1.3 رودمپ)
- 4.3 گزارش نتیجه → `docs/research/007-agent-v1-vs-starter.md`
- 4.4 Commit: `docs: agent v1 vs starter results`

## Step 5 — ورود به لدربرد حریف‌ها

- 5.1 بازی v1 مقابل 19 حریف (هر کدام 1 بازی، seed ثابت) → جایگاه اولیه ما
- 5.2 ثبت در `docs/research/005-opponent-ladder.md` (بخش ما)
- 5.3 Commit: `docs: agent v1 ladder position`

---

## Verification (definition of done)

- [ ] v1 > starter با معناداری آماری (18 paired seeds، p<0.05)
- [ ] <10ms per turn
- [ ] جایگاه در لدربرد 19 حریف ثبت شده
- [ ] تک‌فایل، قابل ارسال Kaggle (`main.py` با تابع agent)

## Risks / نکات

- محدودیت 1 اکشن farmer/نوبت — ترتیب اولویت حیاتی است؛ آب رساندن به همه تایل‌ها ممکن نیست با 1 farmer → HIRE حیاتی (هر hand اکشن مستقل)
- hands هم اکشن جدا دارند → dispatcher باید برای هر hand جدا تصمیم بگیرد (تابع مشترک `decide_unit`)
- حرکت: تایل‌ها دورند؛ مزارع پخش‌اند — target selection ساده (نزدیک‌ترین) کافی است؛ VRP کامل V5
- خطر رایج: deadlock حرکت (رفت‌وبرگشت بین دو هدف) → قاعده: اکشن روی تایل فعلی اگر ممکن، وگرنه نزدیک‌ترین نیاز

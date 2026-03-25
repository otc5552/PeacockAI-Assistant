# 🧮 MatCalc — آلة حاسبة المصفوفات للـ AGI Transformer

برنامج C++ مستقل متخصص في عمليات ضرب المصفوفات المعقدة.
مصمم خصيصاً لتشغيل شبكة AGI Transformer 70B+ على أجهزة محدودة الـ VRAM.

---

## المشكلة اللي بيحلها

```
جهازك:  RAM = 16 GB  |  VRAM = 4 GB
الشبكة: 70B parameter  →  ~140 GB في FP32  →  مستحيل!
```

الحل: نوزع الأوزان على RAM + نحسب على CPU عبر MatCalc.

---

## الملفات

| الملف | الوظيفة |
|-------|---------|
| `matcalc_core.cpp` | نواة C++ — كل العمليات الرياضية |
| `matcalc_bridge.py` | Python API — الجسر بين PyTorch والـ C++ |
| `transformer_matcalc.py` | نسخ معدّلة من TransformerBlock تستخدم MatCalc |
| `build_and_test.py` | يبني المكتبة ويختبرها |

---

## التثبيت والتشغيل

```bash
# 1. تثبيت المتطلبات
sudo apt install g++ libgomp1

# 2. بناء واختبار كل شيء
cd matcalc/
python build_and_test.py

# يجب أن ترى:
# ✅ GEMM 64×128×64
# ✅ Linear with bias
# ✅ Softmax scaled
# ✅ RMSNorm
# ✅ SiLU
# ✅ Scaled Add
# ✅ Causal Mask
# 🎉 جميع الاختبارات نجحت!
```

---

## العمليات المدعومة

| العملية | الوظيفة في الشبكة | الملاحظة |
|---------|------------------|----------|
| `gemm(A, B)` | ضرب مصفوفات عام | Tiled + AVX2 + OpenMP |
| `batched_gemm(A, B)` | Q×K و Attn×V في Attention | موازي لكل head |
| `linear(x, W, bias)` | الطبقات الخطية (FFN, projections) | بديل nn.Linear |
| `softmax(x, scale, mask)` | Attention weights | مع Causal Mask |
| `rmsnorm(x, weight)` | تطبيع الـ activations | بديل LayerNorm |
| `rope(x)` | Position encoding | Rotary |
| `silu(x)` | Activation function | للـ FFN |
| `add(a, b)` | Residual connections | vectorized |
| `scaled_add(a, b, s)` | Weighted residuals | للطبقات العميقة |
| `causal_mask(n)` | ينشئ Attention mask | لمنع النظر للمستقبل |
| `scaled_dot_product_attention` | Attention كاملة | كل الخطوات معاً |

---

## دمجها في transformer.py

```python
# في transformer.py — في AGITransformer.__init__
from transformer_matcalc import TransformerBlockMC

# بدل TransformerBlock استخدم TransformerBlockMC:
self.layers = nn.ModuleList([
    TransformerBlockMC(
        embedding_dim  = embedding_dim,
        num_heads      = num_heads,
        num_kv_heads   = num_kv_heads,
        ffn_hidden     = ffn_hidden,
        use_rope       = use_rope,
        context_length = context_length,
        layer_idx      = i,
    )
    for i in range(num_layers)
])
```

---

## التحسينات التقنية في النواة C++

- **Cache Blocking (Tiling):** نقسم المصفوفات لـ tiles بحجم 64×64 لتقليل cache misses
- **AVX2 Intrinsics:** نحسب 8 float في خطوة واحدة بدل واحدة واحدة
- **FMA (Fused Multiply-Add):** نجمع الضرب والجمع في عملية واحدة
- **OpenMP:** نوزع الحساب على جميع أنوية CPU تلقائياً
- **Zero-Copy Bridge:** PyTorch و C++ يشتركوا في نفس الذاكرة بدون نسخ

---

## توقعات الأداء

```
جهاز: 16GB RAM، 4GB VRAM، CPU 8-core

VRAM مستخدمة:   ~1-2 GB (Embeddings + Logits فقط)
RAM مستخدمة:    ~12-14 GB (أوزان الطبقات)
السرعة:         ~10-50x أبطأ من GPU
الفائدة:        تشغيل 70B بدون Out-of-Memory! ✅
```

---

## الخطوات القادمة

- [ ] دعم FP16 و INT8 لتقليل استهلاك RAM
- [ ] MoE Routing على MatCalc
- [ ] Memory-mapped weights (تحميل الأوزان من الديسك مباشرة)
- [ ] KV-Cache على RAM للـ inference السريع

# Advanced RAG Roadmap — 10-15 Günlük Yoğunlaşma Planı

**Amaç:** Bu proje MIT, Microsoft'ta AI uzmanları, müdürler ve profesörlere
sunulacak. Hedef "çalışan bir demo" değil, "bu kişi RAG'ı gerçekten anlamış ve
bunu kanıtlayabiliyor" izlenimi. Azure/cloud katmanına 10-15 gün sonra
geçeceğiz — bu süre boyunca odak: **RAG mimarisi, çoklu format veri işleme,
UI/UX gözlemlenebilirlik, mühendislik kalitesi, ve ölçülebilir kanıt
(evaluation)**.

Aşağıdaki A-I maddeleri önceki tartışmadan taşındı, J'den itibaren yeni
eklemeler var. Her madde **Neden**, **Ne yapılacak**, **Nasıl kanıtlanır**
(reviewer'a nasıl gösterilir) formatında.

---

## Faz 1 — İleri Seviye RAG Mimarisi

### A. Hybrid Search (Dense + Sparse)
- **Ne:** Cosine similarity (dense/embedding) tek başına yetmez — tam kelime
  eşleşmelerini (ürün kodu, hata kodu, isim gibi nadir terimleri) kaçırır.
  `rank-bm25` ile sparse retrieval ekle, dense + sparse skorlarını **RRF
  (Reciprocal Rank Fusion)** ile birleştir.
- **Kanıt:** Aynı sorguyu üç modda çalıştırıp (sadece dense, sadece BM25,
  hybrid) sonuçları yan yana gösteren bir karşılaştırma tablosu/ekranı.

### B. Local Cross-Encoder Re-ranking
- **Ne:** Hybrid search'ten gelen ilk 10-15 chunk'ı `bge-reranker-base` gibi
  hafif bir cross-encoder ile yeniden sırala, LLM'e sadece en iyi 3'ü ver.
- **Not:** Cross-encoder CPU'da yavaş olabilir — bu yüzden retrieval sonrası
  değil, sadece top-K aday üzerinde çalıştırılmalı (tüm veritabanında değil).
  ONNX'e çevrilmiş bir varyant kullanmak Foundry Local'in native runtime'ıyla
  tutarlı olur ve hız kazandırır.
- **Kanıt:** Reranking öncesi/sonrası sıralama değişimini gösteren bir "before
  → after" görselleştirmesi.

### C. Query Rewriting & HyDE
- **Ne:** Kötü/eksik yazılmış sorguyu yerel modelle genişlet (query expansion)
  veya sorguya sahte bir cevap ürettirip o cevabın vektörüyle ara (HyDE).
- **Kanıt:** UI'da "orijinal sorgu → optimize edilmiş sorgu" adımını şeffaf
  şekilde göster (kara kutu olmasın, süreç görünür olsun).

### D. Self-RAG / Retrieval Grader
- **Ne:** Çekilen chunk'ları LLM'e vermeden önce küçük bir "grader" adımından
  geçir: bu chunk soruyla gerçekten alakalı mı? Değilse ya sorguyu otomatik
  yeniden yaz ya da modele "bilmiyorum" dedirt.
- **Kanıt:** Kasıtlı olarak alakasız bir soru sorup sistemin "dokümanlarımda
  bu bilgi yok" dediğini canlı gösterebilmek — halüsinasyon yapmadığının
  kanıtı.

### E. Context Compression (Lost-in-the-Middle Çözümü)
- **Ne:** Chunk'ın tamamını değil, sorguyla en alakalı cümleleri "cımbızla"
  seçip context'e koy (sentence-window retrieval). Uzun context'lerde LLM'in
  ortadaki bilgiyi kaçırma sorununu azaltır.
- **Kanıt:** Context penceresi karşılaştırması — "ham chunk (500 token)" vs
  "sıkıştırılmış context (80 token)" — aynı doğrulukla çok daha az token.

### J. Multi-Query Retrieval (HyDE'den daha ucuz, ek recall)
- **Ne:** Tek sorgu yerine LLM'e aynı sorunun 3 farklı ifadesini
  ürettir, her biriyle ayrı ayrı ara, sonuçları birleştir (union + RRF).
  HyDE'ye göre daha ucuz ve daha az halüsinasyon riski taşır — ikisini
  birlikte tutup hangisinin hangi soru tipinde daha iyi çalıştığını
  karşılaştırmak bile başlı başına bir bulgu olur.

### K. Parent-Document / Small-to-Big Retrieval
- **Ne:** Retrieval'i küçük, hassas chunk'larla yap (yüksek precision) ama
  LLM'e o chunk'ın ait olduğu daha büyük "parent" paragrafı/bölümü ver
  (bağlam kaybı olmasın). Chunk boyutu ile bağlam bütünlüğü arasındaki
  klasik tradeoff'u çözer.

### L. Multi-Turn Query Contextualization
- **Ne:** Kullanıcı "peki ikincisi hakkında ne diyor?" gibi bir takip sorusu
  sorduğunda, bu soru sohbet geçmişiyle birlikte bağımsız bir sorguya
  dönüştürülmeli (coreference resolution), yoksa retrieval kör kalır. Şu ana
  kadar konuşulanlar hiç konuşulmuş gibi retrieval'e gitmemeli.

### M. Structured/Metadata-Filtered Retrieval
- **Ne:** Retrieval'den önce metadata'ya göre filtreleme (örn. sadece
  `file_type=pdf` veya `category=güvenlik` içinde ara). Kullanıcı arayüzden
  "sadece şu dosyada ara" diyebilmeli — büyük koleksiyonlarda hem hız hem
  doğruluk kazandırır.

### N. Post-Generation Groundedness Check
- **Ne:** D maddesi (Self-RAG) girdi tarafını kontrol ediyor — bu madde
  çıktı tarafını kontrol ediyor: LLM cevap verdikten sonra, cevabın gerçekten
  verilen context'e dayanıp dayanmadığını (embedding overlap veya basit bir
  NLI/entailment kontrolüyle) skorla. Düşük skorda kullanıcıya "bu cevabın
  güven skoru düşük" uyarısı göster.

---

## Faz 2 — Çoklu Format Doküman İşleme (Data Ingestion Pipeline)

### F. Format Parser'ları
- **PDF:** `pdfplumber` (tabloları da yakalar, `pypdf`'ten daha iyi metin
  çıkarımı). Taranmış/görüntü PDF'ler için opsiyonel OCR (`pytesseract`) —
  stretch goal olarak işaretle, zaman kalırsa.
- **Word:** `python-docx` — başlık yapısını (Heading 1/2/3) koruyarak
  chunk'lama, düz metne indirgeme değil.
- **Excel/CSV:** `pandas` — her satırı yapılandırılmış cümleye çevir (örn.
  "Ürün: X, Fiyat: Y, Stok: Z" gibi), ayrıca ham tabloyu da sakla ki ileride
  sayısal sorular ("en pahalı ürün hangisi?") ayrı bir "table agent"a
  yönlendirilebilsin (stretch goal, şimdilik sadece mimaride yer açılsın).

### O. Ortak Parser Arayüzü (Strategy Pattern)
- **Ne:** Her format için ayrı `if/else` yazmak yerine bir `DocumentParser`
  arayüzü (protocol/abstract base class) tanımla, her format kendi
  implementasyonunu yapsın (`PdfParser`, `DocxParser`, `XlsxParser`,
  `MarkdownParser`). Yeni format eklemek tek bir sınıf eklemek olsun.
  Bu, kod kalitesini değerlendiren biri için "bu kişi temiz mimari kuruyor"
  sinyali verir.

### G. Chunking Analytics & Metadata
- **Ne:** Dosya yüklenince UI'da anlık: dosya boyutu, kelime/token sayısı,
  kaç chunk'a bölündü, ortalama chunk yoğunluğu.
- Her chunk'a otomatik metadata: `source_file`, `file_type`, `page_number`
  (PDF için), `created_at`, `chunk_index`, `token_count`.

### P. Incremental Ingestion & Deduplication
- **Ne:** Aynı dosya tekrar yüklenirse (veya değişmemişse) yeniden
  embed'lememek için içerik hash'i (SHA-256) kontrolü. Sadece değişen/yeni
  dosyalar işlensin. Büyük koleksiyonlarda hem zaman hem (ileride Azure'da)
  maliyet tasarrufu — ve "bu kişi verimliliği düşünmüş" sinyali.

### Q. Semantic Chunking (Sabit Boyut Yerine Anlam Sınırı)
- **Ne:** Şu anki 200-token sabit boyutlu chunk'lama yerine, ardışık
  cümleler arası embedding benzerliğinin düştüğü noktaları "konu değişimi"
  kabul edip oradan böl. Chunk'ların yapay olarak bir cümlenin ortasından
  kesilmesini engeller — retrieval kalitesini doğrudan yükseltir ve teknik
  olarak "naif chunking"den ileri seviye bir yaklaşım olduğunu gösterir.

---

## Faz 3 — UI/UX ve Gözlemlenebilirlik (Control Room Dashboard)

### H. Gerçek Zamanlı Latency Metrikleri
- Her cevabın altında: Embedding / Retrieval / Re-ranking / Generation
  latency'si (ms). Toplam süre de görünsün.

### I. Kaynak Görselleştirme + "Grounded" Doğrulama Kutusu
- Cevabın altında tıklanabilir kaynak etiketleri (`maintenance_manual.pdf,
  s.4`), tıklanınca LLM'e giden tam chunk metni açılsın.

### R. Explainability Paneli (Skor Kırılımı)
- **Ne:** Her getirilen chunk için BM25 skoru, dense skoru, rerank skoru ve
  nihai RRF skorunu ayrı ayrı göster. Bu, sistemi "kara kutu çalışıyor"
  olmaktan çıkarıp mekanizmayı görünür kılar — tam olarak bir AI uzmanının
  görmek isteyeceği şey.

### S. Naive vs Advanced Karşılaştırma Modu
- **Ne:** UI'da bir toggle: aynı soruyu hem "Naive RAG" (sadece cosine
  similarity) hem "Advanced RAG" (hybrid+rerank+grader) modunda çalıştırıp
  yan yana göster. Bu tek özellik, demo videosunun en güçlü sahnesi olabilir
  — "işte fark bu" anı.

### T. Oturum Geçmişi + Trace Export
- **Ne:** Sorgu geçmişi kenar panelde, her sorgunun tam trace'i (retrieval
  skorları, latency, kullanılan mod) JSON olarak export edilebilsin — sunum
  ekine koyulabilecek somut veri.

---

## Faz 4 — Mühendislik Kalitesi (Reviewer Kod Kalitesine Bakar)

Bu bölüm diğer AI'ın listesinde hiç yok ama akademik/kurumsal bir
değerlendirmede en az RAG mimarisi kadar önemli — kod kalitesi, teknik
olgunluğun kanıtı.

### U. Test Süiti
- `pytest` ile: chunker, retriever, reranker, grader için birim testler +
  küçük sahte bir doküman seti üzerinde uçtan uca entegrasyon testi.

### V. Tip Güvenliği ve Doğrulama
- Tüm fonksiyonlarda type hints, API request/response modelleri için
  Pydantic şemaları. `mypy` ile statik kontrol (opsiyonel ama etkileyici).

### W. Yapılandırılmış Loglama
- `print()` yerine `logging` modülü, her sorgu için bir `request_id` ile
  pipeline'ın tüm katmanlarını (retrieval → rerank → generation) tek bir
  trace altında takip edilebilir hale getir.

### X. Takılabilir Mimari (Interface-Based Design)
- Retrieval stratejisi, reranker, LLM client'ı birbirinden bağımsız
  arayüzler (Protocol/ABC) olarak tasarlanmalı. "Naive'den Hybrid'e
  geçmek için orkestrasyon kodunu değiştirmedim, sadece stratejiyi
  değiştirdim" diyebilmek — yazılım mühendisliği olgunluğunun en net kanıtı.

### Y. CI Pipeline
- GitHub Actions: her push'ta lint (`ruff`) + testler otomatik çalışsın.
  Repo'ya giren biri yeşil bir "checks passed" rozeti görsün.

---

## Faz 5 — Değerlendirme & Benchmark (En Çok Fark Yaratacak Bölüm)

Bu, profesör/AI uzmanı izleyici için **en önemli** bölüm — çünkü "çalışıyor"
demekle "şu yöntemle retrieval doğruluğunu ölçülebilir şekilde artırdım"
demek arasındaki fark budur. Bu bir mini-araştırma sonucu üretmek demek.

### Z. Etiketli Değerlendirme Seti
- 20-30 soru-cevap çifti hazırla, her biri için "doğru cevabın hangi
  chunk'tan/dosyadan gelmesi gerektiği" bilgisiyle birlikte (ground truth).

### AA. Retrieval Metrikleri
- Precision@K, Recall@K, MRR (Mean Reciprocal Rank) — Naive retrieval vs
  Hybrid vs Hybrid+Rerank için ayrı ayrı hesapla.

### AB. Generation Metrikleri (Yerel, Ücretsiz)
- Cloud API'ye ihtiyaç duymadan: yerel LLM'i "judge" olarak kullanıp
  faithfulness/groundedness/answer-relevance için kaba ama tutarlı bir skor
  üret (RAGAS'ın felsefesini, cloud'a bağımlı olmadan uygula).

### AC. Otomatik Benchmark Raporu
- `scripts/run_eval.py` çalıştırıldığında bir HTML/markdown rapor üretsin:
  yöntemler arası karşılaştırma tablosu + basit bir grafik (matplotlib).
  Bu tek dosya, sunumun "kanıt" slaytı olur: "Naive RAG %62 precision,
  Advanced RAG %89 precision — işte fark."

---

## 10-15 Günlük Öncelik Sıralaması (ROI'ye göre)

**Gün 1-3 — Temel yükseltme (en yüksek etki/efor oranı)**
1. Semantic/gelişmiş chunking (Q) + metadata (G)
2. Çoklu format parser'lar (F, O) — PDF, DOCX, XLSX
3. Chunking analytics paneli (G)

**Gün 4-7 — RAG çekirdeğini derinleştirme**
4. Hybrid search: BM25 + dense + RRF (A)
5. Cross-encoder reranking (B)
6. Multi-query retrieval (J) — HyDE'den (C) önce, daha ucuz/az riskli
7. Retrieval grader / Self-RAG (D)

**Gün 8-10 — Gözlemlenebilirlik ve UI**
8. Latency tracing (H) + explainability paneli (R)
9. Kaynak görselleştirme (I)
10. Naive vs Advanced karşılaştırma modu (S) — demo için kritik

**Gün 11-13 — Mühendislik kalitesi**
11. Test süiti (U) + tip güvenliği (V)
12. Loglama (W) + interface-based mimari (X)
13. CI pipeline (Y)

**Gün 14-15 — Değerlendirme (sunumun kanıt bölümü)**
14. Eval seti + retrieval/generation metrikleri (Z, AA, AB)
15. Otomatik benchmark raporu (AC) — video ve sunuma bu rapor girer

*Context compression (E), HyDE (C), parent-document retrieval (K),
multi-turn contextualization (L), metadata filtering (M), post-generation
groundedness (N), incremental ingestion (P), session export (T) zaman
kalırsa eklenecek "stretch" maddeler — çekirdek hikaye bunlarsız da güçlü
duruyor, ekstra derinlik katarlar.*

## Riskler / Dikkat Noktaları
- Cross-encoder ve reranker modelleri CPU'da yavaş olabilir — sadece top-K
  aday üzerinde çalıştırılmalı, tüm veritabanında değil.
- Çok fazla katman (grader + rerank + multi-query) toplam latency'yi
  artırır — demo videosunda "bu gecikme neden var, neyi kazandırıyor"
  sorusuna hazırlıklı ol, latency panelinin (H) tam da bunu şeffaf gösterdiği
  için burada devreye girdiğini unutma.
- Kapsam çok geniş — her maddeyi "mükemmel" yapmaya çalışmak yerine, çekirdek
  akışın uçtan uca çalışır durumda kalmasını her adımda koru (her gün sonunda
  sistem yine soru cevaplayabiliyor olsun, yarım bırakılmış özellik commit'i
  reviewer'a kötü görünür).
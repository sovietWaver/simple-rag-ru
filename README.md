# MyRAG — локальный RAG для русскоязычных документов

RAG-пайплайн без высокоуровневых фреймворков. 
загрузка → чанкинг → эмбеддинги → ChromaDB → Ollama.

## Стек

| Компонент | Инструмент |
|---|---|
| LLM | Ollama (`gemma4:latest`), локально на CPU |
| Эмбеддинги | `BAAI/bge-m3` via sentence-transformers (1024-мерные, мультиязычные) |
| Векторная БД | ChromaDB (PersistentClient, косинусная метрика) |
| Форматы документов | `.pdf`, `.txt` |

## Требования

- Python 3.12+
- [Ollama](https://ollama.com) — запущена локально на `http://localhost:11434`
- Модель в Ollama: `gemma4:latest`

```powershell
ollama pull gemma4:latest
```

## Установка

```powershell
pip install sentence-transformers chromadb ollama pypdf langchain-text-splitters numpy
```

## Использование

### Индексация документа или папки

```powershell
python Main.py index .\docs\example.pdf
python Main.py index .\docs\          # рекурсивно, все .pdf и .txt
```

При первом запуске модель эмбеддингов (~2.5 ГБ) скачается автоматически.

### Вопрос по проиндексированным документам

```powershell
python Main.py ask "О чём говорится в разделе про настройку?"
python Main.py ask "Какова цель проекта?" --top-k 5
```

Пример вывода:

```
Prompt length: 2341 chars
Согласно документу, раздел про настройку описывает...

--- Sources ---
  example.pdf  chunk 4/31  dist=0.1823
  example.pdf  chunk 7/31  dist=0.2541
  example.pdf  chunk 5/31  dist=0.2904
```

`dist` — косинусное расстояние (0 = точное совпадение, < 0.3 — хорошее, > 0.7 — слабое).

### Сброс коллекции

Необходим при смене модели эмбеддингов или параметров чанкинга:

```powershell
python Main.py reset
python Main.py index .\docs\
```

## Структура проекта

```
MyRAG/
├── Main.py              # CLI: index / ask / reset
├── pyproject.toml
├── docs/                # папка для документов
└── rag/
    ├── loader.py        # .pdf / .txt → текст
    ├── chunker.py       # текст → чанки с метаданными
    ├── embedder.py      # чанки → векторы BAAI/bge-m3
    ├── store.py         # ChromaDB: добавление и сброс
    ├── searcher.py      # поиск + сборка контекста
    └── generator.py     # вызов Ollama, стриминг ответа
```

## Ключевые параметры

| Параметр | Значение | Где менять |
|---|---|---|
| `chunk_size` | 800 символов | `chunker.py` |
| `chunk_overlap` | 150 символов | `chunker.py` |
| `top_k` | 3 чанка | `--top-k` в CLI |
| `num_ctx` | 4096 токенов | `generator.py` — не повышать без перезагрузки модели |
| `temperature` | 0.2 | `generator.py` |

## Ограничения

- Модель `gemma4` загружена с контекстом 4096 токенов. При `num_ctx > 4096` Ollama попытается перезагрузить модель и упадёт с 503 при нехватке RAM.
- При изменении модели эмбеддингов или `chunk_size` — обязательно `python Main.py reset` перед переиндексацией, иначе векторы окажутся в разных пространствах.
- Сканированные PDF (без текстового слоя) не поддерживаются.
